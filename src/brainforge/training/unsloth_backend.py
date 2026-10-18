"""Unsloth training backend.

Same contract as :func:`brainforge.training.qlora.train_qlora`: a prepared
dataset directory in, adapter (or full weights) plus ``train_summary.json``
out. Unsloth's patched kernels give roughly 2x speed and lower VRAM than
the stock trl path. The ``unsloth`` extra is opt-in; without it the module
fails fast with an actionable error.

Heavy dependencies (unsloth, torch, trl, datasets) are imported lazily so
the package and the test suite run without the extra.
"""

import json
from pathlib import Path

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError
from brainforge.training.qlora import (
    _latest_checkpoint,
    _messages_dataset,
    _require_cuda,
    _sft_config,
)


def _require_unsloth() -> None:
    try:
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise BrainforgeError(
            "unsloth is not installed; install the unsloth extra first: uv sync --extra unsloth"
        ) from exc


def _load_in_4bit(config: TrainingConfig) -> bool:
    if config.quantization == "8bit":
        raise BrainforgeError(
            "unsloth backend supports 4bit or none quantization; "
            "use the trl backend for 8bit training"
        )
    return config.strategy == "qlora" and config.quantization == "4bit"


def train_unsloth(
    config: TrainingConfig,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
) -> dict:
    """Run Unsloth fine-tuning on a prepared dataset and save the adapter."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        resume = _latest_checkpoint(output_dir)
    _require_unsloth()
    _require_cuda()
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    dataset_dir = Path(dataset_dir)
    train_dataset = _messages_dataset(dataset_dir / "train.jsonl")
    validation_path = dataset_dir / "validation.jsonl"
    eval_dataset = _messages_dataset(validation_path) if validation_path.exists() else None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_length,
        dtype=None,
        load_in_4bit=_load_in_4bit(config),
    )
    if config.strategy == "qlora":
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_gradient_checkpointing="unsloth",
        )
    sft_args = _sft_config(
        config,
        output_dir,
        eval_dataset is not None,
        gradient_checkpointing=config.strategy == "full",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "base_model": config.base_model,
        "backend": "unsloth",
        "strategy": config.strategy,
        "quantization": config.quantization,
        "epochs": config.epochs,
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset) if eval_dataset else 0,
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps({**summary, "log_history": trainer.state.log_history}, indent=2),
        encoding="utf-8",
    )
    return summary
