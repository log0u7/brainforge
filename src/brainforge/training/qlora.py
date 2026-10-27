"""QLoRA training, evaluation and export for the student model.

Heavy dependencies (torch, transformers, peft, trl, datasets) are imported
lazily so the package and the test suite run without the ``training`` extra.
All functions fail fast with a clear ``BrainforgeError`` when CUDA or the
extra is missing.
"""

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brainforge.config.models import TrainingConfig
from brainforge.dataset.writer import read_jsonl
from brainforge.errors import BrainforgeError

if TYPE_CHECKING:
    from datasets import Dataset

    from brainforge.training.hardware import TrainingPlan

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _latest_checkpoint(output_dir: Path) -> Path | None:
    """Pick the highest-step ``checkpoint-<N>`` directory (numeric, not lexicographic)."""
    steps = []
    for path in output_dir.glob("checkpoint-*"):
        match = _CHECKPOINT_RE.match(path.name)
        if match is not None:
            steps.append((int(match.group(1)), path))
    if not steps:
        # crash-resilient: --resume with no checkpoints yet (fresh run dir)
        # starts from scratch instead of crashing - service supervisors
        # (nssm AppRestartDelay) must be safe on the very first boot
        return None
    return max(steps, key=lambda pair: pair[0])[1]


def extract_messages(records: list[dict]) -> list[dict]:
    """Keep only the chat messages of dataset records (TRL conversational format)."""
    if not records:
        raise BrainforgeError("empty dataset split")
    rows = []
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not messages:
            raise BrainforgeError(f"record at index {index} has no messages field")
        rows.append({"messages": messages})
    return rows


def _messages_dataset(split_path: Path) -> "Dataset | None":
    records = read_jsonl(split_path)
    if not records:
        # prepare() can legitimately emit a 0-record split on tiny datasets:
        # treat it like a missing file instead of failing the run (the heavy
        # datasets import stays lazy and is skipped entirely on this path).
        return None
    from datasets import Dataset

    return Dataset.from_list(extract_messages(records))


def _quantization_config(quantization: str) -> Any:
    if quantization == "auto":
        raise BrainforgeError(
            "quantization 'auto' reached the loader unresolved;"
            " resolve it with brainforge.training.hardware.resolve_quantization first"
        )
    import torch
    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _lora_config(config: TrainingConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _trainer_kwargs(
    config: TrainingConfig,
    output_dir: Path,
    has_eval: bool,
    plan: "TrainingPlan",
    gradient_checkpointing: bool = True,
) -> dict[str, Any]:
    """Training knobs shared by the trl, unsloth and kd backends."""
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "max_grad_norm": config.max_grad_norm,
        "gradient_checkpointing": gradient_checkpointing,
        "bf16": plan.dtype == "bfloat16",
        "fp16": plan.dtype == "float16",
        "logging_steps": 1,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": 10,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "seed": config.seed,
        "report_to": [],
    }


def _model_init_kwargs(plan: "TrainingPlan") -> dict:
    """Model load kwargs for the SFT trainer.

    ``device_map`` pins the resolved device: the default auto dispatch
    sends quantized modules to the CPU on a busy 10GB card (found by the
    first 8B QLoRA run on the RTX 3080).
    """
    return {
        "attn_implementation": plan.attn_implementation,
        "device_map": plan.device,
    }


def _sft_config(
    config: TrainingConfig,
    output_dir: Path,
    has_eval: bool,
    plan: "TrainingPlan",
    gradient_checkpointing: bool = True,
) -> Any:
    from trl import SFTConfig

    return SFTConfig(
        **_trainer_kwargs(config, output_dir, has_eval, plan, gradient_checkpointing),
        max_length=config.max_length,
        packing=config.packing,
        assistant_only_loss=config.assistant_only_loss,
        model_init_kwargs=_model_init_kwargs(plan),
    )


def train_qlora(
    config: TrainingConfig,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
) -> dict:
    """Run QLoRA/LoRA/full fine-tuning on a prepared dataset and save the adapter."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        checkpoint = _latest_checkpoint(output_dir)
    from brainforge.training.hardware import plan_for_config

    plan = plan_for_config(config)
    from trl import SFTTrainer

    dataset_dir = Path(dataset_dir)
    train_dataset = _messages_dataset(dataset_dir / "train.jsonl")
    if train_dataset is None:
        raise BrainforgeError("empty dataset split")
    validation_path = dataset_dir / "validation.jsonl"
    eval_dataset = _messages_dataset(validation_path) if validation_path.exists() else None

    sft_args = _sft_config(config, output_dir, eval_dataset is not None, plan)
    sft_kwargs: dict[str, Any] = {
        "model": config.base_model,
        "args": sft_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if plan.peft:
        if plan.quantization != "none":
            sft_kwargs["quantization_config"] = _quantization_config(plan.quantization)
        sft_kwargs["peft_config"] = _lora_config(config)
    trainer = SFTTrainer(**sft_kwargs)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(output_dir))
    summary = {
        "base_model": config.base_model,
        "backend": "trl",
        "strategy": config.strategy,
        "quantization": plan.quantization,
        "device": plan.device,
        "device_kind": plan.device_kind,
        "dtype": plan.dtype,
        "attn_implementation": plan.attn_implementation,
        "effective_strategy": plan.effective_strategy,
        "epochs": config.epochs,
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset or []),
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps({**summary, "log_history": trainer.state.log_history}, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate(
    model_path: Path | str,
    eval_dataset: Path | str,
    quantization: str = "auto",
    batch_size: int = 4,
    attn_implementation: str = "sdpa",
) -> dict:
    """Compute eval loss and perplexity of a trained adapter on a dataset split.

    Forward passes are batched (padded) and the reported ``perplexity`` is the
    true corpus perplexity: exp of the token-weighted mean negative log-likelihood.
    """
    from brainforge.training.hardware import plan_for_inference, torch_dtype

    plan = plan_for_inference(quantization, attn_implementation)
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model_path = Path(model_path)
    eval_dataset = Path(eval_dataset)
    rows = extract_messages(read_jsonl(eval_dataset))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.chat_template is None:
        raise BrainforgeError(
            f"tokenizer at {model_path} has no chat template; cannot evaluate chat records"
        )
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch_dtype(plan),
        "attn_implementation": attn_implementation,
    }
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)
    model = AutoPeftModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()

    tokenized: list[Any] = [
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        for row in rows
    ]
    loss_terms: list[Any] = []
    token_counts: list[int] = []
    with torch.no_grad():
        for start in range(0, len(tokenized), batch_size):
            padded = tokenizer.pad(
                tokenized[start : start + batch_size], padding=True, return_tensors="pt"
            )
            batch = {key: value.to(model.device) for key, value in padded.items()}
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            outputs = model(**batch, labels=labels)
            # HF loss is the mean over non-ignored shifted positions.
            n_tokens = int(batch["attention_mask"][:, 1:].sum())
            loss_terms.append(outputs.loss * n_tokens)
            token_counts.append(n_tokens)
    eval_loss = float(sum(loss_terms)) / sum(token_counts)
    result = {
        "model": str(model_path),
        "dataset": str(eval_dataset),
        "n_records": len(rows),
        "eval_loss": eval_loss,
        "perplexity": math.exp(eval_loss),
        "device": plan.device,
        "device_kind": plan.device_kind,
        "dtype": plan.dtype,
        "quantization": plan.quantization,
        "attn_implementation": plan.attn_implementation,
    }
    (model_path / "eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def export(model_path: Path | str, output_dir: Path | str) -> dict:
    """Merge the LoRA adapter into the base model and save it standalone."""
    from brainforge.training.hardware import plan_for_inference, torch_dtype

    plan = plan_for_inference("none")
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch_dtype(plan),
        attn_implementation=plan.attn_implementation,
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.save_pretrained(str(output_dir))
    return {"model": str(model_path), "output_dir": str(output_dir)}
