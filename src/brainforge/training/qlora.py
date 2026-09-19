"""QLoRA training, evaluation and export for the student model.

Heavy dependencies (torch, transformers, peft, trl, datasets) are imported
lazily so the package and the test suite run without the ``training`` extra.
All functions fail fast with a clear ``BrainforgeError`` when CUDA or the
extra is missing.
"""

import json
import math
from pathlib import Path

from brainforge.config.models import TrainingConfig
from brainforge.dataset.writer import read_jsonl
from brainforge.errors import BrainforgeError


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise BrainforgeError(
            "torch is not installed; install the training extra first: uv sync --extra training"
        ) from exc
    if not torch.cuda.is_available():
        raise BrainforgeError(
            "no CUDA device available; QLoRA training requires a GPU"
            " (see docs/how-to/register-gpu-runner.md)"
        )


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


def _messages_dataset(split_path: Path):
    from datasets import Dataset

    return Dataset.from_list(extract_messages(read_jsonl(split_path)))


def _quantization_config(quantization: str):
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


def _lora_config(config: TrainingConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_qlora(
    config: TrainingConfig, dataset_dir, output_dir, resume: bool | str = False
) -> dict:
    """Run QLoRA fine-tuning on a prepared dataset and save the adapter."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if not checkpoints:
            raise BrainforgeError(
                f"no checkpoint found in {output_dir} to resume from;"
                " run 'brainforge train run' without --resume first"
            )
        resume = checkpoints[-1]
    _require_cuda()
    from trl import SFTConfig, SFTTrainer

    dataset_dir = Path(dataset_dir)
    train_dataset = _messages_dataset(dataset_dir / "train.jsonl")
    validation_path = dataset_dir / "validation.jsonl"
    eval_dataset = _messages_dataset(validation_path) if validation_path.exists() else None

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=10,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        seed=config.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=config.base_model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quantization_config=_quantization_config(config.quantization),
        peft_config=_lora_config(config),
    )
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(output_dir))
    summary = {
        "base_model": config.base_model,
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


def evaluate(model_path, eval_dataset, quantization: str = "4bit") -> dict:
    """Compute eval loss and perplexity of a trained adapter on a dataset split."""
    _require_cuda()
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
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    quant_config = _quantization_config(quantization)
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    model = AutoPeftModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            inputs = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    result = {
        "model": str(model_path),
        "dataset": str(eval_dataset),
        "n_records": len(rows),
        "eval_loss": sum(losses) / len(losses),
        "perplexity": math.exp(sum(losses) / len(losses)),
    }
    (model_path / "eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def export(model_path, output_dir) -> dict:
    """Merge the LoRA adapter into the base model and save it standalone."""
    _require_cuda()
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(model_path), device_map="auto", torch_dtype="bfloat16"
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.save_pretrained(str(output_dir))
    return {"model": str(model_path), "output_dir": str(output_dir)}
