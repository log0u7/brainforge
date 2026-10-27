"""Knowledge distillation training: SFT + KL to the teacher's top-k logprobs.

The dataset (``train.jsonl``) plus a ``kd.jsonl`` sidecar (produced by
``pipeline run --kd``, one record of per-token teacher top-k logprobs per
dataset id) are tokenized with the student's own chat template. Teacher
positions are mapped onto student tokens by character midpoint, so the
student and teacher may use different tokenizers. Top-k tokens that are
absent from the student vocabulary are dropped and the teacher distribution
is renormalized over the remaining tokens.

Loss per position: ``alpha * T^2 * CE(student_T, teacher_topk_T)
+ (1 - alpha) * CE(gold)``.

Experimental. Heavy dependencies (torch, transformers, datasets) are
imported lazily so the package and the unit test suite run without the
``training`` extra.
"""

import json
from pathlib import Path
from typing import Any

from brainforge.config.models import TrainingConfig
from brainforge.dataset.writer import read_jsonl
from brainforge.errors import BrainforgeError
from brainforge.training.hardware import torch_dtype
from brainforge.training.qlora import (
    _latest_checkpoint,
    _lora_config,
    _quantization_config,
    _trainer_kwargs,
)


def align_teacher_positions(
    teacher_positions: list[dict],
    content: str,
    content_start: int,
    full_text: str,
    tokenizer: Any,
) -> list[tuple[int, list[tuple[str, float]]]]:
    """Map teacher top-k positions onto student token indices by char midpoint.

    ``content_start`` is the char offset of the assistant content inside the
    fully rendered chat text. Each teacher token (tokens concatenate back to
    the content) is attributed to the student token whose char span contains
    its midpoint. Returns (student_token_index, topk) pairs; the index is the
    predictor position (token_index - 1) used by the KD loss.
    """
    if not teacher_positions:
        return []
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    spans: list[tuple[int, int]] = []
    offset = 0
    for token_id in input_ids:
        text = tokenizer.decode([token_id])
        spans.append((offset, offset + len(text)))
        offset += len(text)

    aligned: list[tuple[int, list[tuple[str, float]]]] = []
    teacher_offset = 0
    for position in teacher_positions:
        token = position.get("token", "")
        midpoint = content_start + teacher_offset + len(token) / 2
        teacher_offset += len(token)
        for index, (start, end) in enumerate(spans):
            if start <= midpoint < end:
                if index > 0:
                    aligned.append((index - 1, position.get("topk", [])))
                break
    return aligned


def compute_kd_loss(
    logits: Any,
    topk: list[tuple[str, float]],
    tokenizer: Any,
    temperature: float,
) -> Any | None:
    """Cross-entropy of the student's temperature-scaled distribution against
    the teacher's renormalized top-k distribution (None when no top-k token
    exists in the student vocabulary)."""
    import torch

    pairs: list[tuple[int, float]] = []
    vocab = tokenizer.get_vocab()
    for token, logprob in topk:
        token_id = vocab.get(token)
        if token_id is not None:
            pairs.append((token_id, logprob))
    if not pairs:
        return None
    student_logprobs = torch.log_softmax(logits / temperature, dim=-1)
    teacher_lps = torch.tensor(
        [logprob for _, logprob in pairs], device=logits.device, dtype=logits.dtype
    )
    teacher_probs = torch.softmax(teacher_lps / temperature, dim=-1)
    token_ids = torch.tensor([token_id for token_id, _ in pairs], device=logits.device)
    student_q = student_logprobs[token_ids]
    return -(teacher_probs * student_q).sum() * (temperature**2)


class KDDataCollator:
    """Pads input_ids/labels and passes per-row teacher targets through."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)
        pad_id = self.tokenizer.pad_token_id or 0
        batch: dict[str, Any] = {
            "input_ids": torch.full((len(features), max_length), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(features), max_length), dtype=torch.long),
            "labels": torch.full((len(features), max_length), -100, dtype=torch.long),
        }
        for row, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            batch["input_ids"][row, : len(ids)] = ids
            batch["attention_mask"][row, : len(ids)] = 1
            labels = torch.tensor(feature["labels"], dtype=torch.long)
            batch["labels"][row, : len(labels)] = labels
        batch["kd_targets"] = [feature["kd_targets"] for feature in features]
        return batch


class KDTrainer:
    """Mixin adding the distillation term to a transformers Trainer."""

    def __init__(
        self, *args: Any, kd_alpha: float = 0.5, kd_temperature: float = 2.0, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def compute_loss(
        self, model: Any, inputs: dict, return_outputs: bool = False, **kwargs: Any
    ) -> Any:
        kd_targets = inputs.pop("kd_targets")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        ce_loss = outputs.loss
        logits = outputs.logits
        kd_terms = []
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise BrainforgeError("KD trainer has no tokenizer; cannot align teacher tokens")
        for row, targets in enumerate(kd_targets):
            for token_index, topk in targets:
                if not topk:
                    continue
                term = compute_kd_loss(
                    logits[row, token_index], topk, tokenizer, self.kd_temperature
                )
                if term is not None:
                    kd_terms.append(term)
        if kd_terms:
            import torch

            kd_loss = torch.stack(kd_terms).mean()
            loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * ce_loss
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


def _tokenize_row(record: dict, teacher: dict, tokenizer: Any) -> dict:
    messages = record["messages"]
    if any(message.get("role") == "tool" for message in messages):
        raise BrainforgeError(
            f"dataset record '{record['id']}' contains tool messages; knowledge "
            "distillation supports single-turn records only"
        )
    content = messages[-1]["content"]
    if tokenizer.chat_template is None:
        raise BrainforgeError(
            "tokenizer for KD training has no chat template; cannot align teacher positions"
        )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    content_start = full_text.rfind(content)
    if content_start < 0:
        raise BrainforgeError(
            f"assistant content of record '{record['id']}' not found in the rendered chat"
        )
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(input_ids)
    offset = 0
    content_token_count = 0
    for index, token_id in enumerate(input_ids):
        text = tokenizer.decode([token_id])
        start, end = offset, offset + len(text)
        offset = end
        if start >= content_start:
            labels[index] = token_id
            content_token_count += 1
    kd_targets = align_teacher_positions(
        teacher["positions"], content, content_start, full_text, tokenizer
    )
    return {
        "id": record["id"],
        "input_ids": input_ids,
        "labels": labels,
        "kd_targets": kd_targets,
        "content_token_count": content_token_count,
    }


def train_kd(
    config: TrainingConfig,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
) -> dict:
    """Fine-tune with SFT + knowledge distillation from the teacher sidecar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        checkpoint = _latest_checkpoint(output_dir)
    dataset_dir = Path(dataset_dir)
    sidecar_path = dataset_dir / "kd.jsonl"
    if not sidecar_path.exists():
        raise BrainforgeError(
            f"KD training requires {sidecar_path}; regenerate the dataset with "
            "'brainforge pipeline run --kd'"
        )
    if config.packing:
        raise BrainforgeError("KD training does not support packing; set packing to false")
    from brainforge.training.hardware import plan_for_config

    plan = plan_for_config(config, kd=True)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    records = read_jsonl(dataset_dir / "train.jsonl")
    teacher_by_id = {entry["id"]: entry for entry in read_jsonl(sidecar_path)}

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_row(record: dict) -> dict:
        teacher = teacher_by_id.get(record["id"])
        if teacher is None:
            raise BrainforgeError(
                f"dataset record '{record['id']}' has no teacher positions in {sidecar_path}"
            )
        return _tokenize_row(record, teacher, tokenizer)

    train_rows = [build_row(record) for record in records]
    validation_path = dataset_dir / "validation.jsonl"
    eval_rows = (
        [build_row(record) for record in read_jsonl(validation_path)]
        if validation_path.exists()
        else []
    )
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype(plan)}
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)
    model: Any = AutoModelForCausalLM.from_pretrained(config.base_model, **load_kwargs)
    if plan.peft:
        from peft import get_peft_model

        model = get_peft_model(model, _lora_config(config))

    class _KDTrainer(KDTrainer, Trainer):  # type: ignore[misc]  # mixin: compute_loss signature differs across transformers versions
        pass

    trainer = _KDTrainer(
        model=model,
        args=TrainingArguments(
            **_trainer_kwargs(config, output_dir, eval_dataset is not None, plan)
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=KDDataCollator(tokenizer),
        processing_class=tokenizer,
        kd_alpha=config.kd_alpha,
        kd_temperature=config.kd_temperature,
    )
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(output_dir))
    summary = {
        "base_model": config.base_model,
        "backend": "kd",
        "strategy": config.strategy,
        "quantization": plan.quantization,
        "device": plan.device,
        "device_kind": plan.device_kind,
        "dtype": plan.dtype,
        "attn_implementation": plan.attn_implementation,
        "effective_strategy": plan.effective_strategy,
        "kd_alpha": config.kd_alpha,
        "kd_temperature": config.kd_temperature,
        "kd_positions": sum(len(row["kd_targets"]) for row in train_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(eval_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps({**summary, "log_history": trainer.state.log_history}, indent=2),
        encoding="utf-8",
    )
    return summary
