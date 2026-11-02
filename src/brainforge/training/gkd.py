"""GKD training strategy: on-policy distillation against a local teacher.

Implements the ADR-0020 contract: the student trains on its own generations
(``lmbda=1.0``) against a local teacher's full next-token distribution
through TRL's ``GKDTrainer`` (Generalized Knowledge Distillation,
arXiv:2306.13649). The teacher must share the student's vocabulary: TRL
rejects a vocab mismatch at trainer construction, so the shared-tokenizer
discipline of ADR-0015's sidecar path becomes an enforced property instead
of a documented caveat.

Heavy dependencies (torch, trl, peft, datasets) are imported lazily inside
``train_gkd`` so the package and the unit test suite run without the
``training`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brainforge.errors import BrainforgeError


def train_gkd(
    config: Any,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
) -> dict:
    """On-policy distillation against a local teacher; saves the adapter.

    Nothing is silently defaulted: the teacher must be named
    (``training.gkd_teacher``) and must share the student's vocabulary (TRL
    enforces the vocab-size equality itself, so a mismatched teacher fails
    before the first training step).
    """
    teacher = getattr(config, "gkd_teacher", "")
    if not teacher:
        raise BrainforgeError(
            "no GKD teacher: set training.gkd_teacher (a local model sharing "
            "the student's tokenizer, e.g. Qwen/Qwen3-4B for a Qwen student)"
        )

    from brainforge.dataset.writer import read_jsonl

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(dataset_dir / "train.jsonl")

    from brainforge.training.qlora import extract_messages

    rows = extract_messages(records)

    from datasets import Dataset
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    from brainforge.training.hardware import plan_for_config
    from brainforge.training.qlora import _lora_config, _quantization_config

    plan = plan_for_config(config)
    load_kwargs: dict[str, Any] = {}
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)

    # The teacher rides its own quantization (4-bit on the 10GB host): TRL
    # loads it through AutoModelForCausalLM with these kwargs. dtype must be
    # "auto" - the trainer maps non-auto values through torch attributes and
    # a BitsAndBytesConfig already pins the compute dtype.
    teacher_load_kwargs: dict[str, Any] = {"dtype": "auto"}
    if plan.quantization != "none":
        teacher_load_kwargs["quantization_config"] = _quantization_config(plan.quantization)

    gkd_config = GKDConfig(
        output_dir=str(output_dir),
        teacher_model_name_or_path=teacher,
        teacher_model_init_kwargs=teacher_load_kwargs,
        lmbda=config.gkd_lmbda,
        beta=config.gkd_beta,
        seq_kd=config.gkd_seq_kd,
        max_new_tokens=min(512, config.max_length),
        model_init_kwargs=load_kwargs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        max_length=config.max_length,
        bf16=True,
        logging_steps=1,
    )

    trainer = GKDTrainer(
        model=config.base_model,
        teacher_model=teacher,
        args=gkd_config,
        # TRL rejects a plain list: the rows ride a Dataset
        train_dataset=Dataset.from_list(rows),
        peft_config=_lora_config(config),
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    summary = {
        "base_model": config.base_model,
        "backend": "trl-gkd",
        "strategy": "gkd",
        "gkd_teacher": teacher,
        "gkd_lmbda": config.gkd_lmbda,
        "gkd_beta": config.gkd_beta,
        "n_prompts": len(rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
