"""GRPO training: reinforcement learning with verifiable rewards.

Uses TRL's GRPOTrainer with BrainForge reward functions:
- CTF flag verification (task reward, verified tier = exact match)
- compliance reward (GRPO-oblit realignment after abliteration)

Prompts come from a prepared dataset (train.jsonl, ``messages`` format):
the prompt is the rendered conversation minus the final assistant turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from brainforge.errors import BrainforgeError


def _load_flags(dataset_dir: Path) -> dict[str, str]:
    """Optional grpo_flags.jsonl sidecar: id -> expected flag."""
    sidecar = dataset_dir / "grpo_flags.jsonl"
    if not sidecar.exists():
        return {}
    flags: dict[str, str] = {}
    for line in sidecar.open(encoding="utf-8"):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("id") and entry.get("expected_flag"):
            flags[entry["id"]] = str(entry["expected_flag"])
    return flags


def _prompt_column(records: list[dict]) -> list[dict]:
    """GRPOTrainer expects a 'prompt' column (chat format accepted)."""
    rows: list[dict] = []
    for record in records:
        messages = record.get("messages") or []
        if not messages:
            continue
        prompt = list(messages)
        while prompt and prompt[-1].get("role") == "assistant":
            prompt.pop()
        if not prompt:
            continue
        rows.append({"id": record.get("id", ""), "prompt": prompt})
    return rows


def train_grpo(
    config: Any,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
    reward: str = "flag",
) -> dict:
    """GRPO fine-tune with verifiable rewards; saves the adapter."""
    from trl import GRPOConfig, GRPOTrainer

    from brainforge.training.hardware import plan_for_config
    from brainforge.training.qlora import _lora_config, _quantization_config
    from brainforge.training.rewards import compliance_reward, flag_reward

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.jsonl"
    records = [json.loads(line) for line in train_path.open(encoding="utf-8") if line.strip()]
    rows = _prompt_column(records)
    if not rows:
        raise BrainforgeError("empty dataset split")

    plan = plan_for_config(config)
    load_kwargs: dict[str, Any] = {}
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)

    flags = _load_flags(dataset_dir)

    reward_fn = flag_reward if reward == "flag" else compliance_reward

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        max_completion_length=min(512, config.max_length),
        model_init_kwargs=load_kwargs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        num_generations=max(2, min(8, config.batch_size * 2)),
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        bf16=True,
        logging_steps=1,
    )

    def _reward_with_flags(completions: list[str], **kwargs: Any) -> list[float]:
        ids = kwargs.get("id", [""] * len(completions))
        expected_flags = [flags.get(i, "") for i in ids]
        return reward_fn(completions, expected_flag=expected_flags)

    trainer = GRPOTrainer(
        model=config.base_model,
        args=grpo_config,
        train_dataset=rows,
        reward_funcs=[cast(Any, _reward_with_flags if reward == "flag" and flags else reward_fn)],
        peft_config=_lora_config(config),
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    summary = {
        "base_model": config.base_model,
        "backend": "trl-grpo",
        "reward": reward,
        "n_prompts": len(rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
