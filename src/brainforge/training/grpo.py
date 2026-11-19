"""GRPO training: reinforcement learning with verifiable rewards.

Uses TRL's GRPOTrainer with BrainForge reward functions resolved from
the generic registry or the domain pack's hooks (ADR-0022): the pack
declares its reward name, the dataset columns its reward consumes and
the optional sidecar carrying per-row ground truth.

Prompts come from a prepared dataset (train.jsonl, ``messages`` format):
the prompt is the rendered conversation minus the final assistant turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from brainforge.errors import BrainforgeError


def _prompt_column(records: list[dict], columns: tuple[str, ...] = ()) -> list[dict]:
    """GRPOTrainer expects a 'prompt' column (chat format accepted).

    The reward columns declared by the domain pack ride the row: TRL
    passes every extra dataset column to the reward function as kwargs.
    """
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
        row = {"id": record.get("id", ""), "prompt": prompt}
        for column in columns:
            if column in record:
                row[column] = record[column]
        rows.append(row)
    return rows


def _resolve_teacher_kl(
    spec: str,
    opd_teacher_url: str = "",
    base_model: str = "",
    max_scored_tokens: int = 0,
):
    """Resolve the teacher_kl (OPD dense reward) spec into a reward fn.

    The URL is config-driven and mandatory: nothing is silently defaulted.
    The chat-template renderer is the student's tokenizer (the reward must
    score the exact text the student generated from); it loads lazily at
    reward time, so unit tests run without the model.
    """
    if "teacher_kl" not in spec:
        return None
    if not opd_teacher_url:
        raise BrainforgeError(
            "teacher_kl reward in the GRPO spec requires "
            "training.opd_teacher_url (the llama.cpp teacher server to "
            "score against)"
        )
    from brainforge.training.opd import make_teacher_kl_reward

    render = None
    if base_model:
        from functools import lru_cache

        from transformers import AutoTokenizer

        @lru_cache(maxsize=1)
        def _load_tokenizer(name: str):
            return AutoTokenizer.from_pretrained(name)

        def render(messages: Any) -> str:
            return _load_tokenizer(base_model).apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    return make_teacher_kl_reward(
        opd_teacher_url, render=render, max_scored_tokens=max_scored_tokens
    )



def train_grpo(
    config: Any,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
    reward: str = "",
    domain: str | None = None,
) -> dict:
    """GRPO fine-tune with verifiable rewards; saves the adapter.

    Reward resolution order: explicit ``reward`` spec (registry name or
    weighted composition "name*w,name2") -> domain pack hook ->
    error. Nothing is silently defaulted: an unusable reward must fail
    before training starts.
    """
    from trl import GRPOConfig, GRPOTrainer

    from brainforge.domains.registry import get_or_generic
    from brainforge.training.hardware import plan_for_config
    from brainforge.training.qlora import _lora_config, _quantization_config
    from brainforge.training.rewards import compose_reward, get_reward

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.jsonl"
    records = [
        json.loads(line) for line in train_path.open(encoding="utf-8") if line.strip()
    ]

    domain = domain or getattr(config, "domain", "") or None
    pack = get_or_generic(domain) if domain else None
    columns = pack.grpo_dataset_columns() if pack else ()
    sidecar = pack.grpo_sidecar(dataset_dir) if pack else {}

    rows = _prompt_column(records, columns)
    if not rows:
        raise BrainforgeError("empty dataset split")

    plan = plan_for_config(config)
    # TRL create_model_from_path defaults dtype to float32; on a 10 GB card
    # that materializes tied embeddings + prefill logits in fp32 (~2.5 GB
    # each on an 8B) and OOMs at step 0. bf16 keeps the fit.
    load_kwargs: dict[str, Any] = {"device_map": plan.device, "dtype": "bfloat16"}
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)

    spec = reward or getattr(config, "grpo_reward", "")
    if not spec and pack:
        spec = pack.grpo_reward() or ""
    if not spec:
        raise BrainforgeError(
            "no GRPO reward: pass --grpo-reward (registry name or "
            "'name*w,...' composition), set training.grpo_reward, or use "
            "a domain pack that declares grpo_reward()"
        )
    opd_url = getattr(config, "opd_teacher_url", "")
    if "teacher_kl" in spec and "," in spec:
        # weighted composition including the OPD dense reward: the
        # registry does not hold teacher_kl (it needs the teacher URL
        # and the student tokenizer), so route it through a resolver
        def _teacher_kl_resolver(name: str):
            if name == "teacher_kl":
                return _resolve_teacher_kl(
                    name,
                    opd_url,
                    base_model=config.base_model,
                    max_scored_tokens=getattr(config, "opd_max_scored_tokens", 0),
                )
            return None

        reward_fn = compose_reward(spec, resolver=_teacher_kl_resolver)
    elif "teacher_kl" in spec:
        reward_fn = _resolve_teacher_kl(
            spec,
            opd_url,
            base_model=config.base_model,
            max_scored_tokens=getattr(config, "opd_max_scored_tokens", 0),
        )
    elif "," in spec or "*" in spec:
        reward_fn = compose_reward(spec)
    else:
        reward_fn = get_reward(spec)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        max_completion_length=(
            getattr(config, "grpo_max_completion_length", 0)
            or min(512, config.max_length)
        ),
        model_init_kwargs=load_kwargs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        num_generations=max(2, min(2, config.batch_size)),
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        bf16=True,
        # prepare_model_for_kbit_training upcasts lm_head to fp32 (~2.5 GB
        # on an 8B); full activations do not fit next to it on a 10 GB
        # card - checkpointing trades ~25% speed for the fit
        gradient_checkpointing=True,
        logging_steps=1,
        # SOTA recipe: DAPO clip-higher, Dr. GRPO unbiased baseline and no
        # KL penalty. TRL 1.12 takes scale_rewards as a str ("group" |
        # "batch" | "none"); the bool config False maps to "none".
        loss_type=config.grpo_loss_type,
        epsilon_high=config.grpo_epsilon_high,
        scale_rewards="none" if not config.grpo_scale_rewards else "group",
        beta=config.grpo_beta,
    )

    def _reward_with_sidecar(completions: list[str], **kwargs: Any) -> list[float]:
        ids = kwargs.get("id", [""] * len(completions))
        for key in sorted({k for row in sidecar.values() for k in row}):
            kwargs[key] = [sidecar.get(i, {}).get(key, "") for i in ids]
        return reward_fn(completions, **kwargs)

    from datasets import Dataset

    trainer = GRPOTrainer(
        model=config.base_model,
        args=grpo_config,
        train_dataset=Dataset.from_list(rows),
        reward_funcs=[cast(Any, _reward_with_sidecar if sidecar else reward_fn)],
        peft_config=_lora_config(config),
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    summary = {
        "base_model": config.base_model,
        "backend": "trl-grpo",
        "reward": spec,
        "n_prompts": len(rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
