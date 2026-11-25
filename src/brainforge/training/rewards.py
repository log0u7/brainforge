"""Reward functions for GRPO training.

Pure functions: no torch, no model - they score generated strings so the
unit tests can run without the training extra (GRPOTrainer wires them in
at runtime).

Rewards are registered in a global registry so domain packs can expose
them (the CTF pack registers "ctf_flag" and "lab_loop" from
brainforge.domains.ctf.rewards) and training configs can compose several
weighted rewards: "ctf_flag*1.0,compliance*0.5".
"""

from __future__ import annotations

import re
from collections.abc import Callable

# refusal markers (abliteration realignment: a compliant answer scores,
# a refusal on a benign prompt is the failure mode we train away)
_REFUSAL_RE = re.compile(
    r"\b(i (?:cannot|can't|won't|am unable|apologize)|"
    r"i'm sorry,?(?: but)?|as an ai(?: language model)?\b)",
    re.IGNORECASE,
)

ZERO_REWARD = 0.0
COMPLIANCE_REWARD = 1.0

RewardFn = Callable[..., list[float]]

_REWARD_REGISTRY: dict[str, RewardFn] = {}


def register_reward(name: str, fn: RewardFn) -> None:
    _REWARD_REGISTRY[name] = fn


def get_reward(name: str) -> RewardFn:
    if name not in _REWARD_REGISTRY:
        raise KeyError(
            f"unknown GRPO reward '{name}' (known: {sorted(_REWARD_REGISTRY)})"
        )
    return _REWARD_REGISTRY[name]


def reward_names() -> list[str]:
    return sorted(_REWARD_REGISTRY)


def compose_reward(spec: str, resolver: Callable[[str], RewardFn | None] | None = None) -> RewardFn:
    """Compose weighted rewards: 'ctf_flag,compliance*0.5' (default weight 1).

    ``resolver`` lets a caller route names the registry does not hold
    (e.g. teacher_kl, whose reward fn needs the teacher URL and the
    student tokenizer): it is consulted first, and ``None`` falls back
    to the registry.
    """
    parts: list[tuple[RewardFn, float]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            name, weight_s = token.rsplit("*", 1)
            name = name.strip()
            weight = float(weight_s)
        else:
            name, weight = token, 1.0
        fn = resolver(name) if resolver else None
        if fn is None:
            fn = get_reward(name)
        parts.append((fn, weight))
    if not parts:
        raise KeyError(f"empty reward spec '{spec}'")

    def _composed(completions: list, **kwargs: object) -> list[float]:
        totals = [0.0] * len(completions)
        for fn, weight in parts:
            for i, value in enumerate(fn(completions, **kwargs)):
                totals[i] += weight * value
        return totals

    return _composed


def _as_text(completion: object) -> str:
    """TRL chat format passes a list of message dicts; flatten to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict) and message.get("role") == "assistant":
                parts.append(str(message.get("content", "")))
        return "\n".join(parts)
    return str(completion)


def compliance_reward(completions: list, **_: object) -> list[float]:
    """GRPO-oblit reward: 1.0 when the completion answers without refusing.

    Used AFTER abliteration: weight orthogonalization removes the refusal
    direction but models often keep verbal tics; the GRPO pass realigns
    them toward full compliance on benign prompts.
    """
    return [
        ZERO_REWARD if _REFUSAL_RE.search(_as_text(c)) else COMPLIANCE_REWARD
        for c in completions
    ]


register_reward("compliance", compliance_reward)
