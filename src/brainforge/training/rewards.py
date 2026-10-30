"""Reward functions for GRPO training.

Pure functions: no torch, no model - they score generated strings so the
unit tests can run without the training extra (GRPOTrainer wires them in
at runtime).

Rewards are registered in a global registry so domain packs can expose
them (CtfPack -> "ctf_flag") and training configs can compose several
weighted rewards: "ctf_flag*1.0,compliance*0.5".
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable

# CTF flag formats: name{...} - covers flag{}, FCSC{}, WQB{...} etc.
_FLAG_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{0,31}\{[^{}\s]{2,}\}")

# refusal markers (abliteration realignment: a compliant answer scores,
# a refusal on a benign prompt is the failure mode we train away)
_REFUSAL_RE = re.compile(
    r"\b(i (?:cannot|can't|won't|am unable|apologize)|"
    r"i'm sorry,?(?: but)?|as an ai(?: language model)?\b)",
    re.IGNORECASE,
)

FULL_REWARD = 1.0
FORMAT_REWARD = 0.3
ZERO_REWARD = 0.0
COMPLIANCE_REWARD = 1.0

RewardFn = Callable[..., list[float]]

_REWARD_REGISTRY: dict[str, RewardFn] = {}


def register_reward(name: str, fn: RewardFn) -> None:
    _REWARD_REGISTRY[name] = fn


def get_reward(name: str) -> RewardFn:
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"unknown GRPO reward '{name}' (known: {sorted(_REWARD_REGISTRY)})")
    return _REWARD_REGISTRY[name]


def reward_names() -> list[str]:
    return sorted(_REWARD_REGISTRY)


def compose_reward(spec: str) -> RewardFn:
    """Compose weighted rewards: 'ctf_flag,compliance*0.5' (default weight 1)."""
    parts: list[tuple[RewardFn, float]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "*" in token:
            name, weight = token.rsplit("*", 1)
            parts.append((get_reward(name.strip()), float(weight)))
        else:
            parts.append((get_reward(token), 1.0))
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


def extract_flag(text: str) -> str | None:
    """First flag-shaped token in the completion, or None."""
    match = _FLAG_RE.search(text)
    return match.group(0) if match else None


def flag_reward(completions: list, expected_flag: list[str], **_: object) -> list[float]:
    """GRPO reward: exact expected flag = 1.0, valid format = 0.3, else 0.

    ``expected_flag`` arrives as a dataset column via GRPOTrainer kwargs;
    empty expected flag (tier-B, ground truth unknown) scores on format
    only - hallucinating a flag that parses still earns FORMAT_REWARD.
    """
    rewards: list[float] = []
    for completion, expected in zip(completions, expected_flag, strict=False):
        found = extract_flag(_as_text(completion))
        if found is None:
            # no flag-shaped token at all
            rewards.append(ZERO_REWARD)
        elif expected:
            # verified tier: ground truth known, wrong flag = zero
            match = found.strip().lower() == expected.strip().lower()
            rewards.append(FULL_REWARD if match else ZERO_REWARD)
        else:
            # tier-B: ground truth unknown, parseable flag = partial credit
            rewards.append(FORMAT_REWARD)
    return rewards


def compliance_reward(completions: list, **_: object) -> list[float]:
    """GRPO-oblit reward: 1.0 when the completion answers without refusing.

    Used AFTER abliteration: weight orthogonalization removes the refusal
    direction but models often keep verbal tics; the GRPO pass realigns
    them toward full compliance on benign prompts.
    """
    return [
        ZERO_REWARD if _REFUSAL_RE.search(_as_text(c)) else COMPLIANCE_REWARD for c in completions
    ]


# The lab-loop reward server's URL: the counterfactual pipeline (plan ->
# build -> run, raw facts vs suggest-enriched facts) lives on the
# operator's box, reachable through an ssh reverse tunnel.
LAB_SERVER_ENV = "GRPO_REWARD_SERVER_URL"
LAB_DEFAULT_URL = "http://127.0.0.1:9090/reward"


def lab_loop_reward(
    completions: list,
    port: list | None = None,
    flag: list | None = None,
    facts: list | None = None,
    **_: object,
) -> list[float]:
    """GRPO reward: the completion IS the suggest step, the lab is the judge.

    Each completion is POSTed to the reward server with its dataset
    columns (port, flag, facts): the server runs the closed loop twice
    (raw facts, then enriched) and returns the counterfactual reward.
    Missing columns or an unreachable server score zero - the loop
    never guesses.
    """
    if not (port and flag and facts):
        return [ZERO_REWARD for _ in completions]

    url = os.environ.get(LAB_SERVER_ENV, LAB_DEFAULT_URL)
    rewards: list[float] = []
    for completion, one_port, one_flag, one_facts in zip(
        completions, port, flag, facts, strict=False
    ):
        payload = json.dumps(
            {
                "port": one_port,
                "flag": one_flag,
                "facts": one_facts,
                "completion": _as_text(completion),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                rewards.append(float(json.loads(response.read())["reward"]))
        except (OSError, ValueError, KeyError, TimeoutError):
            # unreachable server, bad body, bad reward: the loop never guesses
            rewards.append(ZERO_REWARD)
    return rewards


register_reward("ctf_flag", flag_reward)
register_reward("compliance", compliance_reward)
register_reward("lab_loop", lab_loop_reward)
