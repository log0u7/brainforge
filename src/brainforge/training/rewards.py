"""Reward functions for GRPO training.

Pure functions: no torch, no model - they score generated strings so the
unit tests can run without the training extra (GRPOTrainer wires them in
at runtime).
"""

from __future__ import annotations

import re

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


def flag_reward(completions: list[str], expected_flag: list[str], **_: object) -> list[float]:
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


def compliance_reward(completions: list[str], **_: object) -> list[float]:
    """GRPO-oblit reward: 1.0 when the completion answers without refusing.

    Used AFTER abliteration: weight orthogonalization removes the refusal
    direction but models often keep verbal tics; the GRPO pass realigns
    them toward full compliance on benign prompts.
    """
    return [
        ZERO_REWARD if _REFUSAL_RE.search(_as_text(c)) else COMPLIANCE_REWARD for c in completions
    ]
