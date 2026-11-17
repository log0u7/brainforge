"""CTF GRPO rewards: the flag-oracle rewards of the CTF domain pack.

Pure functions: no torch, no model - they score generated strings so the
unit tests can run without the training extra. They live in the pack
(ADR-0022): the generic reward registry in ``brainforge.training.rewards``
holds the mechanics (compose_reward, signatures), the CTF content lives
here and registers itself under its historic names ("ctf_flag", "lab_loop").
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from brainforge.training.rewards import (
    ZERO_REWARD,
    _as_text,
    register_reward,
)

# CTF flag formats: name{...} - covers flag{}, FCSC{}, WQB{...} etc.
_FLAG_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{0,31}\{[^{}\s]{2,}\}")

FULL_REWARD = 1.0
FORMAT_REWARD = 0.3


def extract_flag(text: str) -> str | None:
    """First flag-shaped token in the completion, or None."""
    match = _FLAG_RE.search(text)
    return match.group(0) if match else None


def flag_reward(
    completions: list, expected_flag: list[str], **_: object
) -> list[float]:
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
register_reward("lab_loop", lab_loop_reward)
