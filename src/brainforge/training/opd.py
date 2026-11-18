"""OPD dense reward: the teacher's NLL of the student's completion tokens.

The Thinking Machines on-policy distillation formulation (on-policy
distillation as a dense reward inside GRPO): the reward is the teacher's
negative log-likelihood of the student's completion, conditioned on the
prompt. Unlike the binary ctf_flag reward it is dense (every token
contributes signal) and it cannot be reward-hacked: the only way up is
to match the teacher's distribution.

Scoring runs against a llama.cpp server (the teacher):
POST /v1/completions with ``echo: true`` and ``logprobs`` returns the
per-token logprobs of the echoed sequence; the first ``n_prompt_tokens``
belong to the prompt (counted via POST /tokenize on the rendered prompt),
the rest belong to the completion. Reward = -mean(logprob over the
completion tokens).

Heavy dependencies are not imported here: the module is pure HTTP.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from brainforge.errors import BrainforgeError
from brainforge.training.rewards import RewardFn


def _urlopen(req: urllib.request.Request, timeout: int = 60):
    return urllib.request.urlopen(req, timeout=timeout)


def _as_text(completion: object) -> str:
    """TRL passes completions as strings or assistant-message lists."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(
            str(part.get("content", ""))
            for part in completion
            if isinstance(part, dict) and part.get("role") == "assistant"
        )
    return str(completion)


def _render_prompt(messages: Any, render: Callable[[Any], str] | None) -> str:
    if isinstance(messages, str):
        return messages
    if render is None:
        raise BrainforgeError(
            "teacher_kl reward: conversational prompts need a chat "
            "template renderer (the student's tokenizer renders the same "
            "text the student generated from); pass render= or wire the "
            "reward via grpo (opd_tokenizer)"
        )
    return render(messages)


def make_teacher_kl_reward(
    teacher_url: str,
    render: Callable[[Any], str] | None = None,
) -> RewardFn:
    """Build the dense OPD reward against a llama.cpp teacher server.

    ``render`` turns chat-message prompts into the exact text the student
    generated from (the student's chat template). Plain string prompts
    skip the renderer; conversational prompts without one fail loud.
    """
    if not teacher_url:
        raise BrainforgeError(
            "teacher_kl reward: training.opd_teacher_url is required "
            "(the llama.cpp teacher server to score against)"
        )
    base = teacher_url.rstrip("/")

    def _post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _score_one(prompt: Any, completion: str) -> float:
        rendered = _render_prompt(prompt, render)
        full_text = rendered + completion
        n_prompt = len(_post("/tokenize", {"content": rendered})["tokens"])
        body = {
            "prompt": full_text,
            "echo": True,
            "logprobs": 1,
            "max_tokens": 1,
            "temperature": 0.0,
        }
        data = _post("/v1/completions", body)
        logprobs = data["choices"][0].get("logprobs") or []
        completion_lps = [
            entry["logprob"]
            for entry in logprobs[n_prompt:]
            if isinstance(entry, dict) and "logprob" in entry
        ]
        if not completion_lps:
            raise BrainforgeError(
                "teacher_kl reward: the teacher returned no logprobs for "
                "the completion tokens (echo+logprobs unsupported?)"
            )
        return -sum(completion_lps) / len(completion_lps)

    def _teacher_kl(completions: list, prompts: Any = None, **_: object) -> list[float]:
        prompt_list = prompts if prompts is not None else [""] * len(completions)
        return [
            _score_one(prompt, _as_text(completion))
            for prompt, completion in zip(prompt_list, completions, strict=True)
        ]

    return _teacher_kl
