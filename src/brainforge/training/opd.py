"""OPD dense reward: the teacher's NLL of the student's completion tokens.

The Thinking Machines on-policy distillation formulation (on-policy
distillation as a dense reward inside GRPO): the reward is the teacher's
negative log-likelihood of the student's completion, conditioned on the
prompt. Unlike the binary ctf_flag reward it is dense (every token
contributes signal) and it cannot be reward-hacked: the only way up is
to match the teacher's distribution.

Scoring runs against a llama.cpp server (the teacher). The student's
completion is tokenized with the teacher's own tokenizer (the Qwen3
family shares one vocab, so ids align), then a growing-prefix chain of
native /completion calls with ``n_probs`` reads the teacher's next-token
distribution at every position. The student's token id is matched in the
top-k; a miss falls back to the min of the top-k (a lower bound on the
true logprob - documented approximation, not a paper-grade KL).

Pure HTTP: no torch, no model - unit tests run without the training
extra.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from brainforge.errors import BrainforgeError


def _urlopen(req: urllib.request.Request, timeout: int = 60) -> Any:
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
            "text the student generated from); wire the reward via grpo "
            "(opd_tokenizer)"
        )
    return render(messages)


def make_teacher_kl_reward(
    teacher_url: str,
    render: Callable[[Any], str] | None = None,
    n_probs: int = 20,
    max_scored_tokens: int = 0,
) -> Callable[..., list[float]]:
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
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data

    def _tokenize(text: str) -> list[int]:
        tokens: list[int] = _post("/tokenize", {"content": text})["tokens"]
        return tokens

    def _score_one(prompt: Any, completion: str) -> float:
        rendered = _render_prompt(prompt, render)
        full_ids = _tokenize(rendered + completion)
        n_prompt = len(_tokenize(rendered))
        completion_ids = full_ids[n_prompt:]
        if not completion_ids:
            raise BrainforgeError(
                "teacher_kl reward: the completion tokenized to nothing "
                "(empty completion or a template/tokenizer mismatch)"
            )
        # cap the scored prefix (0 = all): the reward is the mean over the
        # scored tokens; the growing-prefix chain costs one /completion call
        # per scored token
        if max_scored_tokens:
            completion_ids = completion_ids[:max_scored_tokens]
        logprobs: list[float] = []
        for i, student_id in enumerate(completion_ids):
            prefix = full_ids[: n_prompt + i]
            body = {
                "prompt": prefix,
                "n_predict": 1,
                "n_probs": n_probs,
                "cache_prompt": True,
                "temperature": 0.0,
            }
            # llama.cpp occasionally serves a well-formed response without
            # completion_probabilities (once in ~800 calls); one retry
            # before failing loud
            for attempt in (0, 1):
                data = _post("/completion", body)
                top = data.get("completion_probabilities")
                if top:
                    break
                if attempt:
                    raise BrainforgeError(
                        "teacher_kl reward: the teacher returned no "
                        f"completion_probabilities at token {i}/{len(completion_ids)} "
                        f"(raw response: {json.dumps(data)[:300]})"
                    )
            if not top:
                raise BrainforgeError("teacher_kl reward: no completion_probabilities from teacher")
            top = top[0]["top_logprobs"]
            lp = next(
                (e["logprob"] for e in top if e["id"] == student_id),
                min(e["logprob"] for e in top),
            )
            logprobs.append(lp)
        return -sum(logprobs) / len(logprobs)

    def _teacher_kl(completions: list, prompts: Any = None, **_: object) -> list[float]:
        prompt_list = prompts if prompts is not None else [""] * len(completions)
        return [
            _score_one(prompt, _as_text(completion))
            for prompt, completion in zip(prompt_list, completions, strict=True)
        ]

    return _teacher_kl
