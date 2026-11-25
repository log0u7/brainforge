"""OPD dense reward: the teacher's NLL of the student's completion tokens.

The Thinking Machines on-policy distillation formulation: the GRPO reward
is the dense per-token reverse-KL surrogate - the teacher's negative
log-likelihood of the student's completion, conditioned on the prompt.
Unlike the binary ctf_flag reward it is dense (every token contributes
signal) and it cannot be reward-hacked: the only way up is to match the
teacher.

Scoring runs against a llama.cpp server (the teacher): the student's
completion is tokenized with the teacher's own tokenizer (same Qwen3
vocab, ids align), then a growing-prefix chain of /completion calls with
``n_probs`` reads the teacher's next-token distribution at every
position; the student's token id is matched in the top-k (else the
min of the top-k is used as a lower bound).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from brainforge.errors import BrainforgeError
from brainforge.training.opd import make_teacher_kl_reward


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_empty_url_rejected():
    with pytest.raises(BrainforgeError):
        make_teacher_kl_reward("")


def test_reward_is_negative_mean_teacher_logprob(monkeypatch):
    """Prefix chain: /tokenize twice, then one /completion per token."""
    calls: list[dict] = []

    def _fake_urlopen(req: urllib.request.Request, timeout: int = 120):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body)
        if req.full_url.endswith("/tokenize"):
            content = body["content"]
            # "Hi" -> 2 tokens, "HiHello world" -> 5 tokens
            return _FakeResponse({"tokens": list(range(2 if content == "Hi" else 5))})
        # /completion: the teacher's top-k at this position; the student's
        # token (id 3 for prefix len 2, id 4 for len 3...) matches in top-k
        prefix_len = len(body["prompt"])
        student_id = 2 + prefix_len - 2  # completion ids are 2,3,4
        return _FakeResponse(
            {
                "completion_probabilities": [
                    {
                        "top_logprobs": [
                            {"id": student_id, "logprob": -0.5},
                            {"id": 999, "logprob": -9.0},
                        ]
                    }
                ]
            }
        )

    monkeypatch.setattr("brainforge.training.opd._urlopen", _fake_urlopen)
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    scores = reward(["Hello world"], prompts=["Hi"])
    # 3 completion tokens, all matched at -0.5 -> reward = 0.5
    assert scores == [pytest.approx(0.5)]
    completion_calls = [c for c in calls if "prompt" in c]
    assert len(completion_calls) == 3
    assert all(c["n_probs"] == 20 and c["cache_prompt"] for c in completion_calls)


def test_fallback_when_student_token_misses_topk(monkeypatch):
    def _fake_urlopen(req: urllib.request.Request, timeout: int = 120):
        body = json.loads(req.data.decode("utf-8"))
        if req.full_url.endswith("/tokenize"):
            content = body["content"]
            return _FakeResponse({"tokens": list(range(1 if content == "Hi" else 2))})
        return _FakeResponse(
            {"completion_probabilities": [{"top_logprobs": [{"id": 777, "logprob": -3.0}]}]}
        )

    monkeypatch.setattr("brainforge.training.opd._urlopen", _fake_urlopen)
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    scores = reward(["x"], prompts=["Hi"])
    # the student's token is not in the top-k -> min of the top-k (-3.0)
    assert scores == [pytest.approx(3.0)]


def test_conversational_prompts_need_a_renderer():
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    with pytest.raises(BrainforgeError):
        reward(["x"], prompts=[{"role": "user", "content": "q"}])


def test_grpo_wiring_requires_url():
    from brainforge.training.grpo import _resolve_teacher_kl

    with pytest.raises(BrainforgeError):
        _resolve_teacher_kl("teacher_kl", opd_teacher_url="")
    fn = _resolve_teacher_kl("teacher_kl", opd_teacher_url="http://127.0.0.1:9935")
    assert callable(fn)


def test_dataset_smoke_rows_exist():
    path = Path("datasets/grpo-mr22/train.jsonl")
    if not path.exists():
        pytest.skip("mr22 dataset not present on this machine")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    assert len(rows) >= 8
    assert all("messages" in row for row in rows[:8])
