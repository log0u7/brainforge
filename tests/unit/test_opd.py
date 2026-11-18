"""OPD dense reward: the teacher's NLL of the student's completion tokens.

The Thinking Machines on-policy distillation formulation: the GRPO reward
is the dense per-token reverse-KL surrogate - the teacher's negative
log-likelihood of the student's completion conditioned on the prompt,
scored against a llama.cpp server via echo + logprobs. Unlike the binary
ctf_flag reward it is dense (every token contributes signal) and it
cannot be reward-hacked: the only way up is to match the teacher.
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


def _server_logprobs(prompt_tokens: int, completion_lps: list[float]) -> dict:
    """llama.cpp /v1/completions shape: logprobs aligned with all tokens."""
    total = prompt_tokens + len(completion_lps)
    lps = [float("-1.0")] * prompt_tokens + completion_lps
    return {
        "choices": [
            {
                "text": "x" * total,
                "logprobs": [
                    {"token": f"t{i}", "logprob": lp} for i, lp in enumerate(lps)
                ],
            }
        ]
    }


def test_empty_url_rejected():
    with pytest.raises(BrainforgeError):
        make_teacher_kl_reward("")


def test_reward_is_negative_mean_teacher_logprob(monkeypatch):
    def _fake_urlopen(req: urllib.request.Request, timeout: int = 60):
        if req.full_url.endswith("/tokenize"):
            return _FakeResponse({"tokens": list(range(10))})
        return _FakeResponse(_server_logprobs(10, [-0.5, -2.0, -1.5]))

    monkeypatch.setattr("brainforge.training.opd._urlopen", _fake_urlopen)
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    scores = reward(["hello world"], prompts=["Hi"])
    # teacher logprobs for the 3 completion tokens: -0.5, -2.0, -1.5
    assert scores == [pytest.approx((0.5 + 2.0 + 1.5) / 3)]


def test_completion_token_boundary_skips_prompt(monkeypatch):
    def _fake_urlopen(req: urllib.request.Request, timeout: int = 60):
        if req.full_url.endswith("/tokenize"):
            return _FakeResponse({"tokens": list(range(7))})
        return _FakeResponse(_server_logprobs(7, [-3.0]))

    monkeypatch.setattr("brainforge.training.opd._urlopen", _fake_urlopen)
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    scores = reward(["the completion"], prompts=["user text"])
    assert scores == [pytest.approx(3.0)]


def test_conversational_prompts_need_a_renderer():
    with pytest.raises(BrainforgeError):
        make_teacher_kl_reward("http://127.0.0.1:9935")(
            ["c"], prompts=[{"role": "user", "content": "q"}]
        )


def test_conversational_completion_unwrapped(monkeypatch):
    def _fake_urlopen(req: urllib.request.Request, timeout: int = 60):
        if req.full_url.endswith("/tokenize"):
            return _FakeResponse({"tokens": list(range(4))})
        return _FakeResponse(_server_logprobs(4, [-1.0, -1.0]))

    monkeypatch.setattr("brainforge.training.opd._urlopen", _fake_urlopen)
    reward = make_teacher_kl_reward(
        "http://127.0.0.1:9935", render=lambda msgs: "USER: q\nASSISTANT:"
    )
    scores = reward(
        [[{"role": "assistant", "content": "answer"}]],
        prompts=[{"role": "user", "content": "q"}],
    )
    assert scores == [pytest.approx(1.0)]


def test_grpo_wiring_requires_url():
    """grpo resolves teacher_kl from the config: missing URL must fail loud."""
    from brainforge.training.grpo import _resolve_teacher_kl

    with pytest.raises(BrainforgeError):
        _resolve_teacher_kl("teacher_kl", opd_teacher_url="")
    fn = _resolve_teacher_kl("teacher_kl", opd_teacher_url="http://127.0.0.1:9935")
    assert callable(fn)


def test_dataset_smoke_rows_exist():
    """The smoke dataset is a subset of the mr22 GRPO dataset."""
    path = Path("datasets/grpo-mr22/train.jsonl")
    if not path.exists():
        pytest.skip("mr22 dataset not present on this machine")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    assert len(rows) >= 8
    assert all("messages" in row for row in rows[:8])
