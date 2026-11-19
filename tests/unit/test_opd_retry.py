"""Retry semantics for the OPD scoring chain: llama.cpp occasionally
serves a well-formed response without completion_probabilities (observed
once in ~800 calls); the call is retried once before failing loud."""

import json
import urllib.request

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


def _wire(monkeypatch, responses: list):
    """responses: per /completion call in order (tokenize calls always ok)."""
    calls = []

    def _fake_urlopen(req: urllib.request.Request, timeout: int = 60):
        body = json.loads(req.data.decode("utf-8"))
        if req.full_url.endswith("/tokenize"):
            return _FakeResponse({"tokens": list(range(len(body["content"])) )})
        calls.append(body)
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr("brainforge.training.opd._urlopen", _fake_urlopen)
    return calls


def _probs(student_id: int = 7) -> dict:
    return {
        "completion_probabilities": [
            {"top_logprobs": [{"id": student_id, "logprob": -1.0}]}
        ]
    }


def test_retry_recovers_from_missing_probabilities(monkeypatch):
    # 1 completion token; first /completion lacks probs, retry succeeds
    responses = [{"index": 0, "content": "\ufffd", "stop": True}, _probs()]
    _wire(monkeypatch, responses)
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    scores = reward(["x"], prompts=["p"])
    assert scores == [pytest.approx(1.0)]


def test_retry_exhausted_fails_loud(monkeypatch):
    bad = {"index": 0, "content": "\ufffd", "stop": True}
    responses = [bad, bad]
    _wire(monkeypatch, responses)
    reward = make_teacher_kl_reward("http://127.0.0.1:9935")
    with pytest.raises(BrainforgeError):
        reward(["x"], prompts=["p"])
