"""The GRPO lab-loop reward: the counterfactual pipeline behind the
lictor reward server (1.0 the suggestion was the difference, 0.5
redundant, 0.0 nothing landed, -0.25 the schema refused)."""

from __future__ import annotations

import json

import pytest

from brainforge.training.grpo import _prompt_column
from brainforge.training.rewards import get_reward

FAKE_RESPONSES = {
    "decisive": {"reward": 1.0, "detail": "the suggestion was the difference"},
    "redundant": {"reward": 0.5, "detail": "the pipeline lands without the suggestion"},
    "zero": {"reward": 0.0, "detail": "the suggestion did not make it fall"},
    "malformed": {"reward": -0.25, "detail": "suggest refused: unknown predicate"},
}


class _FakeHTTP:
    """Stands in for urllib.request.urlopen; records the payloads."""

    def __init__(self, response: dict):
        self.response = response
        self.payloads: list[dict] = []

    def __call__(self, request, timeout=0):
        self.payloads.append(json.loads(request.data.decode("utf-8")))
        body = json.dumps(self.response).encode("utf-8")
        import io

        resp = io.BytesIO(body)
        return resp


def test_prompt_column_keeps_the_lab_columns():
    record = {
        "id": "lab-ret2csu-001",
        "messages": [{"role": "user", "content": "propose"}],
        "port": 9992,
        "flag": "ROPE{a_placeholder_32byte_flag!}",
        "facts": 'nx("true").\n',
    }
    rows = _prompt_column([record])
    assert len(rows) == 1
    assert rows[0]["port"] == 9992
    assert rows[0]["flag"] == "ROPE{a_placeholder_32byte_flag!}"
    assert rows[0]["facts"] == 'nx("true").\n'


def test_prompt_column_without_lab_columns_stays_clean():
    rows = _prompt_column([{"id": "x", "messages": [{"role": "user", "content": "y"}]}])
    assert rows == [{"id": "x", "prompt": [{"role": "user", "content": "y"}]}]


def test_lab_loop_registered_and_decisive(monkeypatch):
    fake = _FakeHTTP(FAKE_RESPONSES["decisive"])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    lab_loop = get_reward("lab_loop")
    rewards = lab_loop(
        ['vuln("sof").'],
        port=[9992],
        flag=["ROPE{a_placeholder_32byte_flag!}"],
        facts=['nx("true").\n'],
    )
    assert rewards == [1.0]
    assert fake.payloads[0]["port"] == 9992
    assert fake.payloads[0]["completion"] == 'vuln("sof").'
    assert fake.payloads[0]["facts"] == 'nx("true").\n'


def test_lab_loop_batch_maps_each_completion(monkeypatch):
    fake = _FakeHTTP(FAKE_RESPONSES["redundant"])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    lab_loop = get_reward("lab_loop")
    rewards = lab_loop(
        ['vuln("sof").', "nothing useful"],
        port=[9999, 9999],
        flag=["FLAG{a}", "FLAG{a}"],
        facts=["f1.", "f2."],
    )
    assert rewards == [0.5, 0.5]
    assert [p["completion"] for p in fake.payloads] == ['vuln("sof").', "nothing useful"]


def test_lab_loop_missing_columns_are_zero_without_a_call(monkeypatch):
    calls: list[dict] = []

    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        calls.append(kw)
        raise AssertionError("the server must not be called")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    lab_loop = get_reward("lab_loop")
    assert lab_loop(["x"]) == [0.0]
    assert lab_loop(["x"], port=[9992]) == [0.0]
    assert lab_loop(["x"], flag=["F"], facts=["f."]) == [0.0]
    assert not calls


def test_lab_loop_server_error_is_zero(monkeypatch):
    import urllib.error

    def _boom(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    lab_loop = get_reward("lab_loop")
    rewards = lab_loop(["x"], port=[9992], flag=["F"], facts=["f."])
    assert rewards == [0.0]


def test_lab_loop_malformed_penalty_rides_through(monkeypatch):
    fake = _FakeHTTP(FAKE_RESPONSES["malformed"])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    lab_loop = get_reward("lab_loop")
    rewards = lab_loop(["x"], port=[9992], flag=["F"], facts=["f."])
    assert rewards == [-0.25]


@pytest.mark.parametrize(
    ("key", "expected"),
    [("decisive", 1.0), ("redundant", 0.5), ("zero", 0.0), ("malformed", -0.25)],
)
def test_lab_loop_semantics_table(monkeypatch, key, expected):
    fake = _FakeHTTP(FAKE_RESPONSES[key])
    monkeypatch.setattr("urllib.request.urlopen", fake)
    lab_loop = get_reward("lab_loop")
    assert lab_loop(["c"], port=[9992], flag=["F"], facts=["f."]) == [expected]
