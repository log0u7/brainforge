import json

import pytest

from brainforge.errors import BrainforgeError
from brainforge.training.task_eval import (
    expected_from_record,
    prediction_from_text,
    run_task_eval,
    score_task,
)


def make_record(verdict: str, cwe: str | None = "CWE-78", domain: str = "security") -> dict:
    expected = {"verdict": verdict, "confidence": 0.9, "reasoning": "r", "evidence": []}
    if cwe:
        expected["cwe"] = cwe
    return {
        "id": f"rec-{verdict}",
        "domain": domain,
        "messages": [
            {"role": "user", "content": "analyze this snippet"},
            {"role": "assistant", "content": json.dumps(expected)},
        ],
        "metadata": {},
    }


def test_expected_from_record_parses_judge_verdict():
    expected = expected_from_record(make_record("confirmed"))
    assert expected == {"vulnerability_found": True, "abstain": False, "cwe": "CWE-78"}


def test_expected_from_record_rejected():
    assert expected_from_record(make_record("rejected", cwe=None)) == {
        "vulnerability_found": False,
        "abstain": False,
        "cwe": None,
    }


def test_expected_from_record_marks_abstention():
    expected = expected_from_record(make_record("insufficient_information", cwe=None))
    assert expected["abstain"] is True
    assert expected["vulnerability_found"] is False


def test_prediction_from_text_parses_json():
    text = '```json\n{"verdict": "confirmed", "cwe": "CWE-79"}\n```'
    assert prediction_from_text(text) == {
        "vulnerability_found": True,
        "abstain": False,
        "cwe": "CWE-79",
    }


def test_prediction_from_text_unparseable_is_miss():
    assert prediction_from_text("garbage, no json here") == {
        "vulnerability_found": None,
        "abstain": False,
        "cwe": None,
    }


def test_prediction_from_text_marks_abstention():
    text = json.dumps({"verdict": "insufficient_information", "reasoning": "not enough info"})
    assert prediction_from_text(text)["abstain"] is True


def test_prediction_from_text_propagates_unexpected_errors(monkeypatch):
    from brainforge.training import task_eval

    def boom(_text: str) -> dict:
        raise ValueError("extract_json bug")

    monkeypatch.setattr(task_eval, "extract_json", boom)
    with pytest.raises(ValueError):
        prediction_from_text("x")


def test_generate_content_flattens_message_list():
    from brainforge.training.task_eval import _flatten_message

    assert _flatten_message("single") == "single"
    assert _flatten_message(["first", "second"]) == "first\n\nsecond"
    assert _flatten_message([]) == ""


def test_score_task_metrics():
    # 4 records: TP, TN, 1 FP, 1 FN
    pairs = [
        (
            {"vulnerability_found": True, "abstain": False, "cwe": "CWE-78"},
            {"vulnerability_found": True, "abstain": False, "cwe": "CWE-78"},
        ),
        (
            {"vulnerability_found": False, "abstain": False, "cwe": None},
            {"vulnerability_found": False, "abstain": False, "cwe": None},
        ),
        (
            {"vulnerability_found": True, "abstain": False, "cwe": "CWE-79"},
            {"vulnerability_found": False, "abstain": False, "cwe": None},
        ),
        (
            {"vulnerability_found": None, "abstain": False, "cwe": None},
            {"vulnerability_found": True, "abstain": False, "cwe": "CWE-78"},
        ),
    ]
    metrics = score_task(pairs)
    assert metrics["n_records"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["false_negative_rate"] == 0.25
    assert metrics["cwe_accuracy"] == 0.5
    assert metrics["abstain_recall"] is None


def test_score_task_counts_abstention():
    # judge abstained on one record; the student abstains too -> correct.
    # on the second, the student guesses instead of abstaining -> false positive.
    pairs = [
        (
            {"vulnerability_found": False, "abstain": True, "cwe": None},
            {"vulnerability_found": False, "abstain": True, "cwe": None},
        ),
        (
            {"vulnerability_found": True, "abstain": False, "cwe": "CWE-78"},
            {"vulnerability_found": False, "abstain": True, "cwe": None},
        ),
        (
            {"vulnerability_found": False, "abstain": True, "cwe": None},
            {"vulnerability_found": True, "abstain": False, "cwe": "CWE-78"},
        ),
    ]
    metrics = score_task(pairs)
    assert metrics["n_records"] == 3
    assert metrics["accuracy"] == 1 / 3
    assert metrics["false_positive_rate"] == 1 / 3
    assert metrics["false_negative_rate"] == 1 / 3
    assert metrics["abstain_recall"] == 0.5
    assert metrics["abstain_rate"] == 2 / 3


def test_run_task_eval_rejects_non_security():
    records = [make_record("confirmed", domain="coding")]
    with pytest.raises(BrainforgeError, match="domain"):
        run_task_eval(records, generate=lambda messages: ["{}"] * len(messages))


def test_run_task_eval_with_stub_generate():
    records = [make_record("confirmed", cwe="CWE-78"), make_record("rejected", cwe=None)]
    stub_reply = json.dumps({"verdict": "confirmed", "cwe": "CWE-78"})
    received = []
    result = run_task_eval(
        records,
        generate=lambda messages: received.extend(messages) or [stub_reply] * len(messages),
    )
    assert result["accuracy"] == 0.5
    assert result["n_records"] == 2
    # One batched call; each item is the per-record user-message list.
    assert received == [["analyze this snippet"], ["analyze this snippet"]]


def test_run_task_eval_preserves_record_order():
    records = [make_record("confirmed", cwe="CWE-78"), make_record("rejected", cwe=None)]

    def generate(messages: list) -> list[str]:
        return [
            json.dumps({"verdict": "confirmed", "cwe": "CWE-78"}),
            json.dumps({"verdict": "rejected"}),
        ]

    result = run_task_eval(records, generate)
    assert result["accuracy"] == 1.0
    assert result["false_positive_rate"] == 0.0
