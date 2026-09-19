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
    assert expected == {"vulnerability_found": True, "cwe": "CWE-78"}


def test_expected_from_record_rejected():
    assert expected_from_record(make_record("rejected", cwe=None)) == {
        "vulnerability_found": False,
        "cwe": None,
    }


def test_prediction_from_text_parses_json():
    text = '```json\n{"verdict": "confirmed", "cwe": "CWE-79"}\n```'
    assert prediction_from_text(text) == {"vulnerability_found": True, "cwe": "CWE-79"}


def test_prediction_from_text_unparseable_is_miss():
    assert prediction_from_text("garbage, no json here") == {
        "vulnerability_found": None,
        "cwe": None,
    }


def test_score_task_metrics():
    # 4 records: TP, TN, 1 FP, 1 FN
    pairs = [
        (
            {"vulnerability_found": True, "cwe": "CWE-78"},
            {"vulnerability_found": True, "cwe": "CWE-78"},
        ),
        ({"vulnerability_found": False, "cwe": None}, {"vulnerability_found": False, "cwe": None}),
        (
            {"vulnerability_found": True, "cwe": "CWE-79"},
            {"vulnerability_found": False, "cwe": None},
        ),
        (
            {"vulnerability_found": None, "cwe": None},
            {"vulnerability_found": True, "cwe": "CWE-78"},
        ),
    ]
    metrics = score_task(pairs)
    assert metrics["n_records"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["false_negative_rate"] == 0.25
    assert metrics["cwe_accuracy"] == 0.5


def test_run_task_eval_rejects_non_security():
    records = [make_record("confirmed", domain="coding")]
    with pytest.raises(BrainforgeError, match="domain"):
        run_task_eval(records, generate=lambda messages: "{}")


def test_run_task_eval_with_stub_generate():
    records = [make_record("confirmed", cwe="CWE-78"), make_record("rejected", cwe=None)]
    stub_reply = json.dumps({"verdict": "confirmed", "cwe": "CWE-78"})
    received = []
    result = run_task_eval(
        records, generate=lambda messages: received.append(messages) or stub_reply
    )
    assert result["accuracy"] == 0.5
    assert result["n_records"] == 2
    # One generate() call per record, user message only.
    assert received == [["analyze this snippet"]] * 2
