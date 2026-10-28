import json
import pathlib

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


def make_ctf_record(verdict: str, flag: str | None = "FLAG{a}", domain: str = "ctf") -> dict:
    expected = {"verdict": verdict, "confidence": 0.9, "reasoning": "r", "disagreement": False}
    if flag:
        expected["flag"] = flag
    return {
        "id": f"rec-ctf-{verdict}",
        "domain": domain,
        "messages": [
            {"role": "user", "content": "solve this challenge"},
            {"role": "assistant", "content": json.dumps(expected)},
        ],
        "metadata": {},
    }


def test_expected_from_record_ctf_solved():
    expected = expected_from_record(make_ctf_record("solved", flag="FLAG{a}"))
    assert expected["verdict"] == "solved"
    assert expected["abstain"] is False
    assert expected["flag"] == "FLAG{a}"


def test_expected_from_record_ctf_abstains():
    expected = expected_from_record(make_ctf_record("insufficient_information", flag=None))
    assert expected["abstain"] is True
    assert expected["flag"] is None


def test_prediction_from_text_ctf_parses_verdict_and_flag():
    text = json.dumps({"verdict": "solved", "flag": "FLAG{a}"})
    prediction = prediction_from_text(text, domain="ctf")
    assert prediction["verdict"] == "solved"
    assert prediction["flag"] == "FLAG{a}"


def test_prediction_from_text_ctf_unparseable_is_miss():
    prediction = prediction_from_text("not json at all", domain="ctf")
    assert prediction["verdict"] is None
    assert prediction["flag"] is None


def test_score_task_ctf_matches_verdict_and_flag():
    pairs = [
        (
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
        (
            {"verdict": "solved", "abstain": False, "flag": "FLAG{bad}"},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
        (
            {"verdict": "solved", "abstain": False, "flag": "FLAG{x}"},
            {"verdict": "partial", "abstain": False, "flag": None},
        ),
        (
            {"verdict": "insufficient_information", "abstain": True, "flag": None},
            {"verdict": "insufficient_information", "abstain": True, "flag": None},
        ),
    ]
    result = score_task(pairs, domain="ctf")
    assert result["n_records"] == 4
    # verdict classes: solved==solved (even with a wrong flag), solved!=partial,
    # abstain==abstain -> 3/4; the wrong flag shows in flag_accuracy instead
    assert result["accuracy"] == 0.75
    assert result["flag_accuracy"] == 0.5
    assert result["false_negative_rate"] == 0.0
    assert result["false_positive_rate"] == 0.25
    assert result["abstain_recall"] == 1.0


def test_score_task_ctf_flag_accuracy_none_without_flags():
    pairs = [
        (
            {"verdict": "solved", "abstain": False, "flag": None},
            {"verdict": "solved", "abstain": False, "flag": None},
        ),
    ]
    result = score_task(pairs, domain="ctf")
    assert result["flag_accuracy"] is None


def test_run_task_eval_accepts_ctf():
    records = [make_ctf_record("solved", flag="FLAG{a}"), make_ctf_record("partial", flag=None)]
    replies = [
        json.dumps({"verdict": "solved", "flag": "FLAG{a}"}),
        json.dumps({"verdict": "partial"}),
    ]
    result = run_task_eval(records, generate=lambda messages: replies)
    assert result["accuracy"] == 1.0
    assert result["domains"] == ["ctf"]


def test_run_task_eval_rejects_mixed_domains():
    records = [make_record("confirmed", domain="security"), make_ctf_record("solved")]
    with pytest.raises(BrainforgeError, match="mixed"):
        run_task_eval(records, generate=lambda messages: ["{}"] * len(messages))


def test_evaluate_base_model_by_hub_id_skips_file_write(tmp_path, monkeypatch):
    import brainforge.training.task_eval as te

    split = tmp_path / "split.jsonl"
    split.write_text(json.dumps(make_record("confirmed", cwe="CWE-78")) + "\n", encoding="utf-8")
    stub_reply = json.dumps({"verdict": "confirmed", "cwe": "CWE-78"})
    monkeypatch.setattr(
        te, "_build_generator", lambda *a, **k: lambda messages: [stub_reply] * len(messages)
    )
    result = te.evaluate_model_on_records("Qwen/Qwen3-0.6B", split)
    assert result["accuracy"] == 1.0
    # a write would land in the CWD-relative Qwen/ directory, never tmp_path
    assert not pathlib.Path("Qwen").exists()


def test_evaluate_writes_task_eval_json_for_local_dir(tmp_path, monkeypatch):
    import brainforge.training.task_eval as te

    split = tmp_path / "split.jsonl"
    split.write_text(json.dumps(make_record("confirmed", cwe="CWE-78")) + "\n", encoding="utf-8")
    stub_reply = json.dumps({"verdict": "confirmed", "cwe": "CWE-78"})
    monkeypatch.setattr(
        te, "_build_generator", lambda *a, **k: lambda messages: [stub_reply] * len(messages)
    )
    result = te.evaluate_model_on_records(tmp_path, split)
    assert result["accuracy"] == 1.0
    assert (tmp_path / "task_eval.json").is_file()


def test_run_task_eval_rejects_domainless_records_in_ctf_runs():
    record = make_ctf_record("solved")
    del record["domain"]
    records = [make_ctf_record("partial"), record]
    with pytest.raises(BrainforgeError, match="without a domain"):
        run_task_eval(records, generate=lambda messages: ["{}"] * len(messages))


def test_score_task_ctf_solved_judge_with_partial_student_is_false_negative():
    pairs = [
        (
            {"verdict": "partial", "abstain": False, "flag": None},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
    ]
    result = score_task(pairs, domain="ctf")
    assert result["accuracy"] == 0.0
    assert result["false_negative_rate"] == 1.0


def test_score_task_ctf_valid_json_without_verdict_is_unparseable_miss():
    pairs = [
        (
            {"verdict": None, "abstain": False, "flag": "FLAG{x}"},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
    ]
    result = score_task(pairs, domain="ctf")
    assert result["accuracy"] == 0.0


def test_score_task_ctf_abstain_on_solved_judge_is_false_negative():
    pairs = [
        (
            {"verdict": "insufficient_information", "abstain": True, "flag": None},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
    ]
    result = score_task(pairs, domain="ctf")
    assert result["false_negative_rate"] == 1.0
    assert result["abstain_rate"] == 1.0


def test_conversations_one_per_prompt():
    from brainforge.training.task_eval import _conversations

    result = _conversations(["first prompt", "second prompt"])
    assert result == [
        [{"role": "user", "content": "first prompt"}],
        [{"role": "user", "content": "second prompt"}],
    ]


def test_generation_budget_warns_when_too_short(caplog):
    """A 512-token default truncates CTF answers of ~2000 tokens: every
    answer comes back cut and unparseable (accuracy 0 by construction).
    The run must warn before scoring against an impossible budget."""
    import json
    import logging

    from brainforge.training.task_eval import _budget_warning

    record = {
        "id": "rec-long",
        "domain": "ctf",
        "messages": [
            {"role": "user", "content": "challenge"},
            {"role": "assistant", "content": json.dumps({"verdict": "partial", "pad": "x" * 8000})},
        ],
    }
    with caplog.at_level(logging.WARNING):
        warning = _budget_warning([record], 512)
    assert warning is not None
    assert "512" in warning and "rec-long" in warning


def test_generation_budget_accepts_generous_limit():
    from brainforge.training.task_eval import _budget_warning

    record = {
        "id": "rec-short",
        "messages": [
            {"role": "user", "content": "challenge"},
            {"role": "assistant", "content": "x" * 100},
        ],
    }
    assert _budget_warning([record], 2048) is None
