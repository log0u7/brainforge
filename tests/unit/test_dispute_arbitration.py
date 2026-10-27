"""Arbitration-by-ground-truth tests: disagreements must train, not discard."""

from brainforge.pipeline.engine import arbitrate_by_ground_truth


def test_arbitration_accepts_when_correct_flag_in_teacher_output():
    teacher = {"ctf_teacher": {"verdict": "confirmed", "flag": "WQB{s0lved}"}}
    judge = {"verdict": "insufficient_information"}
    assert arbitrate_by_ground_truth(judge, teacher, "WQB{s0lved}") is True


def test_arbitration_accepts_case_insensitive_match():
    teacher = {"ctf_teacher": {"flag": "wqb{s0lved}"}}
    judge = {"verdict": "insufficient_information"}
    assert arbitrate_by_ground_truth(judge, teacher, "WQB{S0LVED}") is True


def test_arbitration_rejects_when_no_flag_matches():
    teacher = {"ctf_teacher": {"flag": "WQB{wrong}"}}
    judge = {"verdict": "insufficient_information"}
    assert arbitrate_by_ground_truth(judge, teacher, "WQB{right}") is False


def test_arbitration_uses_judge_reasoning_too():
    """The judge contested but quoted the right flag in its reasoning."""
    judge = {"verdict": "insufficient_information", "reasoning": "candidate WQB{maybe}"}
    assert arbitrate_by_ground_truth(judge, {}, "WQB{maybe}") is True


def test_arbitration_no_ground_truth_returns_false():
    """Tier-B: nothing to verify against - no arbitration, gate decides."""
    teacher = {"ctf_teacher": {"flag": "WQB{anything}"}}
    judge = {"verdict": "insufficient_information"}
    assert arbitrate_by_ground_truth(judge, teacher, "") is False
