import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.code_review.schemas import (
    CodeReviewTeacherAnalysis,
    CodeReviewVerdict,
    ReviewFinding,
)
from brainforge.types import Mode, RoleKind


@pytest.fixture
def review_case():
    return Case(
        id="pr-42",
        source=CaseSource(type="pull-request", repository="example-app", date="2026-08-15"),
        input=CaseInput(
            code=(
                "diff --git a/api.py b/api.py\n"
                "+def get_user(uid):\n"
                '+    return db.query(f"SELECT * FROM users WHERE id = {uid}")'
            ),
            description="PR #42: add a user lookup endpoint. The author was in a hurry.",
            question="Review this pull request for security issues.",
        ),
    )


def test_code_review_registered():
    assert "code_review" in pack_names()


def test_output_schemas():
    pack = get_pack("code_review")
    assert pack.output_schema(RoleKind.TEACHER) is CodeReviewTeacherAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is CodeReviewVerdict


def test_recommended_steps():
    pack = get_pack("code_review")
    assert pack.steps_for_mode(Mode.CHEAP) == ["review_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == ["review_teacher", "coding_teacher", "judge"]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "review_teacher",
        "coding_teacher",
        "general_teacher",
        "judge",
    ]


def test_teacher_prompts_contain_case_and_rag(review_case):
    pack = get_pack("code_review")

    class Chunk:
        text = "Team standard: all queries parameterized"
        source = "kb"

    teacher = pack.teacher_prompt("review_teacher", review_case, [Chunk()])
    assert "SELECT * FROM users" in teacher
    assert "Team standard" in teacher
    judge = pack.judge_prompt(review_case, [], {"review_teacher": {"summary": "s"}})
    assert "canonical review verdict" in judge


def test_teacher_gate_passes_valid():
    pack = get_pack("code_review")
    data = {
        "summary": "One high finding: SQL injection in the new endpoint",
        "findings": [
            {
                "severity": "high",
                "category": "injection",
                "location": "api.py:2",
                "issue": "f-string in SQL query",
                "suggested_fix": "use parameterized queries",
            }
        ],
        "overall_risk": "high",
        "confidence": 0.9,
        "reasoning": "User-controlled uid is interpolated into the query.",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert result.passed


def test_teacher_gate_rejects_missing_reasoning():
    pack = get_pack("code_review")
    data = {
        "summary": "s",
        "findings": [],
        "overall_risk": "none",
        "confidence": 0.9,
        "reasoning": "",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("reasoning" in reason for reason in result.reasons)


def test_judge_gate_passes_changes_requested():
    pack = get_pack("code_review")
    data = {
        "verdict": "changes_requested",
        "findings": [
            {
                "severity": "high",
                "category": "injection",
                "location": "api.py:2",
                "issue": "SQL injection",
                "suggested_fix": "parameterize",
            }
        ],
        "overall_risk": "high",
        "suggested_patch": "db.query('SELECT * FROM users WHERE id = ?', (uid,))",
        "confidence": 0.9,
        "reasoning": "Unparameterized query with user input.",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_judge_gate_rejects_changes_requested_without_findings():
    pack = get_pack("code_review")
    data = {
        "verdict": "changes_requested",
        "findings": [],
        "overall_risk": "none",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("findings" in reason for reason in result.reasons)


def test_judge_gate_rejects_approved_with_high_finding():
    pack = get_pack("code_review")
    data = {
        "verdict": "approved",
        "findings": [
            {
                "severity": "high",
                "category": "injection",
                "location": "api.py:2",
                "issue": "SQL injection",
                "suggested_fix": "parameterize",
            }
        ],
        "overall_risk": "high",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("approved" in reason for reason in result.reasons)


def test_student_input(review_case):
    pack = get_pack("code_review")
    text = pack.student_input(review_case)
    assert "Review this pull request" in text
    assert "SELECT * FROM users" in text


def test_finding_schema():
    finding = ReviewFinding(severity="medium", category="secrets", issue="hardcoded token")
    assert finding.location is None
    assert finding.suggested_fix is None
