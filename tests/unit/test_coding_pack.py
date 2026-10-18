import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.coding.schemas import CodingJudgeVerdict, CodingTeacherAnalysis
from brainforge.types import Mode, RoleKind


@pytest.fixture
def coding_case():
    return Case(
        id="coding-001",
        source=CaseSource(type="repo", repository="example-app", date="2026-09-01"),
        input=CaseInput(
            code="def parse(s): return s.split(',')",
            description="A parser function that splits CSV lines.",
            question="Explain what this code does.",
        ),
    )


def test_coding_registered():
    assert "coding" in pack_names()


def test_output_schemas():
    pack = get_pack("coding")
    assert pack.output_schema(RoleKind.TEACHER) is CodingTeacherAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is CodingJudgeVerdict


def test_recommended_steps():
    pack = get_pack("coding")
    assert pack.steps_for_mode(Mode.CHEAP) == ["coding_teacher", "judge"]


def test_system_prompt_kinds():
    pack = get_pack("coding")
    assert "software engineering teacher" in pack.system_prompt("coding_teacher", RoleKind.TEACHER)
    assert "judge" in pack.system_prompt("judge", RoleKind.JUDGE).lower()
    assert "devil" in pack.system_prompt("critic", RoleKind.CRITIC).lower()
    assert "security" in pack.system_prompt("security_teacher", RoleKind.TEACHER)


def test_teacher_prompt_and_critic_prompt(coding_case):
    pack = get_pack("coding")

    class Chunk:
        text = "split() is unsafe for CSV"
        source = "doc"

    teacher = pack.teacher_prompt("coding_teacher", coding_case, [Chunk()])
    assert "split()" in teacher
    security = pack.teacher_prompt("security_teacher", coding_case, [Chunk()])
    assert "split()" in security
    critic = pack.critic_prompt("critic", coding_case, [], {})
    assert "split(" in critic


def test_judge_prompt(coding_case):
    pack = get_pack("coding")
    text = pack.judge_prompt(coding_case, [], {"coding_teacher": {"summary": "s"}})
    assert "canonical verdict" in text.lower()


def test_quality_gate_low_confidence():
    pack = get_pack("coding")
    result = pack.quality_gate(
        RoleKind.TEACHER,
        {"confidence": 0.1, "behavior": "b", "reasoning": "r"},
    )
    assert not result.passed
    assert any("confidence" in reason for reason in result.reasons)


def test_quality_gate_missing_summary_and_reasoning():
    pack = get_pack("coding")
    result = pack.quality_gate(RoleKind.TEACHER, {"confidence": 0.9, "behavior": "b"})
    assert not result.passed
    assert any("missing summary or reasoning" in reason for reason in result.reasons)


def test_quality_gate_teacher_missing_reasoning():
    pack = get_pack("coding")
    result = pack.quality_gate(
        RoleKind.TEACHER, {"confidence": 0.9, "behavior": "b", "summary": "s"}
    )
    assert not result.passed
    assert any("missing reasoning" in reason for reason in result.reasons)


def test_quality_gate_judge_missing_reasoning():
    pack = get_pack("coding")
    result = pack.quality_gate(
        RoleKind.JUDGE, {"confidence": 0.9, "verdict": "confirmed", "issues": []}
    )
    assert not result.passed
    assert any("missing reasoning" in reason for reason in result.reasons)


def test_student_input(coding_case):
    pack = get_pack("coding")
    text = pack.student_input(coding_case)
    assert "Explain what this code does." in text
    assert "split" in text
