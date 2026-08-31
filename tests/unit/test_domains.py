import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_or_generic, get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.coding.schemas import CodingJudgeVerdict, CodingTeacherAnalysis
from brainforge.domains.generic import GenericAnalysis, GenericVerdict
from brainforge.domains.security.schemas import SecurityJudgeVerdict, SecurityTeacherAnalysis
from brainforge.errors import ConfigError
from brainforge.types import Mode, RoleKind


@pytest.fixture
def case():
    return Case(
        id="case-test",
        source=CaseSource(type="git", repository="repo", commit="abc", date="2026-07-01"),
        input=CaseInput(
            code="import subprocess\nsubprocess.run(user_input, shell=True)",
            description="A CLI helper",
            question="Find vulnerabilities",
        ),
        metadata={"language": "python"},
    )


def test_builtin_packs_registered():
    names = pack_names()
    assert "security" in names
    assert "coding" in names


def test_get_pack():
    assert get_pack("security") is not None
    assert get_pack("ghost") is None


def test_get_or_generic_known():
    pack = get_or_generic("security")
    assert pack.name == "security"


def test_get_or_generic_unknown_falls_back():
    pack = get_or_generic("kubernetes")
    assert pack.name == "kubernetes"
    assert isinstance(pack, type(get_or_generic("kubernetes")))


def test_security_output_schemas():
    pack = get_pack("security")
    assert pack.output_schema(RoleKind.TEACHER) is SecurityTeacherAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is SecurityJudgeVerdict


def test_coding_output_schemas():
    pack = get_pack("coding")
    assert pack.output_schema(RoleKind.TEACHER) is CodingTeacherAnalysis
    assert pack.output_schema(RoleKind.JUDGE) is CodingJudgeVerdict


def test_generic_output_schemas():
    pack = get_or_generic("custom")
    assert pack.output_schema(RoleKind.TEACHER) is GenericAnalysis
    assert pack.output_schema(RoleKind.JUDGE) is GenericVerdict


def test_recommended_steps_security():
    pack = get_pack("security")
    assert pack.steps_for_mode(Mode.CHEAP) == ["security_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == [
        "security_teacher",
        "coding_teacher",
        "judge",
    ]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "security_teacher",
        "coding_teacher",
        "general_teacher",
        "judge",
    ]


def test_unknown_mode_raises():
    pack = get_or_generic("custom")
    with pytest.raises(ConfigError, match="no recommended steps"):
        pack.steps_for_mode(Mode.CHEAP)


def test_security_prompts_contain_case_and_rag(case):
    pack = get_pack("security")

    class Chunk:
        text = "CWE-78 subprocess injection"
        source = "cwe-database"

    teacher = pack.teacher_prompt("security_teacher", case, [Chunk()])
    assert "subprocess.run" in teacher
    assert "CWE-78 subprocess injection" in teacher
    critic = pack.critic_prompt("general_teacher", case, [], {"security_teacher": {"summary": "s"}})
    assert "security_teacher" in critic
    judge = pack.judge_prompt(case, [], {"security_teacher": {"summary": "s"}})
    assert "canonical verdict" in judge
    assert pack.system_prompt("judge", RoleKind.JUDGE).startswith("You are the judge")


def test_security_gate_passes_valid_judge():
    pack = get_pack("security")
    data = {
        "verdict": "confirmed",
        "cwe": "CWE-78",
        "severity": "high",
        "confidence": 0.9,
        "evidence": [{"description": "shell=True with user input"}],
        "reasoning": "User input reaches subprocess.run with shell=True.",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_security_gate_rejects_confirmed_without_cwe():
    pack = get_pack("security")
    data = {
        "verdict": "confirmed",
        "confidence": 0.9,
        "evidence": [{"description": "x"}],
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("CWE" in reason for reason in result.reasons)


def test_security_gate_rejects_low_confidence():
    pack = get_pack("security")
    data = {
        "verdict": "rejected",
        "confidence": 0.2,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert not result.passed


def test_security_gate_rejects_invalid_cwe():
    pack = get_pack("security")
    data = {
        "vulnerability_found": True,
        "cwe": "CWE 78",
        "confidence": 0.9,
        "evidence": [{"description": "x"}],
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("CWE" in reason for reason in result.reasons)


def test_security_gate_rejects_vulnerability_without_evidence():
    pack = get_pack("security")
    data = {
        "vulnerability_found": True,
        "cwe": "CWE-78",
        "confidence": 0.9,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("evidence" in reason for reason in result.reasons)


def test_coding_gate_rejects_confirmed_without_issues():
    pack = get_pack("coding")
    data = {"verdict": "confirmed", "confidence": 0.9, "reasoning": "r", "disagreement": False}
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed


def test_coding_gate_rejects_teacher_without_behavior():
    pack = get_pack("coding")
    data = {"summary": "s", "confidence": 0.9, "reasoning": "r"}
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("behavior" in reason for reason in result.reasons)


def test_generic_gate():
    pack = get_or_generic("custom")
    assert pack.quality_gate(RoleKind.TEACHER, {"summary": "s", "confidence": 0.9}).passed
    assert not pack.quality_gate(RoleKind.TEACHER, {"confidence": 0.9}).passed


def test_student_input(case):
    pack = get_pack("security")
    text = pack.student_input(case)
    assert "Find vulnerabilities" in text
    assert "subprocess.run" in text
