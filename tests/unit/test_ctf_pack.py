import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.ctf.schemas import CtfTeacherAnalysis, CtfVerdict
from brainforge.types import Mode, RoleKind


@pytest.fixture
def ctf_case():
    return Case(
        id="case-pwn-001",
        source=CaseSource(type="ctfd", repository="example-ctf", date="2026-08-15"),
        input=CaseInput(
            code="readelf -h ./chall\nProgram header: GNU_STACK RW",
            description="Pwn challenge: a 64-bit ELF with a gets() call and no canary.",
            question="Solve this CTF challenge.",
        ),
    )


def test_ctf_registered():
    assert "ctf" in pack_names()


def test_output_schemas():
    pack = get_pack("ctf")
    assert pack.output_schema(RoleKind.TEACHER) is CtfTeacherAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is CtfVerdict


def test_recommended_steps():
    pack = get_pack("ctf")
    assert pack.steps_for_mode(Mode.CHEAP) == ["ctf_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == ["ctf_teacher", "coding_teacher", "judge"]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "ctf_teacher",
        "coding_teacher",
        "general_teacher",
        "judge",
    ]


def test_teacher_prompts_contain_case_and_rag(ctf_case):
    pack = get_pack("ctf")

    class Chunk:
        text = "gets() is unsafe: no bounds check"
        source = "man"

    teacher = pack.teacher_prompt("ctf_teacher", ctf_case, [Chunk()])
    assert "gets()" in teacher
    assert "no bounds check" in teacher
    judge = pack.judge_prompt(ctf_case, [], {"ctf_teacher": {"summary": "s"}})
    assert "canonical verdict" in judge


def test_teacher_gate_passes_valid():
    pack = get_pack("ctf")
    data = {
        "summary": "Classic ret2win via gets() overflow",
        "technique": "pwn/buffer-overflow",
        "difficulty": "easy",
        "exploit_path": ["overflow the return address", "jump to win()"],
        "confidence": 0.9,
        "reasoning": "gets() with a 64-byte buffer and no canary.",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert result.passed


def test_teacher_gate_rejects_missing_technique():
    pack = get_pack("ctf")
    data = {
        "summary": "s",
        "technique": "",
        "difficulty": "easy",
        "exploit_path": ["x"],
        "confidence": 0.9,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("technique" in reason for reason in result.reasons)


def test_teacher_gate_rejects_missing_exploit_path():
    pack = get_pack("ctf")
    data = {
        "summary": "s",
        "technique": "pwn/buffer-overflow",
        "difficulty": "easy",
        "exploit_path": [],
        "confidence": 0.9,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("exploit_path" in reason for reason in result.reasons)


def test_judge_gate_passes_solved():
    pack = get_pack("ctf")
    data = {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": ["overflow", "ret2win"],
        "flag": "flag{r3t2w1n}",
        "writeup": "Overflow the buffer and jump to win().",
        "confidence": 0.9,
        "reasoning": "The binary prints the flag from win().",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_judge_gate_rejects_solved_without_flag():
    pack = get_pack("ctf")
    data = {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": ["overflow"],
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("flag" in reason for reason in result.reasons)


def test_judge_gate_rejects_solved_without_exploit_path():
    pack = get_pack("ctf")
    data = {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": [],
        "flag": "flag{x}",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("exploit_path" in reason for reason in result.reasons)


def test_student_input(ctf_case):
    pack = get_pack("ctf")
    text = pack.student_input(ctf_case)
    assert "Solve this CTF challenge." in text
    assert "gets()" in text
