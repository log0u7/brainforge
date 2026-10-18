import json

import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.agentic.schemas import AgenticVerdict, AgentStep, AgentTrajectory
from brainforge.domains.base import GeneralCritique
from brainforge.types import Mode, RoleKind

TOOLS = json.dumps(
    [
        {"name": "read_file", "description": "Read a file by path"},
        {"name": "search", "description": "Search the codebase"},
    ]
)


@pytest.fixture
def agentic_case():
    return Case(
        id="agent-debug-001",
        source=CaseSource(type="task", repository="example-app", date="2026-08-25"),
        input=CaseInput(
            code=TOOLS,
            description=(
                "Task: find why the test suite fails. A traceback points at "
                "utils.parse in parse.py."
            ),
            question="Investigate and fix the failing test.",
        ),
    )


def test_agentic_registered():
    assert "agentic" in pack_names()


def test_output_schemas():
    pack = get_pack("agentic")
    assert pack.output_schema(RoleKind.TEACHER) is AgentTrajectory
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is AgenticVerdict


def test_recommended_steps():
    pack = get_pack("agentic")
    assert pack.steps_for_mode(Mode.CHEAP) == ["agent_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == ["agent_teacher", "coding_teacher", "judge"]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "agent_teacher",
        "coding_teacher",
        "general_teacher",
        "judge",
    ]


def test_teacher_gate_passes_valid_trajectory():
    pack = get_pack("agentic")
    data = {
        "task_summary": "debug failing parse test",
        "steps": [
            {
                "thought": "read the traceback target",
                "action": "read_file",
                "action_input": {"path": "parse.py"},
                "observation": "def parse(s): return s.split",
            }
        ],
        "final_answer": "parse returned the method object instead of calling split()",
        "tools_used": ["read_file"],
        "confidence": 0.8,
        "reasoning": "observation shows the missing call parentheses",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert result.passed


def test_teacher_gate_rejects_empty_steps():
    pack = get_pack("agentic")
    data = {
        "task_summary": "s",
        "steps": [],
        "final_answer": "done",
        "confidence": 0.9,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("steps" in reason for reason in result.reasons)


def test_judge_gate_passes_valid():
    pack = get_pack("agentic")
    data = {
        "verdict": "valid",
        "steps": [
            {
                "thought": "t",
                "action": "search",
                "action_input": {"query": "parse"},
                "observation": "parse.py",
            }
        ],
        "final_answer": "fixed",
        "critique": "the trajectory is sound",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_judge_gate_rejects_valid_without_final_answer():
    pack = get_pack("agentic")
    data = {
        "verdict": "valid",
        "steps": [],
        "final_answer": "",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("final_answer" in reason for reason in result.reasons)


def test_student_input_includes_tools_and_task(agentic_case):
    pack = get_pack("agentic")
    text = pack.student_input(agentic_case)
    assert "read_file" in text
    assert "Investigate and fix" in text


def test_step_schema():
    step = AgentStep(thought="t", action="read_file", action_input={"path": "x"}, observation="ok")
    assert step.action == "read_file"


def test_build_messages_multi_turn(agentic_case):
    pack = get_pack("agentic")
    canonical = {
        "verdict": "valid",
        "steps": [
            {
                "thought": "read the target",
                "action": "read_file",
                "action_input": {"path": "parse.py"},
                "observation": "def parse(s): return s.split",
            }
        ],
        "final_answer": "fixed the missing call",
        "critique": "sound",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    messages = pack.build_messages(pack.student_input(agentic_case), canonical)
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # assistant turn is the tool call as JSON, not the whole verdict
    call = json.loads(messages[1].content)
    assert call["action"] == "read_file"
    assert messages[2].role == "tool"
    assert "def parse" in messages[2].content
    # final assistant turn is the verdict
    final = json.loads(messages[3].content)
    assert final["verdict"] == "valid"


def test_system_prompt_kinds():
    pack = get_pack("agentic")
    assert "agentic" in pack.system_prompt("agent_teacher", RoleKind.TEACHER)
    assert "judge" in pack.system_prompt("judge", RoleKind.JUDGE).lower()
    assert "devil" in pack.system_prompt("critic", RoleKind.CRITIC).lower()


def test_teacher_prompt_and_judge_prompt(agentic_case):
    pack = get_pack("agentic")

    class Chunk:
        text = "read_file is useful"
        source = "doc"

    teacher = pack.teacher_prompt("agent_teacher", agentic_case, [Chunk()])
    assert "read_file is useful" in teacher
    judge = pack.judge_prompt(agentic_case, [], {"agent_teacher": {"summary": "s"}})
    assert "canonical verdict" in judge.lower()


def test_critic_prompt(agentic_case):
    pack = get_pack("agentic")
    text = pack.critic_prompt("critic", agentic_case, [], {})
    assert "read_file" in text


def test_quality_gate_low_confidence():
    pack = get_pack("agentic")
    result = pack.quality_gate(
        RoleKind.TEACHER,
        {
            "confidence": 0.1,
            "steps": [{"thought": "t", "action": "a", "action_input": {}, "observation": "o"}],
            "final_answer": "done",
            "reasoning": "r",
        },
    )
    assert not result.passed
    assert any("confidence" in reason for reason in result.reasons)


def test_quality_gate_missing_reasoning():
    pack = get_pack("agentic")
    result = pack.quality_gate(
        RoleKind.TEACHER,
        {
            "confidence": 0.9,
            "steps": [{"thought": "t", "action": "a", "action_input": {}, "observation": "o"}],
            "final_answer": "done",
        },
    )
    assert not result.passed
    assert any("missing reasoning" in reason for reason in result.reasons)


def test_quality_gate_teacher_missing_final_answer():
    pack = get_pack("agentic")
    result = pack.quality_gate(
        RoleKind.TEACHER,
        {
            "confidence": 0.9,
            "steps": [{"thought": "t", "action": "a", "action_input": {}, "observation": "o"}],
            "reasoning": "r",
        },
    )
    assert not result.passed
    assert any("missing final_answer" in reason for reason in result.reasons)
