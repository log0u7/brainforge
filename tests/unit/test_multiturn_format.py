"""Multi-turn dataset format extension tests.

The `tool` role and N-message records (agentic trajectories) ride on a
`DomainPack.build_messages` hook; the default path (user + assistant)
must stay byte-identical for every existing pack.
"""

import json

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.dataset.validation import validate_record
from brainforge.dataset.writer import DatasetRecord, build_record
from brainforge.domains.generic import GenericPack
from brainforge.providers.base import ChatMessage

_METADATA = {
    "source": {},
    "teachers": [],
    "quality": {"passed": True},
    "recitation_risk": False,
}


def _case():
    return Case(
        id="case-fmt",
        source=CaseSource(type="git", repository="repo", date="2026-08-15"),
        input=CaseInput(description="do the thing"),
    )


def _record(messages: list[dict]) -> dict:
    return {
        "id": "case-fmt",
        "domain": "agentic",
        "messages": messages,
        "metadata": _METADATA,
    }


def test_default_build_messages_two_turns():
    pack = GenericPack()
    messages = pack.build_messages("do the thing", {"verdict": "confirmed"})
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "do the thing"
    assert json.loads(messages[1].content) == {"verdict": "confirmed"}


def test_build_messages_uses_raw_assistant():
    pack = GenericPack()
    messages = pack.build_messages("task", {"verdict": "confirmed"}, raw_assistant=" raw output ")
    assert messages[1].content == " raw output "


def test_validation_accepts_tool_role():
    record = _record(
        [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"action": "run"}'},
            {"role": "tool", "content": "observation output"},
            {"role": "assistant", "content": '{"answer": "done"}'},
        ]
    )
    assert validate_record(record) == []


def test_validation_rejects_orphan_tool_message():
    record = _record(
        [
            {"role": "user", "content": "task"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": '{"answer": "done"}'},
        ]
    )
    errors = validate_record(record)
    assert any("tool message must follow" in error for error in errors)


def test_validation_rejects_tool_after_tool():
    record = _record(
        [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"action": "a"}'},
            {"role": "tool", "content": "obs"},
            {"role": "tool", "content": "obs2"},
            {"role": "assistant", "content": '{"answer": "done"}'},
        ]
    )
    errors = validate_record(record)
    assert any("tool message must follow" in error for error in errors)


def test_validation_rejects_unknown_role():
    record = _record(
        [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "{}"},
            {"role": "banana", "content": "x"},
        ]
    )
    errors = validate_record(record)
    assert any("banana" in error for error in errors)


class _TrajectoryPack(GenericPack):
    """Minimal multi-turn pack for engine wiring tests."""

    name = "trajectory"

    def build_messages(self, student_input, canonical, raw_assistant=None):
        steps = canonical.get("steps", [])
        messages = [ChatMessage(role="user", content=student_input)]
        for step in steps:
            messages.append(
                ChatMessage(role="assistant", content=json.dumps(step.get("action", {})))
            )
            messages.append(ChatMessage(role="tool", content=step["observation"]))
        messages.append(
            ChatMessage(
                role="assistant",
                content=raw_assistant or json.dumps(canonical, indent=2),
            )
        )
        return messages


def test_trajectory_pack_build_messages_shape():
    pack = _TrajectoryPack()
    messages = pack.build_messages(
        "task",
        {
            "verdict": "valid",
            "steps": [
                {"action": {"tool": "ls"}, "observation": "a.py b.py"},
                {"action": {"tool": "cat", "args": "a.py"}, "observation": "print(1)"},
            ],
            "final_answer": "the file prints 1",
        },
    )
    assert [m.role for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    record = DatasetRecord(
        id="case-fmt",
        domain="trajectory",
        messages=messages,
        metadata=_METADATA,
    )
    assert validate_record(record.model_dump()) == []


def test_build_record_default_still_two_turns():
    record = build_record(
        case=_case(),
        domain="security",
        student_input="Analyze",
        canonical={"verdict": "confirmed"},
        metadata=_METADATA,
    )
    assert [m.role for m in record.messages] == ["user", "assistant"]
