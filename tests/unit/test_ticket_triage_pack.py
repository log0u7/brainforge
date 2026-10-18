import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.ticket_triage.schemas import (
    TicketTriageTeacherAnalysis,
    TicketTriageVerdict,
)
from brainforge.types import Mode, RoleKind


@pytest.fixture
def ticket_case():
    return Case(
        id="ticket-vpn-001",
        source=CaseSource(type="ticketing", repository="example-helpdesk", date="2026-08-15"),
        input=CaseInput(
            description="VPN client fails after the Windows update, multiple users affected.",
            question="Classify and triage this ticket.",
        ),
    )


def test_ticket_triage_registered():
    assert "ticket_triage" in pack_names()


def test_get_pack_ticket_triage():
    pack = get_pack("ticket_triage")
    assert pack is not None
    assert pack.name == "ticket_triage"


def test_output_schemas():
    pack = get_pack("ticket_triage")
    assert pack.output_schema(RoleKind.TEACHER) is TicketTriageTeacherAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is TicketTriageVerdict


def test_recommended_steps():
    pack = get_pack("ticket_triage")
    assert pack.steps_for_mode(Mode.CHEAP) == ["ticket_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == [
        "ticket_teacher",
        "general_teacher",
        "judge",
    ]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "ticket_teacher",
        "general_teacher",
        "ticket_teacher_2",
        "judge",
    ]


def test_schemas_validate():
    analysis = TicketTriageTeacherAnalysis(
        summary="Windows VPN client broken after update",
        category="network",
        priority="high",
        team="network",
        confidence=0.9,
        reasoning="Ticket reports a regression after a known update window.",
    )
    assert analysis.category == "network"
    verdict = TicketTriageVerdict(
        verdict="escalated",
        category="network",
        priority="high",
        team="network",
        confidence=0.8,
        reasoning="Impact spans multiple users.",
        disagreement=False,
    )
    assert verdict.verdict == "escalated"
    assert verdict.response_draft is None


def test_prompts_contain_case_and_rag(ticket_case):
    pack = get_pack("ticket_triage")

    class Chunk:
        text = "Known issue KB-4421: VPN 4.2 breaks on Windows 11 24H2"
        source = "kb"

    teacher = pack.teacher_prompt("ticket_teacher", ticket_case, [Chunk()])
    assert "VPN client fails" in teacher
    assert "KB-4421" in teacher
    critic = pack.critic_prompt(
        "general_teacher", ticket_case, [], {"ticket_teacher": {"summary": "s"}}
    )
    assert "ticket_teacher" in critic
    judge = pack.judge_prompt(ticket_case, [], {"ticket_teacher": {"summary": "s"}})
    assert "canonical triage verdict" in judge


def test_system_prompts():
    pack = get_pack("ticket_triage")
    assert pack.system_prompt("judge", RoleKind.JUDGE).startswith("You are the judge")
    assert pack.system_prompt("general_teacher", RoleKind.CRITIC).startswith(
        "You are an independent general analyst"
    )
    assert pack.system_prompt("ticket_teacher", RoleKind.TEACHER).startswith(
        "You are an IT helpdesk triage teacher"
    )


def test_teacher_gate_passes_valid():
    pack = get_pack("ticket_triage")
    data = {
        "summary": "VPN down for a team",
        "category": "network",
        "priority": "high",
        "team": "network",
        "confidence": 0.9,
        "reasoning": "Multiple users report the same failure.",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert result.passed


def test_teacher_gate_rejects_missing_reasoning():
    pack = get_pack("ticket_triage")
    data = {
        "summary": "VPN down",
        "category": "network",
        "priority": "high",
        "team": "network",
        "confidence": 0.9,
        "reasoning": "",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("reasoning" in reason for reason in result.reasons)


def test_teacher_gate_rejects_low_confidence():
    pack = get_pack("ticket_triage")
    data = {
        "summary": "s",
        "category": "network",
        "priority": "low",
        "team": "network",
        "confidence": 0.2,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert not result.passed


def test_judge_gate_passes_triaged():
    pack = get_pack("ticket_triage")
    data = {
        "verdict": "triaged",
        "category": "access",
        "priority": "medium",
        "team": "it-support",
        "response_draft": "Reset your password via the self-service portal.",
        "confidence": 0.9,
        "reasoning": "Self-service reset applies.",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_judge_gate_rejects_triaged_without_response_draft():
    pack = get_pack("ticket_triage")
    data = {
        "verdict": "triaged",
        "category": "access",
        "priority": "medium",
        "team": "it-support",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("response_draft" in reason for reason in result.reasons)


def test_judge_gate_rejects_escalated_without_team():
    pack = get_pack("ticket_triage")
    data = {
        "verdict": "escalated",
        "category": "network",
        "priority": "high",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("team" in reason for reason in result.reasons)


def test_student_input(ticket_case):
    pack = get_pack("ticket_triage")
    text = pack.student_input(ticket_case)
    assert "Classify and triage this ticket." in text
    assert "VPN client fails" in text
