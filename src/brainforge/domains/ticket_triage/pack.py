from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from brainforge.domains.base import (
    GENERAL_CRITIC_PROMPT,
    GENERAL_CRITIC_SYSTEM,
    JUDGE_SYSTEM,
    DomainPack,
    GateResult,
    GeneralCritique,
    format_case,
    format_previous_results,
    format_rag_context,
)
from brainforge.domains.ticket_triage.schemas import (
    TicketTriageTeacherAnalysis,
    TicketTriageVerdict,
)
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case

TICKET_TEACHER_SYSTEM = """\
You are an IT helpdesk triage teacher specialized in incident and request classification.
Your mission:
- classify the ticket into the correct category;
- assess business priority from the described impact;
- route the ticket to the owning team;
- draft a first-response message the requester can act on;
- escalate to a human when the case cannot be decided with the available information;
- justify every conclusion with concrete elements from the ticket.
You must never speculate without pointing at the evidence that supports it.
Output strictly valid JSON matching the requested schema."""

TICKET_TEACHER_PROMPT = """\
Classify and triage the following support ticket.

{case}

Retrieved context:
{rag}

Produce your triage analysis as valid JSON. Justify with evidence from the ticket."""

TICKET_JUDGE_PROMPT = """\
Produce the canonical triage verdict for the following support ticket.

{case}

Retrieved context:
{rag}

Teachers' analyses:
{previous}

Re-derive the triage decision from the ticket evidence before weighing the teachers' opinions.
If the teachers agree but the evidence does not support their classification, escalate the
ticket instead of triaging it. Use verdict "triaged" only when category, priority, team and
a response draft are fully supported by the ticket, and "insufficient_information" when the
ticket cannot be decided with the available information."""


class TicketTriagePack(DomainPack):
    name = "ticket_triage"
    description = (
        "IT helpdesk triage: classification, business priority, team routing, first response."
    )
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {
        Mode.CHEAP: ["ticket_teacher", "judge"],
        Mode.STANDARD: ["ticket_teacher", "general_teacher", "judge"],
        Mode.MAXIMUM: ["ticket_teacher", "general_teacher", "ticket_teacher_2", "judge"],
    }

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return TicketTriageTeacherAnalysis
        if kind == RoleKind.CRITIC:
            return GeneralCritique
        return TicketTriageVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return JUDGE_SYSTEM
        if kind == RoleKind.CRITIC:
            return GENERAL_CRITIC_SYSTEM
        return TICKET_TEACHER_SYSTEM

    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        return TICKET_TEACHER_PROMPT.format(
            case=format_case(case), rag=format_rag_context(rag_context)
        )

    def critic_prompt(
        self, role_name: str, case: "Case", rag_context: list, previous_results: dict
    ) -> str:
        return GENERAL_CRITIC_PROMPT.format(
            case=format_case(case),
            rag=format_rag_context(rag_context),
            previous=format_previous_results(previous_results),
        )

    def judge_prompt(self, case: "Case", rag_context: list, previous_results: dict) -> str:
        return TICKET_JUDGE_PROMPT.format(
            case=format_case(case),
            rag=format_rag_context(rag_context),
            previous=format_previous_results(previous_results),
        )

    def quality_gate(
        self, kind: RoleKind, data: dict, min_confidence: float = 0.5, case: "Case | None" = None
    ) -> GateResult:
        reasons: list[str] = []
        confidence = data.get("confidence", 0)
        if confidence < min_confidence:
            reasons.append(f"confidence {confidence} below threshold {min_confidence}")
        if kind == RoleKind.TEACHER:
            reasons.extend(self._teacher_checks(data))
        elif kind == RoleKind.JUDGE:
            reasons.extend(self._judge_checks(data))
        if not data.get("summary") and not data.get("reasoning"):
            reasons.append("missing summary or reasoning")
        return GateResult(passed=not reasons, reasons=reasons)

    def _teacher_checks(self, data: dict) -> list[str]:
        reasons = []
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        if not data.get("category"):
            reasons.append("missing category")
        return reasons

    def _judge_checks(self, data: dict) -> list[str]:
        reasons = []
        verdict = data.get("verdict")
        if verdict == "triaged":
            if not data.get("category"):
                reasons.append("triaged verdict without category")
            if not data.get("response_draft"):
                reasons.append("triaged verdict without response_draft")
        if verdict == "escalated" and not data.get("team"):
            reasons.append("escalated verdict without team")
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    def student_input(self, case: "Case") -> str:
        task = case.input.question or "Classify and triage the following support ticket."
        description = f"\n\nTicket:\n{case.input.description}" if case.input.description else ""
        code_block = f"\n\n```\n{case.input.code}\n```" if case.input.code else ""
        return f"{task}{description}{code_block}"
