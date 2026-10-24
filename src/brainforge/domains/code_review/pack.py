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
from brainforge.domains.code_review.schemas import (
    CodeReviewTeacherAnalysis,
    CodeReviewVerdict,
)
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case

REVIEW_TEACHER_SYSTEM = """\
You are a senior security reviewer specialized in reviewing changes (diffs,
pull requests, patches).
Your mission:
- identify security issues introduced by the change (injection, authz,
  secrets, crypto, logging gaps);
- assess the overall risk the change adds;
- propose a concrete fix for each finding, with a suggested patch when
  the change is small;
- distinguish real issues from style opinions;
- justify every conclusion with concrete elements from the diff.
You must never inflate severity without evidence.
Output strictly valid JSON matching the requested schema."""

REVIEW_TEACHER_PROMPT = """\
Review the following change for security issues.

{case}

Retrieved context:
{rag}

Produce your review as valid JSON. Justify with evidence from the diff."""

REVIEW_JUDGE_PROMPT = """\
Produce the canonical review verdict for the following change.

{case}

Retrieved context:
{rag}

Teachers' analyses:
{previous}

Re-derive the review from the diff evidence before weighing the teachers'
opinions. Use verdict "changes_requested" when the change introduces a
real finding, "approved" when it does not, and "insufficient_information"
when the change cannot be assessed with the available context. An
"approved" verdict must never coexist with a high or critical finding."""


class CodeReviewPack(DomainPack):
    name = "code_review"
    description = (
        "Security review of changes: findings, severity, suggested patches, approve/block."
    )
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {
        Mode.CHEAP: ["review_teacher", "judge"],
        Mode.STANDARD: ["review_teacher", "coding_teacher", "judge"],
        Mode.MAXIMUM: ["review_teacher", "coding_teacher", "general_teacher", "judge"],
    }

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return CodeReviewTeacherAnalysis
        if kind == RoleKind.CRITIC:
            return GeneralCritique
        return CodeReviewVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return JUDGE_SYSTEM
        if kind == RoleKind.CRITIC:
            return GENERAL_CRITIC_SYSTEM
        return REVIEW_TEACHER_SYSTEM

    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        return REVIEW_TEACHER_PROMPT.format(
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
        return REVIEW_JUDGE_PROMPT.format(
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
        if not data.get("summary"):
            reasons.append("missing summary")
        return reasons

    def _judge_checks(self, data: dict) -> list[str]:
        reasons = []
        verdict = data.get("verdict")
        findings = data.get("findings") or []
        if verdict == "changes_requested" and not findings:
            reasons.append("changes_requested verdict without findings")
        if verdict == "approved":
            severe = [
                finding for finding in findings if finding.get("severity") in {"high", "critical"}
            ]
            if severe:
                reasons.append("approved verdict with high or critical findings")
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    def student_input(self, case: "Case") -> str:
        task = case.input.question or "Review the following change for security issues."
        description = f"\n\nChange:\n{case.input.description}" if case.input.description else ""
        code_block = f"\n\n```\n{case.input.code}\n```" if case.input.code else ""
        return f"{task}{description}{code_block}"
