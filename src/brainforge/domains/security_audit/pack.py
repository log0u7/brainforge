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
from brainforge.domains.security_audit.schemas import (
    SecurityAuditAnalysis,
    SecurityAuditVerdict,
)
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case

AUDIT_TEACHER_SYSTEM = """\
You are a security audit teacher specialized in reviewing systems for security flaws.
Your mission:
- enumerate concrete findings (authn, authz, crypto, secrets, logging, injection, config, deps);
- grade each finding's severity honestly, no inflation;
- justify every finding with specific evidence from the system under review;
- report the scope you covered and the scope you could not reach.
You must never invent a finding without concrete evidence from the system.
Output strictly valid JSON matching the requested schema."""

AUDIT_TEACHER_PROMPT = """\
Audit the following system for security issues.

{case}

Retrieved context:
{rag}

Produce your findings and risk assessment as valid JSON. Justify each finding
with evidence from the system."""

AUDIT_JUDGE_PROMPT = """\
Produce the canonical security audit verdict for the following system.

{case}

Retrieved context:
{rag}

Teachers' analyses:
{previous}

Re-derive the findings from the system evidence before weighing the teachers'
opinions. A critical finding demands a p0 or p1 remediation priority. Use
verdict "audit_inconclusive" or "insufficient_information" when the evidence
does not support a complete audit."""

PRIORITY_BY_SEVERITY = {
    "critical": {"p0", "p1"},
    "high": {"p1"},
}


class SecurityAuditPack(DomainPack):
    name = "security_audit"
    description = "Security audit: enumerate findings, grade severity, set remediation priority."
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {
        Mode.CHEAP: ["audit_teacher", "judge"],
        Mode.STANDARD: ["audit_teacher", "security_teacher", "judge"],
        Mode.MAXIMUM: ["audit_teacher", "security_teacher", "coding_teacher", "judge"],
    }

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return SecurityAuditAnalysis
        if kind == RoleKind.CRITIC:
            return GeneralCritique
        return SecurityAuditVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return JUDGE_SYSTEM
        if kind == RoleKind.CRITIC:
            return GENERAL_CRITIC_SYSTEM
        return AUDIT_TEACHER_SYSTEM

    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        return AUDIT_TEACHER_PROMPT.format(
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
        return AUDIT_JUDGE_PROMPT.format(
            case=format_case(case),
            rag=format_rag_context(rag_context),
            previous=format_previous_results(previous_results),
        )

    def quality_gate(self, kind: RoleKind, data: dict, min_confidence: float = 0.5) -> GateResult:
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
        return reasons

    def _judge_checks(self, data: dict) -> list[str]:
        reasons = []
        verdict = data.get("verdict")
        if verdict == "audit_complete" and not data.get("findings"):
            reasons.append("audit_complete verdict without findings")
        if verdict == "audit_complete":
            reasons.extend(self._priority_mismatches(data.get("findings", [])))
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    @staticmethod
    def _priority_mismatches(findings: list[dict]) -> list[str]:
        reasons = []
        for finding in findings:
            severity = str(finding.get("severity", ""))
            priority = finding.get("priority")
            allowed = PRIORITY_BY_SEVERITY.get(severity)
            if allowed and priority not in allowed:
                reasons.append(
                    f"severity {severity} requires priority in {sorted(allowed)}, got {priority}"
                )
        return reasons

    def student_input(self, case: "Case") -> str:
        task = case.input.question or "Audit the following system for security issues."
        description = f"\n\nSystem:\n{case.input.description}" if case.input.description else ""
        code_block = f"\n\n```\n{case.input.code}\n```" if case.input.code else ""
        return f"{task}{description}{code_block}"
