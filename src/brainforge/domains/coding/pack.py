from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from brainforge.domains.base import (
    DomainPack,
    GateResult,
    GeneralCritique,
    format_case,
    format_previous_results,
    format_rag_context,
)
from brainforge.domains.coding.schemas import CodingJudgeVerdict, CodingTeacherAnalysis
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case

CODING_TEACHER_SYSTEM = """\
You are a software engineering teacher specialized in code comprehension and remediation.
Your mission:
- understand what the code does and explain its behavior;
- trace data flows and call chains relevant to the analysis;
- analyze patches and their consequences;
- understand frameworks and idioms in use;
- propose a concrete correction when a defect is present;
- explain the program behavior step by step.
Justify every conclusion with concrete elements from the provided code or data.
Output strictly valid JSON matching the requested schema."""

SECURITY_TEACHER_SYSTEM = """\
You are a security analysis teacher specialized in vulnerability research.
Your mission:
- detect vulnerabilities in the provided code or data;
- identify the matching CWE identifier when applicable;
- analyze exploitability, impact and preconditions;
- actively look for false positives;
- justify every conclusion with concrete elements from the provided code or data.
Output strictly valid JSON matching the requested schema."""

GENERAL_CRITIC_SYSTEM = """\
You are an independent general analyst acting as a devil's advocate.
Your mission:
- produce an independent analysis before reading the other teachers' conclusions;
- search for contradictions between the provided analyses;
- identify missing information that would change the conclusions;
- challenge the hypotheses of the other teachers.
You must NOT simply paraphrase or rubber-stamp the other analyses.
Output strictly valid JSON matching the requested schema."""

JUDGE_SYSTEM = """\
You are the judge of a multi-teacher analysis pipeline. You receive a case, retrieval
context, and the analyses of several teachers. Your mission:
- re-derive your conclusions from the case evidence itself before reading the teachers;
- list where you agree and disagree with each teacher;
- NEVER accept a teacher claim that you cannot trace back to the provided code or data;
- beware of correlated errors: two teachers agreeing is not proof if the evidence is absent;
- produce the canonical verdict for the case.
Output strictly valid JSON matching the requested schema."""

CODING_TEACHER_PROMPT = """\
Analyze the following code: behavior, data flow, call structure and defects.

{case}

Retrieved context:
{rag}

Produce your code analysis as valid JSON. Justify with evidence from the code."""

SECURITY_TEACHER_PROMPT = """\
Analyze the following case for security vulnerabilities.

{case}

Retrieved context:
{rag}

Produce your security analysis as valid JSON. Justify with evidence from the code."""

GENERAL_CRITIC_PROMPT = """\
Independent analysis of the following case first, then review the other teachers' analyses.

{case}

Retrieved context:
{rag}

Other teachers' analyses:
{previous}

Report agreements, contradictions, missing information and alternative hypotheses.
Do not paraphrase the other analyses."""

JUDGE_PROMPT = """\
Produce the canonical verdict for the following case.

{case}

Retrieved context:
{rag}

Teachers' analyses:
{previous}

Re-derive the conclusion from the evidence before weighing the teachers' opinions.
If the teachers agree but the evidence does not support them, reject the verdict.
If the case cannot be decided with the available information, use verdict
"insufficient_information" and set disagreement accordingly."""


class CodingPack(DomainPack):
    name = "coding"
    description = "Code understanding, defect analysis, patch comprehension and remediation."
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {
        Mode.CHEAP: ["coding_teacher", "judge"],
        Mode.STANDARD: ["coding_teacher", "security_teacher", "judge"],
        Mode.MAXIMUM: ["coding_teacher", "security_teacher", "general_teacher", "judge"],
    }

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return CodingTeacherAnalysis
        if kind == RoleKind.CRITIC:
            return GeneralCritique
        return CodingJudgeVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return JUDGE_SYSTEM
        if kind == RoleKind.CRITIC:
            return GENERAL_CRITIC_SYSTEM
        if "security" in role_name.lower():
            return SECURITY_TEACHER_SYSTEM
        return CODING_TEACHER_SYSTEM

    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        template = (
            SECURITY_TEACHER_PROMPT if "security" in role_name.lower() else CODING_TEACHER_PROMPT
        )
        return template.format(case=format_case(case), rag=format_rag_context(rag_context))

    def critic_prompt(
        self, role_name: str, case: "Case", rag_context: list, previous_results: dict
    ) -> str:
        return GENERAL_CRITIC_PROMPT.format(
            case=format_case(case),
            rag=format_rag_context(rag_context),
            previous=format_previous_results(previous_results),
        )

    def judge_prompt(self, case: "Case", rag_context: list, previous_results: dict) -> str:
        return JUDGE_PROMPT.format(
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
        if not data.get("behavior"):
            reasons.append("missing behavior explanation")
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    def _judge_checks(self, data: dict) -> list[str]:
        reasons = []
        if data.get("verdict") == "confirmed" and not data.get("issues"):
            reasons.append("confirmed verdict without issues")
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    def student_input(self, case: "Case") -> str:
        task = case.input.question or case.input.description or "Explain and analyze this code."
        code_block = f"\n\n```\n{case.input.code}\n```" if case.input.code else ""
        return f"{task}{code_block}"
