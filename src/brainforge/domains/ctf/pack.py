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
from brainforge.domains.ctf.schemas import CtfTeacherAnalysis, CtfVerdict
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case

CTF_TEACHER_SYSTEM = """\
You are a CTF teacher specialized in challenge analysis and exploitation.
Your mission:
- identify the challenge category (pwn, web, crypto, reverse, forensics);
- name the technique that solves it;
- describe the concrete exploit path step by step;
- derive the flag when the provided material allows it;
- actively look for dead ends and red herrings;
- justify every conclusion with concrete elements from the challenge.
You must never invent a flag without the steps that produce it.
Output strictly valid JSON matching the requested schema."""

CTF_TEACHER_PROMPT = """\
Analyze and solve the following CTF challenge.

{case}

Retrieved context:
{rag}

Produce your analysis and exploit path as valid JSON. Justify with evidence
from the challenge."""

CTF_JUDGE_PROMPT = """\
Produce the canonical verdict for the following CTF challenge.

{case}

Retrieved context:
{rag}

Teachers' analyses:
{previous}

Re-derive the solution from the challenge evidence before weighing the
teachers' opinions. If the teachers agree but the evidence does not support
a solution, use verdict "partial" or "insufficient_information" instead of
"solved". Use verdict "solved" only when the flag and the exploit path are
both fully supported by the challenge."""


class CtfPack(DomainPack):
    name = "ctf"
    description = "CTF challenge solving: category, technique, exploit path, flag derivation."
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {
        Mode.CHEAP: ["ctf_teacher", "judge"],
        Mode.STANDARD: ["ctf_teacher", "coding_teacher", "judge"],
        Mode.MAXIMUM: ["ctf_teacher", "coding_teacher", "general_teacher", "judge"],
    }

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return CtfTeacherAnalysis
        if kind == RoleKind.CRITIC:
            return GeneralCritique
        return CtfVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return JUDGE_SYSTEM
        if kind == RoleKind.CRITIC:
            return GENERAL_CRITIC_SYSTEM
        return CTF_TEACHER_SYSTEM

    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        return CTF_TEACHER_PROMPT.format(
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
        return CTF_JUDGE_PROMPT.format(
            case=format_case(case),
            rag=format_rag_context(rag_context),
            previous=format_previous_results(previous_results),
        )

    def quality_gate(
        self,
        kind: RoleKind,
        data: dict,
        min_confidence: float = 0.5,
        case: "Case | None" = None,
    ) -> GateResult:
        reasons: list[str] = []
        confidence = data.get("confidence", 0)
        if confidence < min_confidence:
            reasons.append(f"confidence {confidence} below threshold {min_confidence}")
        if kind == RoleKind.TEACHER:
            reasons.extend(self._teacher_checks(data))
        elif kind == RoleKind.JUDGE:
            reasons.extend(self._judge_checks(data))
            reasons.extend(self._flag_checks(data, case))
        if not data.get("summary") and not data.get("reasoning"):
            reasons.append("missing summary or reasoning")
        return GateResult(passed=not reasons, reasons=reasons)

    def _flag_checks(self, data: dict, case: "Case | None") -> list[str]:
        """Verify the judge verdict against a known flag (verified tier).

        Cases carrying ``metadata.expected_flag`` state the ground truth:
        any non-empty flag that differs from it rejects the record
        whatever the verdict, and a matching flag with a non-solved
        verdict is a contradiction that rejects too. Cases without it
        (tier B) skip the check entirely.
        """
        reasons: list[str] = []
        expected = str(case.metadata.get("expected_flag") or "").strip() if case else ""
        if not expected:
            return reasons
        flag = str(data.get("flag") or "").strip()
        if flag and flag != expected:
            reasons.append(
                "flag does not match expected_flag (verified tier; ground truth is known)"
            )
        if flag == expected and flag and data.get("verdict") != "solved":
            reasons.append("flag matches expected_flag but the verdict is not solved")
        return reasons

    def _teacher_checks(self, data: dict) -> list[str]:
        reasons = []
        if not data.get("technique"):
            reasons.append("missing technique")
        if not data.get("exploit_path"):
            reasons.append("missing exploit_path")
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    def _judge_checks(self, data: dict) -> list[str]:
        reasons = []
        if data.get("verdict") == "solved":
            if not data.get("flag"):
                reasons.append("solved verdict without flag")
            if not data.get("exploit_path"):
                reasons.append("solved verdict without exploit_path")
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return reasons

    def student_input(self, case: "Case") -> str:
        task = case.input.question or "Solve the following CTF challenge."
        description = f"\n\nChallenge:\n{case.input.description}" if case.input.description else ""
        code_block = f"\n\n```\n{case.input.code}\n```" if case.input.code else ""
        return f"{task}{description}{code_block}"
