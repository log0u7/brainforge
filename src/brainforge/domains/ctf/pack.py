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
            expected_flag = str(case.metadata.get("expected_flag") or "").strip() if case else ""
            # on verified-tier cases the flag lives outside the challenge
            # material: the known-flag rules replace the generic demand
            reasons.extend(self._judge_checks(data, require_flag=not expected_flag))
            reasons.extend(self._verification_checks(data, case))
        if not data.get("summary") and not data.get("reasoning"):
            reasons.append("missing summary or reasoning")
        return GateResult(passed=not reasons, reasons=reasons)

    def _verification_checks(self, data: dict, case: "Case | None") -> list[str]:
        """Verify the judge output against curated ground truth (verified tier).

        ``metadata.expected_technique`` (list of acceptable keywords) gates
        the named technique: the analysis must identify it, whatever the
        prose. ``metadata.expected_flag`` anti-hallucinates: a judge that
        claims a flag must produce the known one, but an absent flag is
        not an error (the flag lives outside the challenge material and
        stays the RL reward's business). Cases without this metadata
        (tier B) skip the checks entirely.
        """
        reasons: list[str] = []
        if case is None:
            return reasons
        expected_technique = case.metadata.get("expected_technique")
        if isinstance(expected_technique, str):
            expected_technique = [expected_technique]
        if isinstance(expected_technique, list) and expected_technique:
            technique = str(data.get("technique") or "").lower()
            keywords = [k.lower() for k in expected_technique if isinstance(k, str)]
            if keywords and not any(keyword in technique for keyword in keywords):
                reasons.append(
                    f"technique does not match expected_technique ({'/'.join(keywords)})"
                )
        expected_flag_raw = case.metadata.get("expected_flag")
        expected_flag = expected_flag_raw.strip() if isinstance(expected_flag_raw, str) else ""
        flag = str(data.get("flag") or "").strip()
        if expected_flag and flag and flag != expected_flag:
            reasons.append(
                "flag does not match expected_flag (verified tier; ground truth is known)"
            )
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

    def _judge_checks(self, data: dict, require_flag: bool = True) -> list[str]:
        reasons = []
        if data.get("verdict") == "solved":
            if require_flag and not data.get("flag"):
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
