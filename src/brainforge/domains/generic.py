from typing import Literal

from pydantic import BaseModel, Field

from brainforge.case import Case
from brainforge.domains.base import DomainPack, GateResult
from brainforge.types import RoleKind


class GenericAnalysis(BaseModel):
    summary: str
    details: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class GenericVerdict(BaseModel):
    verdict: Literal["confirmed", "rejected", "insufficient_information"]
    summary: str
    reasoning: str
    confidence: float = Field(ge=0, le=1)
    disagreement: bool


class GenericPack(DomainPack):
    name = "generic"
    description = "Fallback pack for domains without a dedicated implementation."

    def __init__(self, name: str = "generic", description: str | None = None):
        super().__init__()
        self.name = name
        if description:
            self.description = description

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return GenericAnalysis
        return GenericVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return (
                "You are the judge of an analysis pipeline. Re-derive conclusions from the "
                "case evidence, weigh the teachers' analyses, and produce the canonical "
                "verdict. Output strictly valid JSON matching the requested schema."
            )
        if kind == RoleKind.CRITIC:
            return (
                "You are an independent analyst. Challenge the other analyses, find "
                "contradictions and missing information. Do not paraphrase them. "
                "Output strictly valid JSON matching the requested schema."
            )
        return (
            "You are a domain expert teacher. Analyze the provided case and justify every "
            "conclusion with concrete evidence. Output strictly valid JSON matching the "
            "requested schema."
        )

    def teacher_prompt(self, role_name: str, case: Case, rag_context: list) -> str:
        return (
            f"Analyze the following case.\n\n{self._case_text(case)}\n\n"
            f"Retrieved context:\n{self._rag_text(rag_context)}\n\n"
            "Produce your analysis as valid JSON."
        )

    def critic_prompt(
        self, role_name: str, case: Case, rag_context: list, previous_results: dict
    ) -> str:
        return (
            f"Analyze the following case independently, then review the other analyses.\n\n"
            f"{self._case_text(case)}\n\nRetrieved context:\n{self._rag_text(rag_context)}\n\n"
            f"Other analyses:\n{self._previous_text(previous_results)}\n\n"
            "Report contradictions, missing information and alternative hypotheses as valid JSON."
        )

    def judge_prompt(self, case: Case, rag_context: list, previous_results: dict) -> str:
        return (
            f"Produce the canonical verdict for the following case.\n\n"
            f"{self._case_text(case)}\n\nRetrieved context:\n{self._rag_text(rag_context)}\n\n"
            f"Teachers' analyses:\n{self._previous_text(previous_results)}\n\n"
            "Re-derive the conclusion from the evidence before weighing the teachers. "
            "Answer as valid JSON."
        )

    def quality_gate(self, kind: RoleKind, data: dict, min_confidence: float = 0.5) -> GateResult:
        reasons = []
        if data.get("confidence", 0) < min_confidence:
            reasons.append(f"confidence below threshold {min_confidence}")
        if kind == RoleKind.JUDGE and not data.get("reasoning"):
            reasons.append("missing reasoning")
        if kind == RoleKind.TEACHER and not data.get("summary"):
            reasons.append("missing summary")
        return GateResult(passed=not reasons, reasons=reasons)

    def student_input(self, case: Case) -> str:
        task = case.input.question or case.input.description or "Analyze the following case."
        code_block = f"\n\n```\n{case.input.code}\n```" if case.input.code else ""
        return f"{task}{code_block}"

    def _case_text(self, case: Case) -> str:
        from brainforge.domains.base import format_case

        return format_case(case)

    def _rag_text(self, rag_context: list) -> str:
        from brainforge.domains.base import format_rag_context

        return format_rag_context(rag_context)

    def _previous_text(self, previous_results: dict) -> str:
        from brainforge.domains.base import format_previous_results

        return format_previous_results(previous_results)
