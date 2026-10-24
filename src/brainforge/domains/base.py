import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from brainforge.errors import ConfigError
from brainforge.providers.base import ChatMessage
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case


class GateResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    file: str | None = None
    lines: str | None = None
    snippet: str | None = None
    description: str


class GeneralCritique(BaseModel):
    summary: str
    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


def format_case(case: "Case") -> str:
    language = case.language or "unknown"
    parts = [f"Source: {case.source.type}", f"Language: {language}"]
    if case.source.repository:
        parts.append(f"Repository: {case.source.repository}")
    if case.source.commit:
        parts.append(f"Commit: {case.source.commit}")
    if case.input.description:
        parts.append(f"\nDescription:\n{case.input.description}")
    if case.input.code:
        parts.append(f"\nCode:\n```\n{case.input.code}\n```")
    return "\n".join(parts)


def format_rag_context(rag_context: list) -> str:
    if not rag_context:
        return "No RAG context available."
    chunks = []
    for index, chunk in enumerate(rag_context, start=1):
        source = getattr(chunk, "source", "unknown")
        text = getattr(chunk, "text", str(chunk))
        chunks.append(f"[{index}] ({source})\n{text[:800]}")
    return "\n\n".join(chunks)


def format_previous_results(previous_results: dict) -> str:
    if not previous_results:
        return "No prior analyses available."
    import json

    return json.dumps(previous_results, indent=2, ensure_ascii=False, default=str)


JUDGE_SYSTEM = """\
You are the judge of a multi-teacher analysis pipeline. You receive a case, retrieval
context, and the analyses of several teachers. Your mission:
- re-derive your conclusions from the case evidence itself before reading the teachers;
- list where you agree and disagree with each teacher;
- NEVER accept a teacher claim that you cannot trace back to the provided code or data;
- beware of correlated errors: two teachers agreeing is not proof if the evidence is absent;
- produce the canonical verdict for the case.
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

GENERAL_CRITIC_PROMPT = """\
Independent analysis of the following case first, then review the other teachers' analyses.

{case}

Retrieved context:
{rag}

Other teachers' analyses:
{previous}

Report agreements, contradictions, missing information and alternative hypotheses.
Do not paraphrase the other analyses."""


class DomainPack(ABC):
    name: str = ""
    description: str = ""
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {}

    @abstractmethod
    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        pass

    @abstractmethod
    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        pass

    @abstractmethod
    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        pass

    @abstractmethod
    def critic_prompt(
        self, role_name: str, case: "Case", rag_context: list, previous_results: dict
    ) -> str:
        pass

    @abstractmethod
    def judge_prompt(self, case: "Case", rag_context: list, previous_results: dict) -> str:
        pass

    @abstractmethod
    def quality_gate(
        self,
        kind: RoleKind,
        data: dict,
        min_confidence: float = 0.5,
        case: "Case | None" = None,
    ) -> GateResult:
        """Gate the step output. ``case`` carries verification material
        (e.g. ``metadata.expected_flag``); packs that do not use it ignore
        the parameter."""
        pass

    @abstractmethod
    def student_input(self, case: "Case") -> str:
        pass

    def steps_for_mode(self, mode: Mode) -> list[str]:
        steps = self.recommended_steps.get(mode)
        if not steps:
            raise ConfigError(
                f"domain '{self.name}' has no recommended steps for mode '{mode}'; "
                "declare explicit steps in the pipeline configuration"
            )
        return steps

    def build_messages(
        self,
        student_input: str,
        canonical: dict,
        raw_assistant: str | None = None,
    ) -> list[ChatMessage]:
        """Messages of the dataset record for this pack.

        Default: one user turn and one assistant turn (the canonical verdict
        as JSON, or the teacher's exact output in KD mode). Packs whose gold
        output is not a single JSON blob (e.g. agentic trajectories) override
        this to emit multi-turn records with `tool` observations.
        """
        assistant_content = (
            raw_assistant
            if raw_assistant is not None
            else json.dumps(canonical, ensure_ascii=False, indent=2)
        )
        return [
            ChatMessage(role="user", content=student_input),
            ChatMessage(role="assistant", content=assistant_content),
        ]
