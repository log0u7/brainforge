import json
from abc import ABC, abstractmethod
from pathlib import Path
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

    def grpo_reward(self) -> str | None:
        """Verifiable GRPO reward name for this domain, or None when the
        domain has no reinforcement-learning signal (see
        brainforge.training.rewards registry)."""
        return None

    def grpo_dataset_columns(self) -> tuple[str, ...]:
        """Dataset columns the pack's GRPO reward consumes: they ride the
        training rows so TRL passes them to the reward function as kwargs
        (ADR-0022). Default: none - the reward scores completions only."""
        return ()

    def grpo_sidecar(self, dataset_dir: Path) -> dict[str, dict[str, str]]:
        """Per-id reward kwargs loaded from the pack's optional sidecar
        file next to the dataset (ADR-0022). Default: no sidecar - the
        reward needs no ground truth beyond the dataset columns."""
        return {}

    def merge_sidecar(self, dataset_dir: Path, entries: list[dict]) -> None:
        """Merge the consolidation ingest's sidecar rows into the pack's
        cumulative sidecar file (id-keyed, a re-ingested id replaces its
        stale row). Default: no sidecar - a no-op."""
        return None

    def ingest_trajectories(
        self, records: list[dict], cases: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Consolidation ingest hook: map the pilot's execution-verified
        trajectories to dataset rows plus sidecar rows (each sidecar row
        is an id-keyed dict carrying the pack's ground truth). Default:
        the records ARE the rows, with no sidecar."""
        return records, []

    def gate_metrics(self) -> tuple[str, ...]:
        """Metrics that gate consolidation acceptance (a regression on any
        of these keeps the previous checkpoint). Default: accuracy only."""
        return ("accuracy",)

    def eval_score(self, prediction: dict, expected: dict) -> dict | None:
        """Task-eval scoring hook: compute the verdict of one attempt on one
        case (ADR-0022). ``prediction`` and ``expected`` are the domain's
        eval shapes (see eval_shape). Returns the per-case scoring view
        score_task aggregates:
        ``{"predicted": <class>, "actual": <class>,
        "outcome": "correct" | "false_negative" | "false_positive"}``
        where "abstain" is a reserved class name. Default: None — the
        domain does not support task-level evaluation, and task_eval
        rejects it with an explicit error ("domain pack X does not
        implement eval_score")."""
        return None

    def eval_shape(self, data: dict) -> dict:
        """Map a parsed judge/model verdict JSON to the domain's eval shape.
        Default: the security shape (vulnerability_found / abstain / cwe),
        the historic default of task evaluation."""
        return {
            "vulnerability_found": data.get("verdict") == "confirmed",
            "abstain": data.get("verdict") == "insufficient_information",
            "cwe": data.get("cwe"),
        }

    def eval_unparseable(self) -> dict:
        """The eval shape for an answer that yielded no JSON (a miss whose
        direction is unknown). Default: the security shape."""
        return {"vulnerability_found": None, "abstain": False, "cwe": None}

    def eval_secondary_metrics(self, pairs: list[tuple[dict, dict]]) -> dict:
        """Domain-specific secondary metric over the eval pairs, merged into
        the score_task report. Default: the security CWE accuracy (compared
        only where the judge confirmed the vulnerability and classified a
        CWE)."""
        cwe_correct = 0
        cwe_compared = 0
        for prediction, expected in pairs:
            if expected.get("vulnerability_found") and expected.get("cwe"):
                cwe_compared += 1
                if prediction.get("cwe") == expected.get("cwe"):
                    cwe_correct += 1
        return {"cwe_accuracy": (cwe_correct / cwe_compared) if cwe_compared else None}

    def arbitrate(self, case: "Case", canonical: dict, teacher_results: dict) -> bool:
        """Resolve a teacher/judge disagreement with domain ground truth.

        Default: no verification material - the disagreement stands and the
        gate decides. Packs with curated ground truth override this to
        accept the case when verification proves one of the two was right.
        """
        return False

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
