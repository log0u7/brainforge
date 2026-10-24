import json
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from brainforge.domains.agentic.schemas import AgenticVerdict, AgentTrajectory
from brainforge.domains.base import (
    GENERAL_CRITIC_PROMPT,
    GENERAL_CRITIC_SYSTEM,
    JUDGE_SYSTEM,
    ChatMessage,
    DomainPack,
    GateResult,
    GeneralCritique,
    format_case,
    format_previous_results,
    format_rag_context,
)
from brainforge.types import Mode, RoleKind

if TYPE_CHECKING:
    from brainforge.case import Case

AGENT_TEACHER_SYSTEM = """\
You are an agentic task-solving teacher. You solve the task by issuing one
tool call at a time and reacting to each observation.
Your mission:
- state your current thought before every action;
- pick exactly one action per step, grounded in the observations so far;
- record the raw observation for each action;
- finish with a concrete final answer supported by the trajectory.
You must never fabricate an observation or skip the evidence trail.
Output strictly valid JSON matching the requested schema."""

AGENT_TEACHER_PROMPT = """\
Solve the following task using the provided tools.

{case}

Retrieved context:
{rag}

Produce your full trajectory as valid JSON: one step per tool call with its
observation, then the final answer."""

AGENT_JUDGE_PROMPT = """\
Produce the canonical verdict for the following agentic trajectory.

{case}

Retrieved context:
{rag}

Teachers' trajectories:
{previous}

Re-derive the solution from the task evidence before weighing the teachers'
opinions. A trajectory is "valid" only when each step follows from the prior
observations and the final answer is supported by the trajectory. Use
"flawed" when a step is unjustified or the answer does not follow, and
"insufficient_information" when the trajectory cannot be judged."""

# Ponytail: action string in the assistant turn only; upgrade to a full
# tool-call protocol (name+arguments) if the chat template needs it.


class AgenticPack(DomainPack):
    name = "agentic"
    description = "Agentic tasks: multi-turn tool-use trajectories with per-step observations."
    recommended_steps: ClassVar[dict[Mode, list[str]]] = {
        Mode.CHEAP: ["agent_teacher", "judge"],
        Mode.STANDARD: ["agent_teacher", "coding_teacher", "judge"],
        Mode.MAXIMUM: ["agent_teacher", "coding_teacher", "general_teacher", "judge"],
    }

    def output_schema(self, kind: RoleKind) -> type[BaseModel]:
        if kind == RoleKind.TEACHER:
            return AgentTrajectory
        if kind == RoleKind.CRITIC:
            return GeneralCritique
        return AgenticVerdict

    def system_prompt(self, role_name: str, kind: RoleKind) -> str:
        if kind == RoleKind.JUDGE:
            return JUDGE_SYSTEM
        if kind == RoleKind.CRITIC:
            return GENERAL_CRITIC_SYSTEM
        return AGENT_TEACHER_SYSTEM

    def teacher_prompt(self, role_name: str, case: "Case", rag_context: list) -> str:
        return AGENT_TEACHER_PROMPT.format(
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
        return AGENT_JUDGE_PROMPT.format(
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
        if not data.get("reasoning"):
            reasons.append("missing reasoning")
        return GateResult(passed=not reasons, reasons=reasons)

    @staticmethod
    def _teacher_checks(data: dict) -> list[str]:
        reasons = []
        if not data.get("steps"):
            reasons.append("missing steps")
        if not data.get("final_answer"):
            reasons.append("missing final_answer")
        return reasons

    @staticmethod
    def _judge_checks(data: dict) -> list[str]:
        reasons = []
        if data.get("verdict") == "valid" and not data.get("final_answer"):
            reasons.append("valid verdict without final_answer")
        return reasons

    def student_input(self, case: "Case") -> str:
        task = case.input.question or "Solve the following task with the provided tools."
        description = f"\n\nTask:\n{case.input.description}" if case.input.description else ""
        tools_block = f"\n\nAvailable tools (JSON):\n{case.input.code}" if case.input.code else ""
        return f"{task}{description}{tools_block}"

    def build_messages(
        self,
        student_input: str,
        canonical: dict,
        raw_assistant: str | None = None,
    ) -> list[ChatMessage]:
        """Multi-turn record: user, then per step an assistant tool call and a
        tool observation, then a final assistant turn with the verdict JSON.
        """
        messages: list[ChatMessage] = [ChatMessage(role="user", content=student_input)]
        for step in canonical.get("steps", []):
            call = {
                "thought": step.get("thought", ""),
                "action": step.get("action", ""),
                "action_input": step.get("action_input", {}),
            }
            messages.append(
                ChatMessage(role="assistant", content=json.dumps(call, ensure_ascii=False))
            )
            messages.append(ChatMessage(role="tool", content=step.get("observation", "")))
        assistant_content = (
            raw_assistant
            if raw_assistant is not None
            else json.dumps(canonical, ensure_ascii=False, indent=2)
        )
        messages.append(ChatMessage(role="assistant", content=assistant_content))
        return messages
