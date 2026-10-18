from typing import Literal

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    thought: str
    action: str
    action_input: dict = Field(default_factory=dict)
    observation: str


class AgentTrajectory(BaseModel):
    task_summary: str
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str
    tools_used: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class AgenticVerdict(BaseModel):
    verdict: Literal["valid", "flawed", "insufficient_information"]
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str | None = None
    critique: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    disagreement: bool
