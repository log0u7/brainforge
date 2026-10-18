from typing import Literal

from pydantic import BaseModel, Field


class TicketTriageTeacherAnalysis(BaseModel):
    summary: str
    category: str
    priority: Literal["low", "medium", "high", "critical"]
    team: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class TicketTriageVerdict(BaseModel):
    verdict: Literal["triaged", "escalated", "insufficient_information"]
    category: str | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    team: str | None = None
    response_draft: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    disagreement: bool
