from typing import Literal

from pydantic import BaseModel, Field


class ReviewFinding(BaseModel):
    severity: Literal["info", "low", "medium", "high", "critical"]
    category: str
    location: str | None = None
    issue: str
    suggested_fix: str | None = None


class CodeReviewTeacherAnalysis(BaseModel):
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    overall_risk: Literal["none", "low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class CodeReviewVerdict(BaseModel):
    verdict: Literal["approved", "changes_requested", "insufficient_information"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    overall_risk: Literal["none", "low", "medium", "high", "critical"] | None = None
    suggested_patch: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    disagreement: bool
