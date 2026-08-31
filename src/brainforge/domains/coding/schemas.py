from typing import Literal

from pydantic import BaseModel, Field

from brainforge.domains.base import Evidence


class CodeIssue(BaseModel):
    issue_type: str
    description: str
    severity: Literal["info", "low", "medium", "high", "critical"] | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class CodingTeacherAnalysis(BaseModel):
    summary: str
    behavior: str
    dataflow_notes: list[str] = Field(default_factory=list)
    issues: list[CodeIssue] = Field(default_factory=list)
    remediation: str | None = None
    patch_suggestion: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class CodingJudgeVerdict(BaseModel):
    verdict: Literal["confirmed", "rejected", "insufficient_information"]
    issues: list[CodeIssue] = Field(default_factory=list)
    reasoning: str
    remediation: str | None = None
    confidence: float = Field(ge=0, le=1)
    disagreement: bool
