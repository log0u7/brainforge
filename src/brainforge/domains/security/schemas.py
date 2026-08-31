from typing import Literal

from pydantic import BaseModel, Field

from brainforge.domains.base import Evidence


class SecurityTeacherAnalysis(BaseModel):
    summary: str
    vulnerability_found: bool
    vulnerability_type: str | None = None
    cwe: str | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    confidence: float = Field(ge=0, le=1)
    preconditions: list[str] = Field(default_factory=list)
    exploitability: str | None = None
    false_positive_likelihood: float | None = Field(default=None, ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning: str


class SecurityJudgeVerdict(BaseModel):
    verdict: Literal["confirmed", "rejected", "insufficient_information"]
    vulnerability_type: str | None = None
    cwe: str | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning: str
    remediation: str | None = None
    disagreement: bool
