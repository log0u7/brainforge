from typing import Literal

from pydantic import BaseModel, Field


class AuditFinding(BaseModel):
    area: Literal["authn", "authz", "crypto", "secrets", "logging", "injection", "config", "deps"]
    severity: Literal["low", "medium", "high", "critical"]
    evidence: str
    recommendation: str | None = None
    priority: Literal["p0", "p1", "p2", "p3"] | None = None


class SecurityAuditAnalysis(BaseModel):
    summary: str
    findings: list[AuditFinding] = Field(default_factory=list)
    scope_covered: list[str] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class SecurityAuditVerdict(BaseModel):
    verdict: Literal["audit_complete", "audit_inconclusive", "insufficient_information"]
    findings: list[AuditFinding] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical"]
    remediation_priority: Literal["p0", "p1", "p2", "p3"] | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    disagreement: bool
