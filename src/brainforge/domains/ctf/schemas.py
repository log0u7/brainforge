from typing import Literal

from pydantic import BaseModel, Field


class CtfTeacherAnalysis(BaseModel):
    summary: str
    technique: str
    difficulty: Literal["easy", "medium", "hard", "insane"]
    exploit_path: list[str] = Field(default_factory=list)
    flag_candidate: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class CtfVerdict(BaseModel):
    verdict: Literal["solved", "partial", "insufficient_information"]
    technique: str | None = None
    exploit_path: list[str] = Field(default_factory=list)
    flag: str | None = None
    writeup: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    disagreement: bool
