import re

from pydantic import BaseModel, Field, field_validator

_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CaseSource(BaseModel):
    type: str
    repository: str | None = None
    commit: str | None = None
    path: str | None = None
    url: str | None = None
    date: str | None = None


class CaseInput(BaseModel):
    code: str = ""
    description: str = ""
    question: str | None = None


class Case(BaseModel):
    id: str
    source: CaseSource
    input: CaseInput
    metadata: dict = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if _CASE_ID_PATTERN.match(value) is None:
            raise ValueError("case id must match [A-Za-z0-9_-]{1,64}")
        return value

    @property
    def language(self) -> str | None:
        return self.metadata.get("language")
