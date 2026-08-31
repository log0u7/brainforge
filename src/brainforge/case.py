from pydantic import BaseModel, Field


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

    @property
    def language(self) -> str | None:
        return self.metadata.get("language")
