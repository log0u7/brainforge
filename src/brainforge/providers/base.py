import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from brainforge.config.models import ProviderConfig
from brainforge.errors import ProviderError

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int | None = None
    cache_extra: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    content: str | None = None
    data: dict | None = None
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    cached: bool = False
    retries: int = 0


def extract_json(text: str) -> dict:
    candidate = text.strip()
    fenced = _JSON_BLOCK_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ProviderError("response does not contain a JSON object") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("response JSON is not an object")
    return parsed


class Provider(ABC):
    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config

    @abstractmethod
    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        pass

    def structured(self, request: ChatRequest, model: str, schema: type[BaseModel]) -> ChatResponse:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        instruction = (
            "Respond with a single JSON object and nothing else (no markdown, no code fences). "
            f"The object must validate against this JSON Schema:\n{schema_json}"
        )
        messages = list(request.messages)
        if messages and messages[0].role == "system":
            messages[0] = ChatMessage(
                role="system", content=f"{messages[0].content}\n\n{instruction}"
            )
        else:
            messages.insert(0, ChatMessage(role="system", content=instruction))
        parse_retries = 2
        last_error: Exception | None = None
        for attempt in range(parse_retries + 1):
            response = self.complete(request.model_copy(update={"messages": messages}), model)
            try:
                data = schema.model_validate(extract_json(response.content or ""))
                response.data = data.model_dump()
                response.retries = attempt
                return response
            except (ProviderError, ValueError) as exc:
                last_error = exc
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=response.content or ""),
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response was not a valid JSON object matching the "
                            "schema. Answer again with a single valid JSON object only."
                        ),
                    ),
                ]
        raise ProviderError(
            f"structured output for schema '{schema.__name__}' failed after "
            f"{parse_retries + 1} attempts: {last_error}"
        )
