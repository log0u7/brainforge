import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

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
    # OpenAI SDK extra_body passthrough (e.g. llama.cpp chat_template_kwargs
    # for Qwen3 thinking control). Provider-specific fields ride here so the
    # request model stays provider-agnostic.
    extra_body: dict[str, Any] | None = None


class PositionLogprobs(BaseModel):
    """Top-k alternatives for one generated token position (KD teacher signal)."""

    token: str
    logprob: float
    topk: list[tuple[str, float]] = Field(default_factory=list)


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
    logprobs: list[PositionLogprobs] | None = None


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

    def complete_logprobs(
        self, request: ChatRequest, model: str, top_logprobs: int = 8
    ) -> ChatResponse:
        """Complete with per-token top-k logprobs (teacher KD signal).

        Providers that cannot serve logprobs keep the default and raise.
        """
        raise ProviderError(f"provider '{self.name}' does not support logprobs")

    def _structured_with(
        self,
        request: ChatRequest,
        model: str,
        schema: type[BaseModel],
        complete: Callable[[ChatRequest, str], ChatResponse],
    ) -> ChatResponse:
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
            response = complete(request.model_copy(update={"messages": messages}), model)
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

    def structured(self, request: ChatRequest, model: str, schema: type[BaseModel]) -> ChatResponse:
        return self._structured_with(request, model, schema, self.complete)

    def structured_logprobs(
        self, request: ChatRequest, model: str, schema: type[BaseModel], top_logprobs: int = 8
    ) -> ChatResponse:
        """Like structured(), but the response also carries per-token top-k logprobs."""
        return self._structured_with(
            request,
            model,
            schema,
            lambda req, mdl: self.complete_logprobs(req, mdl, top_logprobs),
        )
