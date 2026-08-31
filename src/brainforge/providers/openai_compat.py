import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from brainforge.config.models import ProviderConfig
from brainforge.errors import ProviderError, ProviderNotSupportedError, ProviderUnavailableError
from brainforge.providers.base import ChatRequest, ChatResponse, Provider, Usage
from brainforge.types import ApiStyle

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatProvider(Provider):
    default_base_url: str | None = "https://api.openai.com/v1"
    supported_api_styles = (ApiStyle.CHAT_COMPLETIONS,)

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        http_client: httpx.Client | None = None,
    ):
        if config.api_style not in self.supported_api_styles:
            raise ProviderNotSupportedError(
                f"provider '{name}' does not support api_style '{config.api_style}' "
                f"(supported: {', '.join(s.value for s in self.supported_api_styles)})"
            )
        super().__init__(name, config)
        base_url = config.base_url or self.default_base_url
        if base_url is None:
            raise ProviderError(f"provider '{name}' requires a base_url")
        self._client = OpenAI(
            base_url=base_url,
            api_key=config.api_key or "not-needed",
            timeout=config.timeout,
            max_retries=0,
            http_client=http_client,
        )

    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                start = time.perf_counter()
                response = self._client.chat.completions.create(**kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                usage = response.usage
                return ChatResponse(
                    content=response.choices[0].message.content,
                    provider=self.name,
                    model=model,
                    usage=Usage(
                        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    ),
                    latency_ms=latency_ms,
                    retries=attempt,
                )
            except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if isinstance(exc, APIStatusError) and status not in _RETRYABLE_STATUS:
                    detail = getattr(exc, "message", None) or str(exc)
                    raise ProviderError(
                        f"provider '{self.name}' rejected request (HTTP {status}): {detail}"
                    ) from exc
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(min(2**attempt * 0.5, 8.0))
        raise ProviderUnavailableError(
            f"provider '{self.name}' failed after {attempts} attempts: {last_error}"
        )

    def list_models(self) -> list[str]:
        return [model.id for model in self._client.models.list()]
