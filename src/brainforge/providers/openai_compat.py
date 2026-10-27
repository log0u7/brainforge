import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from brainforge.config.models import ProviderConfig
from brainforge.errors import ProviderError, ProviderNotSupportedError, ProviderUnavailableError
from brainforge.providers.base import (
    ChatRequest,
    ChatResponse,
    PositionLogprobs,
    Provider,
    Usage,
)
from brainforge.types import ApiStyle

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _create_call(create: Any, kwargs: dict[str, Any]) -> Any:
    """Module-level target for the wall-clock guard thread (mypy-friendly)."""
    return create(**kwargs)


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
            http_client=http_client,  # type: ignore[arg-type]  # openai>=3 types httpx2 but accepts httpx at runtime
        )

    def _create_with_retries(
        self, request: ChatRequest, model: str, extra_kwargs: dict[str, Any]
    ) -> tuple[Any, float, int]:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            **extra_kwargs,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        deadline = self.config.timeout + 30.0
        for attempt in range(attempts):
            try:
                start = time.perf_counter()
                # httpx read timeouts bound the time BETWEEN chunks, so a
                # drip-feeding stream can hang a call past any read timeout
                # (observed twice on gpu3080: pipeline frozen mid-batch at
                # zero CPU). Run the call in a worker thread and enforce a
                # wall-clock deadline here; the orphaned request thread dies
                # with the process.
                # no 'with': exiting the context manager calls
                # shutdown(wait=True), which waits for the orphaned hung
                # request thread and reintroduces the hang we are guarding
                # against. shutdown(wait=False) leaves it running (it dies
                # with the process) and unblocks the retry loop.
                pool = ThreadPoolExecutor(max_workers=1)
                try:
                    future = pool.submit(_create_call, self._client.chat.completions.create, kwargs)
                    response = future.result(timeout=deadline)
                finally:
                    pool.shutdown(wait=False)
                latency_ms = (time.perf_counter() - start) * 1000
                return response, latency_ms, attempt
            except TimeoutError as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(min(2**attempt * 0.5, 8.0))
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

    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        response, latency_ms, retries = self._create_with_retries(request, model, {})
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
            retries=retries,
        )

    def complete_logprobs(
        self, request: ChatRequest, model: str, top_logprobs: int = 8
    ) -> ChatResponse:
        response, latency_ms, retries = self._create_with_retries(
            request, model, {"logprobs": True, "top_logprobs": top_logprobs}
        )
        choice = response.choices[0]
        positions: list[PositionLogprobs] = []
        raw = getattr(choice, "logprobs", None)
        for entry in getattr(raw, "content", None) or []:
            topk = [(item.token, item.logprob) for item in (entry.top_logprobs or [])][
                :top_logprobs
            ]
            positions.append(PositionLogprobs(token=entry.token, logprob=entry.logprob, topk=topk))
        usage = response.usage
        return ChatResponse(
            content=choice.message.content,
            provider=self.name,
            model=model,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            retries=retries,
            logprobs=positions,
        )
