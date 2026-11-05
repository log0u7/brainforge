import threading
import time
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

#: wall-clock grace added on top of ``config.timeout`` before the guard
#: declares the call hung: the SDK read timeout fires first in the normal
#: case, this backstop covers drip-feed streams and wedged transports
DEADLINE_GRACE = 30.0
#: idle keep-alive sockets are dropped after this many seconds - before the
#: server closes them itself. A server-closed socket handed back from the
#: pool freezes the reader (the gpu3080 CLOSE-WAIT freeze).
HTTPX_KEEPALIVE_EXPIRY = 30.0


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
        self._limits: httpx.Limits | None = None
        if http_client is None:
            # own the connection pool: a short keepalive_expiry drops idle
            # keep-alive sockets before the server closes them itself, so
            # the next request is never handed a dead (CLOSE-WAIT) socket
            limits = httpx.Limits(
                max_connections=64,
                max_keepalive_connections=16,
                keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY,
            )
            http_client = httpx.Client(
                timeout=httpx.Timeout(config.timeout, connect=10.0),
                limits=limits,
            )
            self._limits = limits
        else:
            self._limits = getattr(http_client, "_limits", None) or getattr(
                http_client, "limits", None
            )
        self._http_client = http_client
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
        if request.extra_body:
            kwargs["extra_body"] = request.extra_body
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        deadline = self.config.timeout + DEADLINE_GRACE
        for attempt in range(attempts):
            try:
                start = time.perf_counter()
                # httpx read timeouts bound the time BETWEEN chunks, so a
                # drip-feeding stream can hang a call past any read timeout
                # (observed twice on gpu3080: pipeline frozen mid-batch at
                # zero CPU). Run the call in a dedicated daemon thread and
                # enforce a wall-clock deadline here: a daemon dies with the
                # process, so an orphaned hung thread can never block
                # interpreter exit (the gpu3080 freeze was the ThreadPool-
                # Executor atexit hook joining a worker stuck on a server-
                # closed socket - 67 minutes at zero CPU).
                box: dict[str, Any] = {}
                done = threading.Event()

                def run_call(box: dict[str, Any] = box, done: threading.Event = done) -> None:
                    try:
                        box["response"] = _create_call(self._client.chat.completions.create, kwargs)
                    except Exception as exc:  # re-raised on the caller side
                        box["error"] = exc
                    finally:
                        done.set()

                worker = threading.Thread(target=run_call, daemon=True)
                worker.start()
                if not done.wait(timeout=deadline):
                    raise TimeoutError(
                        f"provider '{self.name}' call exceeded the "
                        f"{deadline:.0f}s wall-clock deadline"
                    )
                if "error" in box:
                    raise box["error"]
                latency_ms = (time.perf_counter() - start) * 1000
                return box["response"], latency_ms, attempt
            except TimeoutError:
                # a wall-clock deadline hit means the transport is wedged:
                # retrying would just burn another full deadline on the same
                # hang, so the guard raises straight through (SDK-level
                # timeouts below still retry)
                raise
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
