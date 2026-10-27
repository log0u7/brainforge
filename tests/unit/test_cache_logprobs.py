"""CacheProvider must delegate complete_logprobs (KD teacher signal)."""

import pytest

from brainforge.config.models import ProviderConfig
from brainforge.errors import ProviderError
from brainforge.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PositionLogprobs,
    Provider,
)


def _config() -> ProviderConfig:
    return ProviderConfig(type="mock")


def _request() -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content="hi")])


class _NoLogprobsProvider(Provider):
    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        raise NotImplementedError


class _LogprobsProvider(_NoLogprobsProvider):
    def complete_logprobs(
        self, request: ChatRequest, model: str, top_logprobs: int = 8
    ) -> ChatResponse:
        return ChatResponse(
            provider=self.name,
            model=model,
            logprobs=[PositionLogprobs(token="ok", logprob=-0.1, topk=[("ok", -0.1)])],
        )


def test_cache_delegates_logprobs(tmp_path):
    """A provider that supports logprobs must be reachable THROUGH the
    cache wrapper - base.complete_logprobs would raise otherwise."""
    from brainforge.providers.cache import CacheProvider

    inner = _LogprobsProvider("judge", _config())
    wrapped = CacheProvider(inner, tmp_path / "cache.sqlite")
    response = wrapped.complete_logprobs(_request(), "m")
    assert response.logprobs is not None
    assert response.logprobs[0].token == "ok"


def test_cache_still_raises_for_unsupported_provider(tmp_path):
    from brainforge.providers.cache import CacheProvider

    wrapped = CacheProvider(_NoLogprobsProvider("nope", _config()), tmp_path / "cache.sqlite")
    with pytest.raises(ProviderError, match="does not support logprobs"):
        wrapped.complete_logprobs(_request(), "m")
