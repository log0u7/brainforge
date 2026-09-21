import json

import httpx
import pytest
from pydantic import BaseModel

from brainforge.config.models import ProviderConfig
from brainforge.errors import ProviderError, ProviderNotSupportedError, ProviderUnavailableError
from brainforge.providers.base import ChatMessage, ChatRequest, ChatResponse, Provider, extract_json
from brainforge.providers.cache import CacheProvider
from brainforge.providers.mock import MockProvider
from brainforge.providers.observability import UsageLogger, estimate_cost
from brainforge.providers.openai_compat import OpenAICompatProvider
from brainforge.providers.registry import build_provider
from brainforge.types import ApiStyle


class SimpleSchema(BaseModel):
    verdict: str
    confidence: float
    items: list[str]


class NestedSchema(BaseModel):
    name: str
    detail: SimpleSchema
    note: str | None = None


def make_request(**kwargs) -> ChatRequest:
    defaults = {
        "messages": [ChatMessage(role="user", content="analyze this code")],
        "temperature": 0.2,
    }
    defaults.update(kwargs)
    return ChatRequest(**defaults)


def make_provider(**config_kwargs) -> OpenAICompatProvider:
    config = ProviderConfig(
        type="openai", base_url="http://testserver/v1", api_key="k", **config_kwargs
    )
    return OpenAICompatProvider("test", config)


def make_provider_with_responses(
    responses: list[httpx.Response], calls: list[httpx.Request] | None = None, **config_kwargs
) -> OpenAICompatProvider:
    queue = list(responses)
    recorded = calls if calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if len(queue) == 1:
            return queue[0]
        return queue.pop(0)

    config = ProviderConfig(
        type="openai", base_url="http://testserver/v1", api_key="k", **config_kwargs
    )
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider("test", config, http_client=http_client)


COMPLETION_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class CountingProvider(Provider):
    def __init__(self):
        super().__init__("counting", ProviderConfig(type="mock"))
        self.calls = 0

    def complete(self, request, model):
        self.calls += 1
        response = ChatResponse(
            content=json.dumps({"verdict": "ok", "confidence": 0.5, "items": ["x"]}),
            provider=self.name,
            model=model,
        )
        return response


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_embedded():
    text = 'Here is my analysis:\n{"a": 1} hope that helps'
    assert extract_json(text) == {"a": 1}


def test_extract_json_invalid():
    with pytest.raises(ProviderError):
        extract_json("no json here")


def test_extract_json_not_object():
    with pytest.raises(ProviderError):
        extract_json("[1, 2]")


def test_mock_complete():
    provider = MockProvider("mock", ProviderConfig(type="mock"))
    response = provider.complete(make_request(), "mock-model")
    assert response.provider == "mock"
    assert response.model == "mock-model"
    assert response.content.startswith("[mock:mock-model]")
    assert response.usage.input_tokens > 0


def test_mock_structured_simple():
    provider = MockProvider("mock", ProviderConfig(type="mock"))
    response = provider.structured(make_request(), "m", SimpleSchema)
    assert response.data == {"verdict": "mock-verdict", "confidence": 0.5, "items": ["mock-items"]}


def test_mock_structured_nested():
    provider = MockProvider("mock", ProviderConfig(type="mock"))
    response = provider.structured(make_request(), "m", NestedSchema)
    assert response.data["detail"] == {
        "verdict": "mock-verdict",
        "confidence": 0.5,
        "items": ["mock-items"],
    }
    assert response.data["note"] == "mock-note"


class FlakyProvider(Provider):
    def __init__(self):
        super().__init__("flaky", ProviderConfig(type="mock"))
        self.attempts = 0

    def complete(self, request, model):
        self.attempts += 1
        inner = MockProvider(self.name, self.config)
        response = inner.complete(request, model)
        if self.attempts == 1:
            response.content = "not json at all"
        else:
            response.content = json.dumps({"verdict": "ok", "confidence": 0.9, "items": []})
        return response


def test_structured_retries_on_invalid_json():
    provider = FlakyProvider()
    response = provider.structured(make_request(), "m", SimpleSchema)
    assert response.data is not None
    assert response.retries == 1
    assert provider.attempts == 2


class BrokenProvider(Provider):
    def __init__(self):
        super().__init__("broken", ProviderConfig(type="mock"))

    def complete(self, request, model):
        response = MockProvider(self.name, self.config).complete(request, model)
        response.content = "nope"
        return response


def test_structured_fails_after_retries():
    with pytest.raises(ProviderError, match="failed after"):
        BrokenProvider().structured(make_request(), "m", SimpleSchema)


def test_cache_hit_and_miss(tmp_path):
    inner = CountingProvider()
    cached = CacheProvider(inner, tmp_path / "cache.sqlite")
    first = cached.complete(make_request(), "m")
    second = cached.complete(make_request(), "m")
    assert first.cached is False
    assert second.cached is True
    assert inner.calls == 1
    assert first.content == second.content


def test_cache_structured_hit_and_miss(tmp_path):
    inner = CountingProvider()
    cached = CacheProvider(inner, tmp_path / "cache.sqlite")
    first = cached.structured(make_request(), "m", SimpleSchema)
    second = cached.structured(make_request(), "m", SimpleSchema)
    assert first.cached is False
    assert second.cached is True
    assert inner.calls == 1
    assert first.data is not None
    assert second.data is not None


def test_cache_disabled(tmp_path):
    inner = CountingProvider()
    cached = CacheProvider(inner, tmp_path / "cache.sqlite", enabled=False)
    cached.complete(make_request(), "m")
    cached.complete(make_request(), "m")
    assert inner.calls == 2


def test_cache_differentiates_requests(tmp_path):
    inner = CountingProvider()
    cached = CacheProvider(inner, tmp_path / "cache.sqlite")
    cached.complete(make_request(messages=[ChatMessage(role="user", content="one")]), "m")
    cached.complete(make_request(messages=[ChatMessage(role="user", content="two")]), "m")
    assert inner.calls == 2


def test_openai_compat_success():
    calls = []
    provider = make_provider_with_responses([httpx.Response(200, json=COMPLETION_BODY)], calls)
    response = provider.complete(make_request(), "m")
    assert len(calls) == 1
    assert response.content == "hello"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.provider == "test"


def test_openai_compat_retries_on_500(monkeypatch):
    monkeypatch.setattr("brainforge.providers.openai_compat.time.sleep", lambda s: None)
    calls = []
    provider = make_provider_with_responses(
        [
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(200, json=COMPLETION_BODY),
        ],
        calls,
        max_retries=2,
    )
    response = provider.complete(make_request(), "m")
    assert len(calls) == 2
    assert response.retries == 1


def test_openai_compat_no_retry_on_400():
    calls = []
    provider = make_provider_with_responses(
        [httpx.Response(400, json={"error": {"message": "bad request"}})],
        calls,
        max_retries=3,
    )
    with pytest.raises(ProviderError, match="HTTP 400"):
        provider.complete(make_request(), "m")
    assert len(calls) == 1


def test_openai_compat_exhausted_retries(monkeypatch):
    monkeypatch.setattr("brainforge.providers.openai_compat.time.sleep", lambda s: None)
    provider = make_provider_with_responses(
        [httpx.Response(503, json={"error": "down"})], max_retries=1
    )
    with pytest.raises(ProviderUnavailableError, match="after 2 attempts"):
        provider.complete(make_request(), "m")


def test_zen_rejects_unsupported_api_style():
    config = ProviderConfig(type="zen", api_style=ApiStyle.ANTHROPIC)
    with pytest.raises(ProviderNotSupportedError, match="api_style"):
        OpenAICompatProvider("zen", config)


def test_build_provider_all_types():
    for provider_type in (
        "openrouter",
        "zen",
        "mlgw",
        "openai",
        "local",
        "mock",
    ):
        config = ProviderConfig(type=provider_type)
        provider = build_provider("test", config)
        assert provider.name == "test"


def test_estimate_cost_computation():
    from brainforge.config.models import ModelDef, PricingConfig
    from brainforge.providers.base import Usage

    model = ModelDef(
        provider="p",
        model="m",
        pricing=PricingConfig(input_per_mtok=1.0, output_per_mtok=2.0),
    )
    response = MockProvider("p", ProviderConfig(type="mock")).complete(make_request(), "m")
    response.usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_cost(model, response)
    assert cost == 3.0


def test_usage_logger(tmp_path):
    logger = UsageLogger(tmp_path / "logs")
    provider = MockProvider("mock", ProviderConfig(type="mock"))
    response = provider.complete(make_request(), "m")
    logger.log(response, role="security_teacher", cost_usd=0.01)
    lines = (tmp_path / "logs" / "usage.jsonl").read_text().strip().splitlines()
    entry = json.loads(lines[0])
    assert entry["role"] == "security_teacher"
    assert entry["provider"] == "mock"
    assert entry["cost_usd"] == 0.01
    assert entry["cache_hit"] is False
