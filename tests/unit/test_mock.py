import pytest
from pydantic import BaseModel

from brainforge.config.models import ProviderConfig
from brainforge.providers.base import ChatMessage, ChatRequest
from brainforge.providers.mock import MockProvider


class SimpleSchema(BaseModel):
    verdict: str


class AllTypes(BaseModel):
    const_field: str
    enum_field: str
    maybe: int | None
    default_field: str
    flags: list[bool]
    cwe: str
    count: int
    ratio: float


def make_provider() -> MockProvider:
    return MockProvider("mock", ProviderConfig(type="mock"))


def make_request(content: str = "hello\nworld") -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=content)])


def test_generate_all_branches():
    data = make_provider().generate(AllTypes)
    assert isinstance(data["const_field"], str)
    assert isinstance(data["enum_field"], str)
    assert data["maybe"] is not None
    assert isinstance(data["default_field"], str)
    assert data["flags"] == [True]
    assert data["cwe"] == "CWE-78"
    assert isinstance(data["count"], int)
    assert isinstance(data["ratio"], float)


def test_generate_rejects_non_object():
    provider = make_provider()
    provider._instance = lambda node, defs, depth, key: []
    with pytest.raises(ValueError, match="did not produce a JSON object"):
        provider.generate(SimpleSchema)


def test_depth_limit():
    node = {"type": "object", "properties": {}}
    with pytest.raises(ValueError, match="too deep"):
        make_provider()._instance(node, {}, 21, None)


def test_ref_resolves_to_def():
    schema = {"$defs": {"Thing": {"type": "string"}}, "$ref": "#/$defs/Thing"}
    assert make_provider()._instance(schema, schema["$defs"], 0, None) == "mock"


def test_scalar_fallbacks():
    instance = make_provider()._instance
    assert instance({"type": "array"}, {}, 0, None) == []
    assert instance({"type": "boolean"}, {}, 0, None) is True
    assert instance({"type": "null"}, {}, 0, None) is None
    assert instance({"type": "unknown"}, {}, 0, None) is None
    assert instance({"anyOf": [{"type": "null"}]}, {}, 0, None) is None
    assert instance({"type": "integer", "minimum": 5}, {}, 0, "n") == 6
    assert instance({"type": "integer", "minimum": 5, "maximum": 9}, {}, 0, "n") == 6
    assert instance({"type": "number", "minimum": 2.0, "maximum": 8.0}, {}, 0, "r") == 5.0
    assert instance({"type": "string", "examples": ["a"]}, {}, 0, "s") == "a"
    assert instance({"type": "string", "minimum": 1}, {}, 0, "s") == "mock-s"
    assert instance({"type": "string"}, {}, 0, "cwe") == "CWE-78"
    assert instance({"type": "string"}, {}, 0, None) == "mock"
