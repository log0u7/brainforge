import json

from pydantic import BaseModel

from brainforge.providers.base import ChatRequest, ChatResponse, Provider, Usage

_MAX_DEPTH = 20


class MockProvider(Provider):
    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        last = request.messages[-1].content if request.messages else ""
        preview = last[:80].replace("\n", " ")
        return ChatResponse(
            content=f"[mock:{model}] {preview}",
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=max(1, len(last) // 4), output_tokens=16),
        )

    def structured(self, request: ChatRequest, model: str, schema: type[BaseModel]) -> ChatResponse:
        data = self.generate(schema)
        return ChatResponse(
            content=json.dumps(data),
            data=data,
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=64, output_tokens=64),
        )

    def generate(self, schema: type[BaseModel]) -> dict:
        js = schema.model_json_schema()
        value = self._instance(js, js.get("$defs", {}), 0, None)
        if not isinstance(value, dict):
            raise ValueError(f"schema '{schema.__name__}' did not produce a JSON object")
        return value

    def _instance(self, node: dict, defs: dict, depth: int, key: str | None) -> object:
        if depth > _MAX_DEPTH:
            raise ValueError("json schema instance generation too deep")
        if "$ref" in node:
            return self._instance(defs[node["$ref"].rsplit("/", 1)[-1]], defs, depth + 1, key)
        if "const" in node:
            return node["const"]
        if "enum" in node:
            return node["enum"][0]
        if "anyOf" in node:
            for sub in node["anyOf"]:
                if sub.get("type") != "null":
                    return self._instance(sub, defs, depth + 1, key)
            return None
        if "default" in node and node["default"] is not None:
            return node["default"]
        node_type = node.get("type")
        if node_type == "object":
            return {
                prop: self._instance(sub, defs, depth + 1, prop)
                for prop, sub in node.get("properties", {}).items()
            }
        if node_type == "array":
            items = node.get("items")
            return [self._instance(items, defs, depth + 1, key)] if items else []
        if node_type == "string":
            examples = node.get("examples")
            if examples:
                return examples[0]
            if key == "cwe":
                return "CWE-78"
            return f"mock-{key}" if key else "mock"
        if node_type == "integer":
            low = node.get("minimum", 0)
            high = node.get("maximum", low + 1)
            return max(low, min(high, low + 1))
        if node_type == "number":
            low = node.get("minimum", 0.0)
            high = node.get("maximum", low + 1.0)
            return max(low, min(high, (low + high) / 2))
        if node_type == "boolean":
            return True
        if node_type == "null":
            return None
        return None
