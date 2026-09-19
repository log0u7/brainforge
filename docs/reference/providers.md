# Providers reference

All providers implement the same interface
(`src/brainforge/providers/base.py`), decided in
[ADR-0004](../adr/0004-provider-abstraction-openai-sdk.md):

- `complete(request, model) -> ChatResponse`: plain completion with usage
  stats (input/output tokens, latency, retries).
- `structured(request, model, schema) -> ChatResponse`: injects the JSON
  Schema into the system prompt, parses the reply (fenced or embedded JSON
  tolerated), validates against the pydantic schema, and retries with a
  corrective message on invalid output (2 parse retries by default).

## OpenAI-compatible core

`OpenAICompatProvider` (on the `openai` SDK) provides base URLs, API keys,
timeouts, retry/backoff on 429/5xx, immediate failure on other 4xx, and usage
extraction. Accepts an injected `httpx.Client` for testing.

## OpenRouter

```json
{"type": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "${OPENROUTER_API_KEY}"}
```

Model ids are OpenRouter model strings. Declare them in `models`; never
hardcode one in a pipeline. Cost estimates: add `pricing` on the model.

## OpenCode Zen ("ZenCode Go")

```json
{"type": "zen", "base_url": "${ZENCODE_BASE_URL:https://opencode.ai/zen/v1}", "api_key": "${ZENCODE_API_KEY}", "api_style": "chat_completions"}
```

Zen exposes three endpoint dialects:

| `api_style` | Endpoint | Model families |
|---|---|---|
| `chat_completions` | `/chat/completions` | GLM, Kimi, DeepSeek, MiniMax, free models |
| `responses` | `/responses` | GPT, Grok families |
| `anthropic` | `/messages` | Claude, Qwen Plus/Max |

Only `chat_completions` is implemented; unsupported styles raise
`ProviderNotSupportedError` when the provider is built, so
`brainforge config validate` catches them before any run. The full model list:
`GET https://opencode.ai/zen/v1/models`. Roadmap: `responses` and `anthropic`
styles.

## MLGW

```json
{"type": "mlgw", "base_url": "${MLGW_BASE_URL:http://localhost:8080/v1}", "api_key": "${MLGW_API_KEY:}"}
```

Local OpenAI-compatible gateway (llama.cpp, Ollama, vLLM backends). With an
empty key the provider fails at call time; set `MLGW_API_KEY` to use it.

## Local

```json
{"type": "local", "base_url": "http://localhost:11434/v1"}
```

Defaults to the Ollama OpenAI-compatible endpoint. The training student does
not depend on the inference provider.

## Mock

```json
{"type": "mock"}
```

Deterministic provider that generates schema-valid structured outputs from the
JSON Schema itself: first enum value, `CWE-78` for `cwe` keys, `mock-<key>`
strings, midpoint numbers for bounded fields, one-element arrays. Powers
`--provider mock` runs and the test suite at zero cost.

## Cache

`CacheProvider` wraps any provider with a sqlite cache
(`data/cache/<provider>.sqlite`), keyed on
`sha256(provider, model, messages, temperature, top_p, max_tokens,
cache_extra)` where `cache_extra` hashes case + RAG content. Hits replay the
stored response with `cached: true`; `--fresh` bypasses the wrapper.
Decision: [ADR-0009](../adr/0009-sqlite-response-cache.md).

## Observability

`UsageLogger` appends one JSON line per call to `data/logs/usage.jsonl`:

```json
{"ts": "2026-08-31T15:20:00+0000", "provider": "openrouter", "model": "deepseek-v4-pro", "role": "security_teacher", "latency_ms": 812.4, "input_tokens": 1450, "output_tokens": 320, "retries": 0, "cache_hit": false, "cost_usd": 0.0021, "error": null}
```

`cost_usd` requires `pricing` on the model; otherwise `null`.
