# Providers

All providers implement the same interface (`providers/base.py`):

- `complete(request, model) -> ChatResponse`: plain completion with usage stats.
- `structured(request, model, schema) -> ChatResponse`: injects the JSON Schema
  into the system prompt, parses and validates the reply against the pydantic
  schema, retries on invalid JSON.

## OpenAI-compatible core

`OpenAICompatProvider` (built on the `openai` SDK) handles base URLs, API keys,
timeouts, retries with backoff (429/5xx), and usage extraction.
`OpenRouterProvider`, `ZenProvider`, `MLGWProvider` and `LocalProvider` are
thin subclasses with sensible default base URLs.

### OpenRouter

```json
{"type": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "${OPENROUTER_API_KEY}"}
```

Model ids are OpenRouter model strings. Never hardcode one in the pipeline;
declare it in `models`.

### OpenCode Zen ("ZenCode Go")

```json
{"type": "zen", "base_url": "https://opencode.ai/zen/v1", "api_key": "${ZENCODE_API_KEY}", "api_style": "chat_completions"}
```

Zen exposes three endpoint families:

| Style | Endpoint | Models |
|---|---|---|
| `chat_completions` | `/chat/completions` | GLM, Kimi, DeepSeek, MiniMax, free models |
| `responses` | `/responses` | GPT, Grok families |
| `anthropic` | `/messages` | Claude, Qwen Plus/Max |

Only `chat_completions` is implemented; the others raise
`ProviderNotSupportedError` at provider build time (see ROADMAP). The full
model list is available at `GET https://opencode.ai/zen/v1/models`.

### MLGW

```json
{"type": "mlgw", "base_url": "http://localhost:8080/v1", "api_key": "${MLGW_API_KEY:deadbeef}"}
```

Local OpenAI-compatible gateway (llama.cpp, Ollama, vLLM backends). The
`deadbeef` development key is intentionally fake.

### Local

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
JSON Schema itself (first enum value, `CWE-78` for `cwe` keys, keyed strings,
bounded numeric midpoints). Powers `--provider mock` runs and the test suite
without any API cost.

## Cache and observability

`CacheProvider` wraps any provider with a sqlite cache keyed on
`sha256(provider, model, messages, params, cache_extra)` where `cache_extra`
hashes the RAG context, so identical inputs never re-hit the API unless
`--fresh` is passed. `UsageLogger` appends one JSON line per call to
`data/logs/usage.jsonl` (tokens, latency, retries, cache hit, estimated cost
when `pricing` is configured).
