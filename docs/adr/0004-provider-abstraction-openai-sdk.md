# 0004 - Provider abstraction on the OpenAI SDK with schema-in-prompt structured outputs

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

Teachers run on heterogeneous gateways (OpenRouter, OpenCode Zen, MLGW, local
servers) that expose different endpoint dialects. Every teacher call must
return a *validated structured object* (pydantic), because raw text cannot be
quality-gated reliably. The abstraction must not leak any vendor into the
pipeline or domain packs.

## Decision Drivers

* One interface (`chat`/`structured`) for all providers; domain packs must
  never see a vendor SDK.
* Structured outputs must work on gateways that do not support native JSON
  response formats (llama.cpp, Ollama, older gateways).
* Retries, timeouts and usage accounting must be uniform.
* Minimal dependency surface; prefer maintained building blocks.

## Considered Options

* Hand-rolled HTTP client on httpx
* Per-vendor SDKs (openai, anthropic, google, ...)
* OpenAI SDK for the compatible dialects + schema-in-prompt structured outputs
* Native structured-output parameters only (`response_format`)

## Decision Outcome

Chosen option: "OpenAI SDK for the compatible dialects + schema-in-prompt
structured outputs".

`providers/base.py` defines the `Provider` interface (`complete`,
`structured`) plus JSON extraction; `providers/openai_compat.py` implements the
OpenAI-compatible dialect on the `openai` SDK with retries (429/5xx),
timeouts and usage extraction. `structured()` injects the JSON Schema into the
system prompt, parses the reply (fences and embedded objects tolerated),
validates against the pydantic schema and retries with a corrective message on
failure. OpenRouter, Zen, MLGW and local adapters are thin subclasses.

### Consequences

* Structured outputs work everywhere, including gateways without
  `response_format`.
* Schema enforcement is probabilistic (the model can still fail); the parse
  retry loop plus the quality gate catch residual failures, and
  `ProviderError` is explicit after retries.
* The `openai` SDK is a hard dependency; `httpx` injection is available for
  tests (`http_client` parameter).

### Confirmation

* `tests/unit/test_providers.py` covers JSON extraction, parse retries,
  retry/backoff on 5xx, no-retry on 4xx and exhausted retries, using an
  injected `httpx.MockTransport`.
* `brainforge config validate` dry-builds every provider, so unsupported
  `api_style` values fail at validation time.

## Pros and Cons of the Options

### Hand-rolled httpx client

* Good, because zero vendor SDK dependencies.
* Bad, because retries, streaming, usage parsing and auth edge cases get
  re-implemented per dialect; the SDK is battle-tested.

### Per-vendor SDKs

* Good, because each dialect is natively supported.
* Bad, because it multiplies dependencies and abstractions; only the OpenAI
  dialect (and later anthropic-style endpoints) is needed for MVP teachers.

### OpenAI SDK + schema-in-prompt (chosen)

* Good, because one dependency covers OpenRouter, Zen, MLGW and local gateways.
* Good, because structured outputs degrade gracefully on limited gateways.
* Bad, because prompt-injected schemas cost tokens and are not contractually
  enforced.

### Native `response_format` only

* Good, because the provider contractually guarantees the JSON shape.
* Bad, because MLGW/Ollama builds frequently lack it; support would fork the
  code path and break the "works everywhere" driver.

## Links

* [Providers reference](../reference/providers.md)
* ADR-0006 (packs never import vendor SDKs)
* `src/brainforge/providers/base.py`, `src/brainforge/providers/openai_compat.py`
