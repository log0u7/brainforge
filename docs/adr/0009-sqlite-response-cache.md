# 0009 - sqlite response cache keyed on the full request

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

Teacher calls are the dominant cost of dataset generation (tokens and time).
Re-running a pipeline (a crash after 40 cases, a gate tweak, a re-index) must
not re-pay for identical calls, yet results must be identical to what a fresh
call would return, including when the RAG context changed.

## Decision Drivers

* Same input (provider, model, messages, params, RAG context) -> same output,
  no API call.
* A changed RAG index must invalidate the affected cache entries.
* No server infrastructure; the cache lives in the project directory.
* Bypass must be explicit (`--fresh`) for regeneration runs.

## Considered Options

* No caching
* Filesystem blob cache (sha256 -> file)
* Redis
* sqlite cache keyed on sha256 of the full logical request (chosen)
* sqlite cache keyed on prompt text only

## Decision Outcome

Chosen option: "sqlite cache keyed on sha256 of the full logical request".
`providers/cache.py` wraps any provider (`CacheProvider`): the key hashes
provider name, model, messages, temperature, `top_p`, `max_tokens` and
`cache_extra` (a hash of the RAG context and case content supplied by the
engine). Responses are stored as serialized `ChatResponse` payloads and
replayed with `cached: true`. The engine sets `cache_extra` from case + RAG
texts, so re-indexing changes keys naturally. `pipeline run --fresh` disables
the wrapper.

### Consequences

* Re-runs after crashes or gate changes are effectively free for completed
  steps.
* The cache is content-addressed: prompt tweaks legitimately miss.
* sqlite handles concurrent CLI runs poorly for heavy parallelism (single
  writer); acceptable at MVP scale, file-per-provider layout limits blast
  radius.

### Confirmation

* `tests/unit/test_providers.py` covers hit, miss, disabled and
  request-differentiation cases against a counting inner provider.
* `data/logs/usage.jsonl` shows `cache_hit` per call.

## Pros and Cons of the Options

### No caching

* Good, because zero code.
* Bad, because every crash or gate iteration re-pays tokens for already
  verified answers.

### Filesystem blobs

* Good, because trivially inspectable.
* Bad, because no atomic listing/stats and directory explosion at scale.

### Redis

* Good, because concurrent access and TTLs are built-in.
* Bad, because a server for a single-workstation tool violates the
  local-first constraint.

### Full-request sqlite key (chosen)

* Good, because one file per provider, content-addressed, with `cache_extra`
  folding RAG state into the key.
* Bad, because long entries accumulate; the file is disposable
  (`data/cache/`, gitignored) and can be deleted without risk.

### Prompt-text-only key

* Good, because shorter keys.
* Bad, because temperature, model or RAG changes would silently return stale
  results; that violates the correctness driver.

## Links

* [Providers reference](../reference/providers.md)
* ADR-0005 (RAG state feeding `cache_extra`)
* `src/brainforge/providers/cache.py`
