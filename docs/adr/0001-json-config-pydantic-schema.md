# 0001 - JSON configuration validated by pydantic-generated JSON Schema

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

BrainForge needs a configuration format for providers, models, roles and
pipelines that is machine-validated, secret-safe and pleasant to edit by hand.
The format must support environment variable expansion for secrets, and the
validation rules must stay in sync with the runtime types without manual
effort.

## Decision Drivers

* Secrets must never be committed; expansion from environment variables is
  required.
* Validation errors must be explicit and early (fail fast at load time).
* Runtime types and validation must not drift apart.
* The format must stay readable in pull requests and CI logs.

## Considered Options

* JSON with a hand-maintained JSON Schema
* JSON validated by pydantic models, with the schema generated
* YAML with a schema
* TOML with a schema
* Python module as configuration

## Decision Outcome

Chosen option: "JSON validated by pydantic models, with the schema
generated". The pydantic models in `src/brainforge/config/models.py` are the
single source of truth; `brainforge config schema --write` regenerates
`config/schema.json` and `brainforge config schema --check` fails when they
drift. `${VAR}` and `${VAR:default}` expansion runs before validation, so an
unset secret is a hard error at load time.

### Consequences

* No comments in configuration files (JSON limitation); the docs carry the
  explanations instead.
* Schema drift is caught in CI, not in production.
* Everything downstream benefits from typed objects instead of raw dicts.

### Confirmation

* `tests/unit/test_config.py::test_schema_roundtrip` checks the sync.
* The CI `config` job runs both `brainforge config validate` and
  `brainforge config schema --check`.

## Pros and Cons of the Options

### JSON + pydantic-generated schema

* Good, because types and validation cannot drift (single source of truth).
* Good, because validation errors from pydantic are precise and typed.
* Bad, because JSON forbids comments.

### JSON + hand-maintained schema

* Good, because the schema file is freely editable.
* Bad, because the schema and the runtime types drift silently.

### YAML

* Good, because comments and anchors are convenient.
* Bad, because YAML is an injection-prone, error-prone format (Norway
  problem, tabs) and adds a dependency for no structural gain.

### TOML

* Good, because it is comments-friendly.
* Bad, because deeply nested structures (providers -> models -> roles ->
  pipelines) become hard to scan in TOML.

### Python module as configuration

* Good, because it allows arbitrary logic.
* Bad, because configuration becomes executable code: secrets leak into
  imports, diffing and reviewing get harder, and non-Python tooling cannot
  read it.

## Links

* [Configuration reference](../reference/configuration.md)
* `src/brainforge/config/loader.py`
* Supersedes nothing; foundational decision.
