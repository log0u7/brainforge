# Contributing to BrainForge

Thanks for helping. This document covers setup, conventions and the review flow.

## Development setup

Requirements: Python 3.12+, mise (recommended) or uv.

```bash
git clone <repo-url> && cd brainforge
mise install        # installs the pinned uv from mise.toml
uv sync             # creates .venv and installs everything (incl. dev tools)
pre-commit install  # ruff + gitleaks hooks
```

Optional dependency groups:

```bash
uv sync --extra rag        # sqlite-vec vector backend
uv sync --extra training   # torch, transformers, peft, trl, bitsandbytes
```

## Daily workflow

```bash
uv run pytest                       # unit tests (integration tests are opt-in)
uv run pytest -m integration        # only with BRAINFORCE_IT=1 and real API keys
uv run ruff check . && uv run ruff format .   # lint + format
uv run brainforge config validate   # after touching config/
uv run brainforge config schema --check       # schema must stay in sync
```

Branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`. Keep changes small and
focused; one topic per pull request.

## Commits

Conventional Commits, English only:

```text
feat(pipeline): add cost-mode step filtering
fix(rag): guard empty embedding batches
docs: rewrite dataset contamination section
```

## Rules of the codebase

- Comments and identifiers in English; no secrets, ever (gitleaks runs in CI
  and pre-commit).
- The provider abstraction is the only way to reach an LLM API; domain packs
  must not import provider SDKs.
- Every dataset-facing feature must preserve provenance (teachers, judge,
  source dates, RAG chunks).
- New provider? Subclass `OpenAICompatProvider` or implement `Provider`; add a
  unit test with a mocked transport; never hardcode model ids.
- New domain? Create a pack under `src/brainforge/domains/<name>/` with
  schemas, prompts and a quality gate; register it; add tests. See
  [docs/domain-packs.md](docs/domain-packs.md).
- Contamination discipline: any feature touching dataset generation must keep
  `source_date` propagation and `recitation_risk` tagging intact.

## Tests

- Unit tests are mandatory for new code paths (see `tests/unit/`).
- Integration tests hitting real providers are marked `integration` and gated
  by `BRAINFORCE_IT=1`.
- The training smoke test runs only on a GPU runner (manual job).

## Documentation and ADRs

The docs follow the Diataxis split (tutorials, how-to, reference,
explanation); put new content in the section that matches the reader's goal,
and cross-link across sections.

Architectural decisions are recorded as MADR 3.0 ADRs in `docs/adr/`: number
never reused, rejected options included, confirmation section pointing at
tests or CLI commands. The full rules and the workflow are in
[docs/adr/index.md](docs/adr/index.md) and
[docs/how-to/write-an-adr.md](docs/how-to/write-an-adr.md).

## Pull requests

1. Rebase on `main`; CI must pass (lint, secrets, tests, config, build).
2. Describe what changed and why; link related issues.
3. One approval required; the reviewer pays special attention to provenance
   and contamination handling.

## Reporting issues

Include: BrainForge version (`brainforge --version` equivalent: pyproject
version), the failing command, full traceback, and a minimal config/case that
reproduces the problem (redact secrets and API keys).
