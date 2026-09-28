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
uv run pytest -m integration        # cross-boundary subprocess tests (BRAINFORCE_IT=1)
uv run ruff check . && uv run ruff format .   # lint + format
uv run brainforge config validate   # after touching config/
uv run brainforge config schema --check       # schema must stay in sync
```

Branches follow the gitflow described in
[Branches, worktrees and reviews](#branches-worktrees-and-reviews): one topic
per branch, `feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `docs/<topic>`,
landed through a merge request.

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
- Integration tests run the CLI chain as a subprocess with the mock provider
  (no API keys, no GPU); they are marked `integration` and gated by
  `BRAINFORCE_IT=1`.
- The training smoke test runs only on a GPU runner (manual job).

### Red-green-refactor

- Every new feature or bugfix starts with a failing test (red), then the
  minimal code that makes it pass (green), then refactor with tests green.
- Bug reports first get a regression test that reproduces the bug, then the fix.
- No production code lands without a test demanding its existence.
- `make mutate` after touching core logic; every surviving mutant is either
  killed by strengthening a test or justified explicitly.
- PR checklist: ruff clean, mypy clean, tests green, coverage >= `fail_under`
  (`pyproject.toml`), mutation survivors triaged when core logic changed.

## Documentation and ADRs

The docs follow the Diataxis split (tutorials, how-to, reference,
explanation); put new content in the section that matches the reader's goal,
and cross-link across sections.

Architectural decisions are recorded as MADR 3.0 ADRs in `docs/adr/`: number
never reused, rejected options included, confirmation section pointing at
tests or CLI commands. The full rules and the workflow are in
[docs/adr/index.md](docs/adr/index.md) and
[docs/how-to/write-an-adr.md](docs/how-to/write-an-adr.md).

## Branches, worktrees and reviews

A simple, battle-tested gitflow for team work:

- `main` is protected: release-ready only, no direct pushes; everything lands
  through a merge request.
- One topic per branch, cut from `main`: `feat/<topic>`, `fix/<topic>`,
  `chore/<topic>`, `docs/<topic>`.
- One session per worktree. The main clone stays on `main` and is never
  mutated by working sessions; do the work in a worktree instead:

```bash
git worktree add ~/projets/wt/brainforge/<topic> -b feat/<topic>
cd ~/projets/wt/brainforge/<topic>
uv sync               # worktrees have their own .venv
```

- Rebase on `main` before opening the MR (`git fetch && git rebase origin/main`).
- Merge request flow:
  1. Push the branch and open the MR (`gh pr create`); describe what changed
     and why; link related issues.
  2. CI must be green (lint, secrets, tests, config, build, audit).
  3. At least one approval; the reviewer pays special attention to provenance
     and contamination handling.
  4. Merge with a merge commit (history stays readable one topic at a time)
     and delete the branch.
- Clean up once the work is merged:

```bash
git worktree remove ~/projets/wt/brainforge/<topic>
```

## Releases

- `CHANGELOG.md` accumulates changes under `[Unreleased]` (Keep a Changelog
  format); English only.
- A release moves `[Unreleased]` into a dated section, bumps `version` in
  `pyproject.toml` and lands as a `chore(release): X.Y.Z` commit inside an MR.
- Tag `vX.Y.Z` only after the merge lands on `main`, on the exact merge
  commit, pushed one by one:

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

- SemVer: `0.x` is unstable; breaking changes bump the minor until 1.0.0.

## Reporting issues

Include: BrainForge version (`brainforge --version` equivalent: pyproject
version), the failing command, full traceback, and a minimal config/case that
reproduces the problem (redact secrets and API keys).
