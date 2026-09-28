# AGENTS.md

Guidance for coding agents working in this repository. Read CONTRIBUTING.md
for the full setup, gitflow and review process.

## What this is

BrainForge generates verified security/coding training datasets through a
multi-teacher LLM pipeline (teachers, adversarial critic, independent judge)
and fine-tunes a student model with QLoRA (TRL). No custom transformer code
lives here; model internals come from Hugging Face.

## Commands

- `make test`: unit tests. Integration tests are opt-in:
  `BRAINFORCE_IT=1 uv run pytest -m integration`.
- `make lint && make mypy`: ruff + mypy. Run on a plain `uv sync` environment.
- `make ci`: the full local CI gate (lint, secrets, test, config, build, audit).
- `make config`: validate configuration and keep `config/schema.json` in sync.
  After touching `src/brainforge/config/models.py`, regenerate with
  `uv run brainforge config schema --write` (CI fails otherwise).
- `make train-smoke-cpu`: CPU smoke training; needs `uv sync --extra training`.
- `make mutate`: mutation testing after touching core logic.

## Non-obvious rules

- Heavy dependencies (torch, trl, peft, transformers, bitsandbytes) are
  imported lazily inside functions. The base package and the unit test suite
  must run without the `training` extra. Never add a module-level import of
  them.
- TDD is mandatory: a failing test first, then the minimal implementation.
  Bug fixes get a regression test reproducing the bug before the fix.
- Conventional Commits, English only (code, comments, docs, changelog).
- The provider abstraction (`src/brainforge/providers/`) is the only way to
  reach an LLM API; domain packs must not import provider SDKs.
- Dataset-facing changes must preserve provenance and contamination tagging
  (`source_date`, `recitation_risk`, judge independence).
- Secrets never land here: gitleaks runs in pre-commit and CI.

## Gitflow (details in CONTRIBUTING.md)

- `main` is protected: no direct pushes; everything lands through a merge
  request.
- One topic per branch, cut from `main`: `feat|fix|chore|docs/<topic>`.
- One session = one worktree:
  `git worktree add ~/projets/wt/brainforge/<topic> -b <topic>`.
  The main clone stays on `main` and is never mutated by working sessions.
- MR flow: push branch, `gh pr create`, CI green, at least one approval,
  merge with a merge commit, delete the branch, remove the worktree.
- Tags `vX.Y.Z` go on `main` only after the merge lands, on the exact merge
  commit, pushed one by one.
