# 0008 - Tooling: uv pinned by mise, src layout, ruff and pytest

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

The project needs reproducible installs (CI and workstation), a clean
packaging boundary, and one fast toolchain for linting and testing, without
scattering tool versions across developer machines.

## Decision Drivers

* Lockfile-based reproducibility for CI and local runs.
* The Python version is managed by mise on the workstation; the tool manager
  must cooperate, not fight it.
* Fast iteration: lint and tests must run in seconds.
* Packaging must work for `uv build` and future extras (`[rag]`, `[training]`).

## Considered Options

* pip + requirements files + venv
* Poetry
* uv pinned in mise.toml, src layout (chosen)
* uv without mise (standalone install script)

## Decision Outcome

Chosen option: "uv pinned in mise.toml, src layout". `mise.toml` pins
`uv = "0.12.7"` so every contributor and CI job resolves the same uv through
`mise install`. The package lives under `src/brainforge/` (hatchling build
backend), preventing accidental imports of uninstalled code. Dev tools
(pytest, ruff, pre-commit, mkdocs-material) live in the `[dependency-groups]`
dev group; heavy optional stacks are extras. Ruff covers lint and format with
per-file ignores for typer's `B008` pattern; pytest runs everything except
integration-marked tests by default.

### Consequences

* `uv.lock` is committed; CI caches `.uv-cache` keyed on it.
* Contributors need mise (or any uv 0.12.x); the standalone installer remains
  a documented fallback.
* Training deps (torch, TRL, ...) never slow down the core install.

### Confirmation

* `mise install && uv sync` is the documented setup and the CI image uses the
  uv container.
* `tests/` imports the installed package (would fail with a broken src
  layout); `uv build` produces sdist+wheel in the CI `build` job.

## Pros and Cons of the Options

### pip + requirements files

* Good, because universally understood.
* Bad, because resolution is slow, lockfile hygiene is manual, and extras
  management is error-prone.

### Poetry

* Good, because it is mature and lockfile-based.
* Bad, because it duplicates the runtime manager (mise) and is slower for
  workspace-style workflows.

### uv + mise + src layout (chosen)

* Good, because one pinned binary covers sync, lock, build and script running;
  mise respects the user's existing runtime management.
* Bad, because uv is a young tool; the lockfile pins mitigate churn.

### uv without mise

* Good, because one less requirement.
* Bad, because version drift across machines reintroduces "works on my
  machine"; pinning in `mise.toml` is one line.

## Links

* [CONTRIBUTING](https://gitlab.com/6admin.io/brainforge/-/blob/main/CONTRIBUTING.md)
* `mise.toml`, `pyproject.toml`, `.gitlab-ci.yml`
