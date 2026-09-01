# CI reference

The same six gates (lint, secrets, test, config, build, audit) run on every
platform, each in its native syntax:

| Platform | File | Status |
|---|---|---|
| GitHub Actions | `.github/workflows/ci.yml` | reference implementation; branch protection on `main` requires all six checks |
| GitLab CI | `.gitlab-ci.yml` | parity port (pinned `uv==0.12.7` via pip, gitleaks image `v8.30.1`) |
| Forgejo (act_runner) | `.forgejo/workflows/ci.yml` | written, not yet tested against a live runner (5 jobs: no build artifact, no GPU) |
| Local | `make ci` | exact equivalent of the pipeline, no CI server needed (needs `mise`/`uv` and the `gitleaks` binary on PATH) |

GitHub Actions also has `gpu-smoke.yml` (manual) and GitLab has a manual
`train-smoke` job with `tags: [gpu]`; both only run where a self-hosted GPU
runner with the `gpu` label is registered.

## GitHub Actions

`ci.yml` runs on pushes to `main` and on pull requests targeting `main`;
superseded runs on the same ref are cancelled (`concurrency`).
`gpu-smoke.yml` runs on `workflow_dispatch` only.

## Workflows

### `.github/workflows/ci.yml` (automatic)

| Job | Content | Runs |
|---|---|---|
| `lint` | `uv sync`, `ruff check .`, `ruff format --check .` | every run |
| `secrets` | gitleaks over full history (`fetch-depth: 0`), SARIF report | every run |
| `test` | `pytest -m "not integration"` with coverage; `coverage.xml` uploaded as a 7-day artifact | every run |
| `config` | `brainforge config validate`, `config schema --check`, mock pipeline over `examples/cases` | every run |
| `build` | `uv build`, sdist+wheel artifact (7 days) | every run |
| `audit` | `uv export` of runtime requirements, `pip-audit --strict` | every run |

All six jobs run in parallel on `ubuntu-latest`. Dependabot opens weekly
update PRs for pip and GitHub Actions dependencies.

## Conventions

- Toolchain: GitHub and Forgejo install uv `0.12.7` through mise
  (`jdx/mise-action@v4` reading `mise.toml`, single source of truth); GitLab
  installs `uv==0.12.7` via pip inside a `python:3.12-bookworm` container
  (runner-friendly, registry-pinned). Everywhere else `uv sync` uses the
  `uv.lock`-keyed cache.
- `permissions: contents: read` at workflow level; nothing else is granted.
- Integration tests hitting real providers are excluded (`-m "not
  integration"`) and run locally with `BRAINFORCE_IT=1` plus real keys; CI
  never holds provider keys.
- The `config` job is the end-to-end guard: it exercises config loading, schema
  sync, provider dry-build and the full mock pipeline (gate, provenance,
  dataset write) on every run.
- gitleaks also runs as a pre-commit hook; secrets belong in GitHub Actions
  secrets or a secrets manager, never in the repository.
- `main` is protected: all six status checks must pass before merge.

## Enabling the GPU job

1. Register a self-hosted runner on the host with the RTX 3080 and give it the
   `gpu` label (adjust `runs-on` in `gpu-smoke.yml` if your label differs).
2. Trigger `GPU smoke` from the Actions tab; it syncs the `[training]` extra
   and runs the smoke module, which skips itself cleanly when CUDA is
   unavailable.
