# CI reference

GitLab CI (`.gitlab-ci.yml`). Pipelines run on merge requests and on `main`
(`CI_DEFAULT_BRANCH`).

## Jobs

| Stage | Job | Content | Runs |
|---|---|---|---|
| lint | `lint` | `uv sync`, `ruff check .`, `ruff format --check .` | every pipeline |
| lint | `secrets` | gitleaks over full history (`GIT_DEPTH: 0`), `--redact` | every pipeline |
| test | `test` | `pytest -m "not integration"` with coverage (GitLab coverage regex for the badge) | every pipeline |
| test | `config` | `brainforge config validate`, `config schema --check`, mock pipeline over `examples/cases` | every pipeline |
| build | `build` | `uv build`, sdist+wheel artifacts (1 week) | every pipeline |
| pages | `pages` | `mkdocs build --strict --site-dir public` | main only |
| gpu | `train-smoke` | 1-step QLoRA smoke (`brainforge.training.smoke`) | manual, `tags: [gpu]`, `allow_failure: true` |

## Conventions

- All Python jobs use the official `ghcr.io/astral-sh/uv:python3.12-bookworm`
  image with `uv.lock`-keyed cache (`.uv-cache`).
- `interruptible: true` on everything: new pushes cancel superseded runs.
- Integration tests hitting real providers are excluded (`-m "not
  integration"`) and run locally with `BRAINFORCE_IT=1` plus real keys; CI
  never holds provider keys.
- The `config` job is the end-to-end guard: it exercises config loading, schema
  sync, provider dry-build and the full mock pipeline (gate, provenance,
  dataset write) on every push.
- gitleaks also runs as a pre-commit hook; secrets belong in GitLab CI/CD
  variables or a secrets manager, never in the repository.

## Enabling the GPU job

1. Register a runner with the `gpu` tag on the host with the RTX 3080.
2. `gitlab-runner exec` or trigger `train-smoke` manually from the pipeline
   view; it syncs the `[training]` extra and runs the smoke module, which
   skips itself cleanly when CUDA is unavailable.
