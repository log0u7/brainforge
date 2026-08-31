# CI

GitLab CI (`.gitlab-ci.yml`). Default rules run the pipeline on merge requests
and on `main`.

## Stages

| Stage | Job | Content |
|---|---|---|
| lint | `lint` | `uv sync`, `ruff check .`, `ruff format --check .` |
| lint | `secrets` | gitleaks over full history (`GIT_DEPTH: 0`), fails on any leak |
| test | `test` | `pytest -m "not integration"` with coverage |
| test | `config` | `brainforge config validate` + `config schema --check` + example mock pipeline run |
| build | `build` | `uv build`, sdist+wheel artifacts |
| pages | `pages` | mkdocs-material site published to GitLab Pages (main only) |
| gpu | `train-smoke` | manual, `tags: [gpu]`: 1-step QLoRA smoke on a self-hosted runner |

## Notes

- All jobs use the official `ghcr.io/astral-sh/uv:python3.12-bookworm` image
  and cache `.uv-cache` between runs.
- Integration tests hitting real APIs are excluded (`-m "not integration"`);
  they run locally with `BRAINFORCE_IT=1` and real keys.
- The `train-smoke` job is `when: manual` and `allow_failure: true`; it skips
  itself when the runner has no CUDA device. Register a runner with the `gpu`
  tag to enable it.
- gitleaks runs pre-commit too; secrets belong in CI/CD variables or a secrets
  manager, never in the repository.
