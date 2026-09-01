# CI reference

GitHub Actions. `ci.yml` runs on pushes to `main` and on pull requests
targeting `main`; superseded runs on the same ref are cancelled
(`concurrency`). `gpu-smoke.yml` runs on `workflow_dispatch` only.

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

### `.github/workflows/gpu-smoke.yml` (manual)

| Job | Content | Runs |
|---|---|---|
| `train-smoke` | 1-step QLoRA smoke (`brainforge.training.smoke`) | `workflow_dispatch`, self-hosted `[self-hosted, gpu]`, `continue-on-error: true` |

## Conventions

- Toolchain managed by mise: jobs run `jdx/mise-action@v4`, which installs uv
  `0.12.7` from `mise.toml` (single source of truth). Python projects sync
  with `uv sync` against the `uv.lock`-keyed Actions cache.
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
