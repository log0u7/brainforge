# CI reference

The same six gates (lint, secrets, test, config, build, audit) run on every
platform, each in its native syntax:

| Platform | File | Status |
|---|---|---|
| GitHub Actions | `.github/workflows/ci.yml` | reference implementation; branch protection on `main` requires all six checks |
| GitLab CI | `.gitlab-ci.yml` | parity port (pinned `uv==0.12.7` via pip, gitleaks image `v8.30.1`) |
| Forgejo (act_runner) | `.forgejo/workflows/ci.yml` | written, not yet tested against a live runner (5 jobs: no build artifact, no GPU) |
| Local | `make ci` | exact equivalent of the pipeline, no CI server needed (needs `mise`/`uv` and the `gitleaks` binary on PATH) |

GitHub Actions also has `gpu.yml` (manual: training smoke + QLoRA
train -> evaluate on the self-hosted GPU runner) and `gpu-ssh.yml` (manual:
same chain over SSH to the GPU host from a standard runner). GitLab has manual
`train-smoke`/`train` jobs with `tags: [gpu]`; Forgejo has dedicated
`gpu-*.yml` dispatch workflows. All GPU jobs only run where a self-hosted
runner with the `gpu` label is registered
(see docs/how-to/register-gpu-runner.md).

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

## GPU jobs

The GPU stage is the core of the tool: it trains the student. Two entry
points on GitHub Actions (both `workflow_dispatch`, `runs-on: [self-hosted, gpu]`):

- `GPU` workflow, input `job`:
  - `smoke`: one QLoRA step on `Qwen/Qwen3-0.6B` (fast CUDA check,
    `continue-on-error`).
  - `train`: `train prepare` (optional input dataset) -> `train run`
    (`--epochs`/`--base-model` overrides) -> `train evaluate`
    (loss + perplexity on the post-cutoff split). Artifacts: adapter +
    `train_summary.json` + `eval.json` (14 days).
- `GPU train via SSH`: same chain, executed on the GPU host over SSH from a
  standard runner (secrets `SSH_HOST`, `SSH_USER`, `SSH_KEY`).

Training code: `brainforge.training.qlora` (TRL `SFTTrainer`, bitsandbytes
NF4 quantization, gradient checkpointing). The GPU host setup is documented in
docs/how-to/register-gpu-runner.md.
