# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-28

### Added

- Configurable training schedule knobs in `TrainingConfig`: `lr_scheduler_type`
  (default `cosine`, previously TRL's silent linear default), `warmup_steps`
  (5), `max_grad_norm` (1.0), `max_length` (1024), `packing`,
  `assistant_only_loss`, `attn_implementation` (`sdpa`/`flash_attention_2`/
  `eager`, default `sdpa`), `eval_batch_size` and `generate_batch_size` (4).
  All passed through to TRL `SFTConfig`; attention implementation also pinned
  on every model load (chat, task-eval, evaluate, export).
- Optional parallel case execution for dataset generation:
  `PipelineDef.concurrency` (default 1 = unchanged behavior). Results are
  collected in submission order so dataset output stays deterministic; file
  writes remain in the main thread. `UsageLogger` writes are now lock-guarded.
- `train evaluate` and `train task-eval` gained `--config` and `--batch-size`
  options (0 = config default).

### Changed

- `evaluate` (loss + perplexity) runs batched padded forward passes instead of
  one unbatched forward with an `.item()` sync per record. Perplexity is now
  the true corpus perplexity (exp of the token-weighted mean NLL) instead of
  the mean of per-sequence perplexities; existing `eval.json` values are not
  directly comparable across the change.
- `train task-eval` generates in batches (left padding) instead of one
  `model.generate` per record; `run_task_eval` now takes a batched generate
  callable (list in, list out).
- RAG vector search caches chunks in memory per `VectorStore` instance instead
  of a full sqlite table scan per query (invalidated on `add`/`clear`);
  `Retriever.index` persists all vectors in one write instead of rewriting
  `vectors.npy` per batch; `HashingBackend` memoizes token digests.

### Fixed

- `train run --resume` picked the latest checkpoint with a lexicographic sort,
  silently resuming from an older checkpoint once step numbers cross 999
  (e.g. `checkpoint-999` over `checkpoint-1000`); checkpoints are now sorted
  by numeric step.

## [0.1.0] - 2026-09-21

### Added

- Task-level evaluation harness (`train task-eval`): verdict accuracy, FP/FN
  rates and CWE classification accuracy on the post-cutoff holdout, scored
  against the judge-verified answer stored in each record (`task_eval.json`).
- Interactive chat with a trained student: `train chat` (adapter or merged
  model, greedy decoding via the model chat template).
- Training checkpointing: `save_strategy="steps"` with `TrainingConfig.save_steps`
  (default 100, keep last 2), seeded runs (`TrainingConfig.seed`, default 42)
  and `train run --resume` (TRL native resume from the latest checkpoint).
- Real QLoRA training pipeline: `train_qlora` (TRL `SFTTrainer`, bitsandbytes
  NF4/8bit quantization, gradient checkpointing, `TrainingConfig`-driven),
  `evaluate` (loss + perplexity on the post-cutoff split, `eval.json`) and
  `export` (LoRA merge). CLI: `train run` with `--dataset-dir`,
  `--epochs`, `--base-model`, `--output`; per-run output directories under
  `experiments/`.
- `train evaluate`/`train export`/`train task-eval`/`train chat` now default
  `--model` to the latest run directory under `experiments/`.
- CPU smoke training job in CI: 1 LoRA step on a tiny model, no GPU or
  bitsandbytes needed (`make train-smoke-cpu`, `smoke-cpu` CI job). The smoke
  test skips only when torch is missing; missing-training-deps or an
  unreachable HF hub now fail the job instead of silently passing.
- GPU CI on all three forges: GitHub `GPU` workflow (manual smoke + full
  train -> evaluate with 14-day artifacts), GitLab manual `train` job
  (`tags: [gpu]`), Forgejo `gpu-*.yml` dispatch workflows (untested runner).
- `GPU train via SSH` workflow: runs the training chain on the GPU host over
  SSH from a standard runner (secrets `SSH_HOST`, `SSH_USER`, `SSH_KEY`).
- GPU host runbook: docs/how-to/register-gpu-runner.md (Windows runner
  registration, OpenSSH Server setup, first training run).
- Portable CI: the same six gates run on GitHub Actions (reference), GitLab
  CI (`.gitlab-ci.yml`, pinned `uv==0.12.7` and gitleaks image `v8.30.1`),
  Forgejo (`.forgejo/workflows/ci.yml`, act_runner, untested against a live
  runner), and locally via `make ci` (new `secrets`, `build`, `audit` Make
  targets).
- Dependency CVE audit job (`pip-audit --strict` over exported runtime
  requirements) in the GitHub Actions CI.
- Dependabot: weekly updates for pip and GitHub Actions dependencies.
- Coverage artifact: `coverage.xml` uploaded by the `test` job (7-day
  retention).
- Branch protection on `main`: all CI status checks required before merge.
- GitHub Actions CI mirroring the GitLab pipeline: ruff lint, gitleaks secret
  scan over full history, unit tests with coverage, config validation plus
  mock pipeline run, and package build published as a 7-day artifact.
- Manual `GPU smoke` workflow (`workflow_dispatch`) targeting a self-hosted
  runner with the `gpu` label for the training smoke test.
- Documentation restructured following Diataxis: guided tutorial, six how-to
  guides, eight reference pages (including new quality-gates and
  dataset-format references), and three explanation pages (architecture,
  contamination, judge correlation).
- Architecture Decision Records in `docs/adr/` (MADR 3.0): index, blank
  template, and ten decision records covering configuration, ensemble and
  judge independence, contamination controls, provider abstraction, RAG,
  domain packs, dataset format, tooling, cache and training targets.
- Coverage badge: GitLab coverage regex on the `test` job.
- Subprocess CLI-chain integration test (`tests/integration/`, opt-in via
  `BRAINFORCE_IT=1`): pipeline run (mock) -> dataset validate -> train prepare
  on the real filesystem, no API keys or GPU.
- Strict mypy typing gate (pydantic plugin, `disallow_untyped_defs`): zero
  errors on `src/`, enforced by a `mypy` job on all three CI systems.
- Blocking coverage gate (`fail_under = 85`, `show_missing`) on all three CI
  systems' test jobs.
- Mutation testing via mutmut (v3): local `make mutate` target plus
  CONTRIBUTING rule to triage surviving mutants after touching core logic.
- ~40 new unit tests covering previously indirect paths: pipeline judge
  contracts, training `prepare`, case builder, dataset validation, mock
  provider branches, CLI models/teacher paths, split fallbacks and defaults,
  engine usage logging and RAG metadata.

### Changed

- `evaluate` now honors the configured quantization (`--quantization
  4bit|8bit|none`) instead of always loading in 4-bit.
- `train prepare` writes the default output to `datasets/prepared/`, the
  directory `train run`/`evaluate`/`task-eval` read by default; the `--output`
  override is unchanged.
- `train run --resume` resumes the latest existing run directory instead of
  creating a new one.
- `latest_run_dir` resolves a run by modification time and honors the
  configured `training.output_dir`.
- CI workflows now install the toolchain through mise (`jdx/mise-action@v4`
  reading `mise.toml`) instead of pinning uv separately.
- CI moved from GitLab to GitHub: badges and clone URL in the README now point
  at `log0u7/brainforge`, CONTRIBUTING reworded for pull requests, and
  `docs/reference/ci.md` documents the GitHub Actions workflows.
- README rewritten: badges, 30-second quickstart with expected output,
  feature table, principles, provider overview and Diataxis docs routing.
- mkdocs navigation reorganized into Tutorials / How-to guides / Reference /
  Explanation / Decision records; Pages job builds with `--strict`.
- Provider wrapper subclasses (OpenRouter/Zen/MLGW/local) replaced by a
  `_DEFAULT_BASE_URLS` mapping in the provider registry: every non-mock type
  resolves to `OpenAICompatProvider`, base URL from config or registry
  default.
- Shared `load_student_model()` (new `training/models.py`): chat and
  task-eval now use one loader for merged models and LoRA adapters.
- `PipelineContext` dataclass inlined into `pipeline/engine.py`.

### Fixed

- `train task-eval` (CLI path): real generation now flattens the message list
  into a single content string, matching the typed `generate(str | list[str])`
  contract; `prediction_from_text` no longer swallows unexpected errors as
  "miss".
- `train run`/`evaluate`/`task-eval`/`chat`/`export` fail with a clear message
  instead of a raw traceback when an input file or model path is missing
  (OSError clamped to a clean exit).
- `train run --resume` on a run without checkpoints fails fast with a clear
  message instead of a raw HF Trainer `ValueError`.
- Removed dead code in `train chat` (misleading `name_or_path` argument,
  redundant `generate_reply_fn`) and the pointless double dataset read in
  `dataset split`.
- `case.id` is validated against `[A-Za-z0-9_-]{1,64}`, preventing path
  traversal via crafted case files.
- Post-cutoff records no longer leak into the `train` split: they are now
  exclusive to `test_postcutoff` (the anti-contamination benchmark).
- Response cache is now consulted by `structured()` calls; previously every
  pipeline structured call bypassed the sqlite cache.

### Security

- `ProviderConfig.base_url` now validates scheme (http/https) and rejects
  link-local and cloud-metadata hosts (SSRF guard); loopback stays allowed for
  local gateways.
- Removed the committed `MLGW_API_KEY:deadbeef` placeholder default from the
  sample config and docs (empty default like the other providers; the provider
  fails at call time when the key is unset).

### Removed

- mkdocs site (`mkdocs.yml`) and `mkdocs-material` dev dependency; the
  `docs/` tree stays as plain markdown rendered by GitHub.
- GitLab remote: the repository lives on GitHub only.
- Legacy flat doc pages (`docs/architecture.md`, `docs/configuration.md`,
  `docs/providers.md`, `docs/domain-packs.md`, `docs/rag.md`,
  `docs/dataset.md`, `docs/training.md`, `docs/cli.md`, `docs/ci.md`)
  migrated into the new structure.
- `sqlite-vec` declared extra (never imported).
- Provider wrapper modules (`providers/openrouter.py`, `zen.py`, `mlgw.py`,
  `local.py`) in favor of the registry mapping.
- `pipeline/context.py` (inlined into the engine).
- Dead code: `QualityGateError` (never raised), `SplitName` enum (never
  imported), `list_models()` provider method (no caller),
  `build_roles_without_cache()` helper (no caller), `RoleRegistry.kinds()`
  (test-only), unused `DEFAULT_REJECTED_DIR`/`DEFAULT_EXAMPLES_DIR`
  constants, `golden/` placeholder directory.

## [0.0.0] - 2026-08-31

Initial MVP release.

### Added

- JSON configuration system with pydantic models, `${VAR:default}` environment
  expansion, JSON Schema export/check, and enforced judge independence
  (judge provider and model family must differ from every teacher).
- Provider layer: generic OpenAI-compatible provider with retries, timeouts and
  structured-output parsing; OpenRouter, OpenCode Zen (`chat_completions`
  style), MLGW, local (Ollama-compatible) adapters; deterministic mock provider
  with generic JSON-Schema instance generation; sqlite response cache keyed on
  provider/model/role/messages/params/RAG context.
- Observability: per-call JSONL usage logs (tokens, latency, retries, cache
  hits, estimated cost from optional pricing config).
- Model and role registries with kind inference (teacher/critic/judge).
- Domain packs: `security` (CWE/severity/evidence schemas, security teacher,
  coding teacher, adversarial judge, quality gates) and `coding` (code analysis
  schemas); generic fallback pack for future domains.
- Pipeline engine: role steps, pipeline context, cost modes (cheap/standard/
  maximum), optional teacher ablation via explicit steps, unresolved-verdict
  rejection, `recitation_risk` tagging from case source dates vs teacher
  knowledge cutoffs.
- Local RAG: text/markdown/code/JSON loaders, markdown-aware chunker, fastembed
  (ONNX, CPU) embeddings with deterministic hashing fallback backend, sqlite +
  numpy vector store with full retrieval provenance.
- Dataset tooling: case builder (JSON cases and plain code files, git commit
  dates), JSONL writer with teacher/judge/RAG provenance, record validation,
  quality gate, exact + near-duplicate detection (Jaccard shingles),
  source-grouped train/validation/test splitting plus post-cutoff holdout.
- Training preparation: validate/split/export for TRL; QLoRA, evaluate and
  export stubs (phase 2); GPU smoke module for the manual CI job.
- Experiments: dated run directories with config snapshot, dataset copy,
  metrics (contamination stats, teacher/judge agreement rates) and a generated
  markdown report including correlated-bias warnings.
- CLI (`brainforge`): `config`, `models`, `roles`, `teacher`, `pipeline`,
  `rag`, `dataset`, `train` command groups with rich output.
- Documentation: README, CONTRIBUTING, ROADMAP, mkdocs-material site.
- CI: GitLab pipeline with lint (ruff), secrets (gitleaks), tests, config
  validation, build, mkdocs Pages, and a manual GPU-tagged training smoke job.
- Example cases spanning a teacher knowledge cutoff (command injection,
  SQL injection, stored XSS) and a working example configuration.
