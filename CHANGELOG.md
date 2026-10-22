# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Graceful failure: `PipelineDef.fallback_examples` (default false) turns a
  rejected case into an abstention training record (assistant target is a
  canonical `insufficient_information` verdict plus the gate reasons) instead
  of dropping it. Metadata carries `fallback` and `fallback_reason`; dataset
  validation accepts those records without a passing quality gate. The
  rejected-case audit trail is unchanged. ADR-0016.
- `train task-eval` treats abstention as a third outcome class and reports
  `abstain_rate` and `abstain_recall` (of the records the judge could not
  resolve, how many the model refused too).
- Critic escalation: `PipelineDef.retry_with_critic` (default null) names a
  critic role that runs, followed by a second judge pass, when the first pass
  is rejected. The critic never sees the first judge verdict. Records from a
  retry carry `critic_retry` and `first_pass_reasons` in metadata.

### Changed

- README, documentation index, and first-model tutorial now state the verified
  scope, known limits, and actual quickstart output. The tutorial uses one
  mock case and separates the CUDA training step; unverified quality, speed,
  and test-count claims are removed.
- AirLLM evaluated and not adopted (ADR-0017): layer streaming helps models
  that do not fit VRAM, but its custom training API is incompatible with the
  PEFT/TRL, KD, merge and GGUF contracts, and throughput rules out local
  dataset generation.
- Provider failures are isolated per case: the case is recorded with a
  `provider_error` reason and the batch continues instead of aborting. Run
  summaries gain `fallback` and `errors` counters.
- `run_pipeline` writes the dataset and the KD sidecar once, atomically, at
  the end of the run (temp file plus rename). A crash plus rerun can no
  longer duplicate lines, and a rerun after a config or prompt tweak no
  longer keeps stale records. The dataset file always exists after a run,
  empty when nothing is accepted.
- New `docs/reference/references.md` maps each research paper and standard to
  what it informs (judge independence, contamination, QLoRA, KD, RAG,
  graceful failure). Stale links fixed: ADR-0008 pointed at the retired
  GitLab host, README advertised a `make docs-serve` target that does not
  exist.

### Fixed

- Force `pillow>=13.0.0` to clear its known CVEs despite fastembed's `<13`
  cap, and bump `certifi` to 2026.10.19.
- Audit the uv lock with `pip-audit --no-deps --disable-pip` so pip's
  resolver cannot reject the Pillow override.

## [0.3.0] - 2026-10-18

BrainForge becomes a generic distillation and fine-tuning framework: born
from a security problem, generalized to any domain. Security stays the
reference domain pack; every other domain inherits the verification
machinery (judge independence, contamination tagging, post-cutoff holdout)
by default. All changes are additive; existing security workflows keep
working unchanged.

### Added

- Quickstart command: `brainforge quickstart run --domain <pack> [--input
  <dir>] [--provider mock] [--base-model X] [--epochs 1] [--strategy
  qlora|full] [--backend trl|unsloth] [--kd]` chains the pipeline, split
  preparation, training and evaluation in one command, then prints the
  next steps (GGUF/Ollama export, chat, full fine-tuning). The train step
  requires a CUDA GPU and the training extra; the dataset steps run
  anywhere, so `--provider mock` is a zero-cost dry run of the whole chain.
  `train_cmd.train_student` is extracted as the shared backend dispatch
  used by `train run` and `quickstart`.
- Knowledge distillation (experimental): `pipeline run --kd` records the
  judge's per-token top-k logprobs into a `<dataset>.kd.jsonl` sidecar, and
  `train run --kd` trains with `alpha * T^2 * CE(student_T, teacher_topk_T)
  + (1 - alpha) * CE(gold)`. New `TrainingConfig` fields `kd_alpha` (0.5),
  `kd_temperature` (2.0), `kd_topk` (8). Providers serve logprobs through
  the OpenAI-compatible chat completions API (`logprobs=true`); the mock
  provider produces word-level positions for offline runs. In KD mode the
  dataset record stores the teacher's exact output (not a re-serialization)
  so sidecar positions align with the student's gold tokens; teacher
  positions are mapped onto student tokens by character midpoint, so
  student and teacher may use different tokenizers. Top-k tokens absent
  from the student vocabulary are dropped with the teacher distribution
  renormalized. Packing is rejected for KD training.
- Abliteration (experimental): `brainforge abliterate run --model <id|dir>
  --calibration <prompts.jsonl> --output <dir> [--layers all|0,2,...]`
  computes a single refusal direction (mean hidden-state difference between
  harmful and benign calibration prompts at a 2/3-depth layer) and projects
  every attention output and MLP down projection orthogonal to it, saving
  the modified model plus `ablation_stats.json` (direction norm before and
  after, suppression ratio). Requires the training extra (CPU or GPU).
  Quality is a heuristic, not a guarantee.
- GGUF export: `train export --format gguf` converts a merged student to a
  quantized GGUF (default Q4_K_M, `--quant`) via llama.cpp
  (`convert_hf_to_gguf.py` + `llama-quantize`), writes a `Modelfile`, and
  with `--ollama <name>` registers the model with Ollama when the binary
  is available. llama.cpp is located via `--llama-cpp-path`, the
  `LLAMA_CPP_DIR` env var, or PATH; missing tools fail fast with an
  install pointer. The produced GGUF can be distributed with llmp2p
  (`llmp2p import --ollama <output-dir>` then `llmp2p seed`).
- Unsloth training backend: `TrainingConfig.backend` (`trl` default,
  `unsloth`), `train run --backend {trl,unsloth}`, and the opt-in
  `unsloth` extra (`uv sync --extra unsloth`). The backend swaps model
  loading and the trainer wrapper only; dataset preparation, splits and
  `train_summary.json` stay identical (the summary now records the
  backend). Unsloth supports 4bit or unquantized loading; 8bit fails fast
  with a pointer to the trl backend. SFTConfig construction is shared
  between both backends.
- Full fine-tuning strategy: `TrainingConfig.strategy` (`qlora` default,
  `full`) and `train run --strategy {qlora,full}`. `full` trains every
  weight in bf16 without quantization or a LoRA adapter (needs more VRAM);
  `qlora` behavior is unchanged. `train_summary.json` records the strategy.
- `ticket_triage` domain pack: IT helpdesk triage (classification, business
  priority, team routing, first-response draft) with dedicated teacher and
  judge schemas, quality gate, and example cases. Registered alongside
  `security` and `coding`; usable through the existing pipeline with any
  provider, including local models via the `local` provider type.
- Four example domain packs demonstrating the pack surface beyond security:
  `ctf` (challenge solving: category, technique, exploit path, flag),
  `code_review` (pull-request review: findings, risk, suggested patch, gate
  forbids "approved" with high/critical findings), `security_audit` (system
  audit: findings with area/priority, gate forbids critical severity at
  p3), and `agentic` (multi-turn tool-use trajectories). Each has a teacher
  and judge schema, a quality gate, and example cases.
- Multi-turn dataset records: the `agentic` pack emits assistant tool calls
  interleaved with `tool` observations via the new
  `DomainPack.build_messages(student_input, canonical, raw_assistant)` hook
  (default remains two-turn user+assistant). Dataset validation accepts the
  `tool` role when it follows an assistant message.
- `train run --kd` rejects multi-turn records with a clear error (knowledge
  distillation aligns a single assistant turn).

### Changed

- Judge and general-critic system prompts and the general critic prompt
  template, previously duplicated across the `security` and `coding` packs,
  are now shared constants in `brainforge.domains.base`. Behavior is
  unchanged; new packs reuse them instead of copying.
- Provider structured output: the retry loop is extracted into
  `_structured_with` and reused by `structured_logprobs`; the
  OpenAI-compatible provider shares one retry helper between `complete`
  and `complete_logprobs`. Behavior unchanged.
- Docs repositioned around the generic framework: README narrative
  (origin + pivot, quickstart, two-phase workflow), full CLI reference
  (the train-as-stub drift is fixed and guarded by a test), complete
  training configuration table, new how-to (ship to Ollama) and
  explanation (knowledge distillation) pages, 5 new ADRs, `ticket_triage`
  tutorial.
- `clamp_errors` moved from `train_cmd` to `cli/_common` for reuse by the
  train and abliterate commands.
- Dependency audit: one documented waiver (`--ignore-vuln
  PYSEC-2026-4217`) for fsspec<=2026.2.0, transitive via torch/datasets
  http extra and unfixable while the unsloth extra requires datasets<5.0
  (datasets 4.8.x caps fsspec at 2026.2.0). Re-check when unsloth
  supports datasets>=5.1.

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
