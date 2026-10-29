# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `brainforge dataset enrich-flags <file.jsonl>`: stamp verified-tier
  `expected_flag` strings (the real flag the gate checks against) into
  record answers and tag `flag_verified` in metadata. The pipeline
  verified flags but stored `flag: null`, so flag_accuracy was
  uncomputable on every dataset (2026-10-29 measurement); with the
  strings in place the holdout ground truth and the training targets
  both carry the flag.

### Changed

- README badges: added Hugging Face and OpenAI, the two missing teacher/export surfaces.
- SFT training now uses TRL's conversational prompt-completion format with
  `completion_only_loss=True`: the loss lands on the final assistant message
  only. The previous messages format with `assistant_only_loss=False` scored
  the whole sequence, so on long CTF prompts (~90% of tokens) the model
  learned to predict challenges instead of answers - measured on the
  26-record holdout as perplexity -37% with generation behavior unchanged
  from the base model. `training.assistant_only_loss` is removed (the
  Qwen3 template has no `{% generation %}` markers, so the flag never
  produced a mask); the agentic multi-turn warning is reworded for
  prompt-completion semantics.

### Fixed

- SFT training fails fast when `max_length` would truncate a completion
  (TRL cuts the sequence tail, so the model never sees the answer
  terminator and never learns the answer format; the default 1024 cut
  every 1145-2168 token KD-GLM completion).
- `train task-eval` thinking default is now `--no-thinking`: the training
  render carries the empty think block into the prompt (per-row
  `chat_template_kwargs: {enable_thinking: false}`), so the inference
  prompt matches it byte-for-byte; `--thinking` stays for models whose
  own think traces eat the token budget.
- `train task-eval` warns when the token budget cannot fit the longest
  ground-truth completion instead of scoring answers that are cut mid-JSON
  by construction, and `--max-new-tokens` defaults to 2048 (was 512).
- CTF archive cases run the same bounded static recon (file type, ELF
  header, protections, symbol hits, disasm head) on ELF attachments as
  the lictor/rop tier-A cases: the archive corpus used to emit only
  `binary/image, not excerpted` placeholders, a prompt distribution the
  trained student never saw at eval time.
- Benchmark doc: third run results - the union SFT run (16 records,
  holdout ids excluded) scores the first non-zero task-eval accuracy
  (0.1923, FPR 0.7308) while perplexity regresses (19.77 vs baseline
  12.61); the table now carries both metrics side by side with the
  explanation (likelihood punishes the verbose style the task scorer
  rewards).

## [0.8.0] - 2026-10-28

### Added

- `brainforge train task-eval --endpoint URL --model-name M`: score a model
  served behind an OpenAI-compatible endpoint (llama-server) without loading
  weights locally; `build_remote_generator` sends one chat completion per
  record prompt through the provider abstraction.

- `brainforge train evaluate --base`: evaluate the untouched base model (no
  adapter) with the same token-weighted scoring as adapter evaluation, so
  before/after perplexity numbers are directly comparable.
- `brainforge ops status`: summarize `*.state.json` run states (done/total,
  current, freshness) plus optional log tails, replacing ad-hoc host check
  scripts.
- `brainforge ops log <path>`: tail the last N non-empty lines of a log with
  an encoding fallback chain (utf-8, utf-16-le, cp1252) and error-line
  highlighting.
- `brainforge dataset split --sidecar <kd.jsonl>`: optionally align a KD
  sidecar to the train split (entries filtered to train ids, written to
  `<output>/kd.jsonl` or `--sidecar-out`). Split outputs are unchanged when
  the option is absent.
- `brainforge pipeline kd-pass`: run `pipeline run ctf --kd --no-rag` per
  case file (resume-safe per-case scratch files, per-case timeout, optional
  `--only-solved`) and assemble a sidecar-aligned KD dataset directory
  (`train.jsonl` + `kd.jsonl`). Replaces the ad-hoc kd_full.py/kd_pass.py
  host scripts.
- `brainforge config set <dotted.path> <value>` and
  `brainforge config set-model <role> <provider> <model>`: edit the config
  file from the CLI (JSON-scalar parsing, nested creation, sibling
  preservation), replacing the ad-hoc fix_*.py host scripts.

- `pipeline run` now self-traces progress to a structured state file next to
  the dataset (`<output>.state.json`, overridable via the new `run_pipeline`
  `state_path` parameter): written atomically (temp file + rename) before and
  after every case, under a lock so parallel mode stays consistent. It records
  `started_at`/`updated_at`, `status` (`running`/`done`), `total_cases`,
  `done`, `accepted`, `rejected`, `errors`, `current_case`, and one entry per
  case (`id`, `status` `running`/`accepted`/`rejected`/`error`, `duration_ms`,
  `gate_reasons` count), so multi-hour runs (and duplicate-instance races) are
  observable while they happen. New `brainforge pipeline status` command reads
  it and prints counts plus a per-case table; a missing or corrupt state file
  exits 1 with a clean error.
- GRPO training backend: `strategy: "grpo"` (`train run --strategy grpo`)
  trains with TRL `GRPOTrainer` against verifiable rewards instead of a
  labeled corpus. Rewards are named callables in
  `brainforge.training.rewards`: `register_reward`/`get_reward` manage the
  registry, `compose_reward("ctf_flag,compliance*0.5")` builds a weighted
  composition so one pass trains against several signals, and the built-ins
  are `ctf_flag` (exact flag match 1.0 on the verified tier, format-only
  0.3, absent 0) and `compliance` (refusal-tic scoring for GRPO-oblit). An
  optional `grpo_flags.jsonl` sidecar (id -> expected_flag) feeds the
  verification signal. Resolution order in `train_grpo`: explicit reward
  spec > `training.grpo_reward` > the `DomainPack.grpo_reward()` hook >
  `BrainforgeError` (no silent default: an unusable reward fails before
  training starts). A new domain joins RL training by declaring a reward;
  `CtfPack` declares `ctf_flag`.
- Teacher/judge dispute arbitration by ground truth: on verified-tier cases
  (an `expected_flag` is known), a judge `insufficient_information` verdict
  no longer discards the case. If the correct flag appears in the teacher
  outputs or in the judge's own reasoning, the case is accepted with the
  `arbitrated_by_ground_truth` reason and the winning answer feeds
  training. When nobody's flag matches, the strict abstention path applies;
  tier-B cases keep the abstention gate unchanged.

### Fixed

- Provider calls now enforce a wall-clock total deadline (configured
  `timeout` plus a 30s grace). httpx timeouts bound the time between reads,
  so a drip-feeding stream never tripped them and could freeze a pipeline
  run forever (observed twice on gpu3080: zero-CPU hangs mid-batch, no
  exception, no rejection). Each attempt now runs in a worker thread; on
  expiry the attempt is abandoned without waiting on the orphaned thread
  (a `shutdown(wait=True)` would have reintroduced the hang) and the retry
  loop proceeds, ending in `ProviderUnavailableError` for the case as
  designed.
- `CachingProvider` now delegates `complete_logprobs` to the inner
  provider: it only overrode `complete()`, so every logprobs call fell
  through and raised "does not support logprobs", making the KD pass
  unreachable through the cache wrapper (judge sidecar empty on gpu3080).
  Logprobs responses stay uncached (live teacher signal).
- KD training runs on Windows: datasets materialize eagerly with
  `Dataset.from_list` instead of `Dataset.from_generator`, whose unstable
  closure fingerprinting collided the train and eval builders onto one
  cache directory and self-locked (WinError 32, reproduced with default
  and fresh `HF_DATASETS_CACHE`); the `kd_targets` column travels through
  the arrow table as a JSON string and `KDDataCollator` decodes it; and
  the KD path sets `TrainingArguments.remove_unused_columns=False` so the
  Trainer no longer strips `kd_targets` before the collator
  (`KeyError: 'kd_targets'` on the first real KD run).
- `evaluate()` pads batches with `return_tensors="pt"`.

- Regenerated `config/schema.json`: the GRPO strategy/reward fields from
  #49 landed in `src/brainforge/config/models.py` without regenerating the
  schema, leaving `config schema --check` (and CI) red on `main`.
- Auto-fit now estimates against the training plan that will actually run
  (`resolve_plan`) instead of a hardcoded QLoRA/4bit plan: on CPU the real
  plan is fp32/none (4 bytes/param), so the previous verdict was ~7x
  optimistic and real training runs OOM-killed whole desktop sessions
  (systemd-oomd killed the GNOME session on 2026-10-27). The device budget
  is free VRAM on CUDA/ROCm/XPU and available system RAM minus a safety
  margin (`max(2 GiB, 15%)`) on CPU/MPS (psutil, cross-platform). A model
  whose weights+adapters+optimizer floor does not fit the budget is now
  refused with an actionable error before `from_pretrained` runs. The
  dead `cpu_offload` flag was removed: it was never consumed and CPU
  offload was never implemented. KD training now runs the same auto-fit
  wiring. Model metadata or hardware detection failures degrade to an
  "auto-fit skipped" warning instead of blocking training or hitting the
  Hub on every run. The preflight disk-space check now targets the
  training output directory instead of the current working directory,
  and skips (instead of crashing) when that directory does not exist
  yet. `train_kd` refuses an empty train split with a clear
  BrainforgeError before any heavy import. The generated KD arrow
  cache is keyed on a content digest of the split files, so a
  regenerated dataset invalidates the stale cached table instead of
  silently training on it. Three unit tests also silently depended on
  the environment: they passed only when the training extra was absent
  and broke with real torch installed (real hardware detection, Hub
  access through auto-fit, cached real transformers mappings). They
  now hide torch via sys.modules or stub the causal-LM mapping, so the
  suite is green with and without the training extra.
- On-policy self-training adoption: ADR-0019 records the decision to
  adopt OPST (data-free self-improvement, arXiv 2610.32770) as an
  experimental `train` strategy `opst`, gated on the vectorized KD loss
  (issue #57, closed) and a validated VRAM plan for the 10GB RTX 3080.
  The ADR fixes the contract shift versus the ADR-0015 KD backend
  (privileged trace z as teacher context, on-policy ordering, asymmetric
  JSD target, EMA teacher, round driver) and the minimal smoke that
  serves as the acceptance test.

### Changed

- KD training vectorizes the distillation loss (issue #57): the
  `KDDataCollator` now encodes the per-row teacher targets once into padded
  tensors (position indices, in-vocabulary top-k ids, logprobs, masks) and
  `KDTrainer.compute_loss` consumes them through the new
  `compute_kd_loss_batch`, which runs one chunked log-softmax/gather kernel
  sequence instead of a Python loop with a full-vocab log-softmax per
  position. The math is unchanged (same renormalized-teacher top-k
  cross-entropy, same None-when-no-valid-token semantics, same T^2
  scaling; equivalence pinned bitwise on fp32 in
  `tests/unit/test_kd_vectorized.py`); batches without the collator
  encoding still fall back to the per-position loop.

- KD training reads records through the new
  `brainforge.dataset.writer.iter_jsonl` reader instead of materializing
  every tokenized record in RAM at once, and builds the arrow datasets
  eagerly (`Dataset.from_list`): the `Dataset.from_generator` builder
  cache is not Windows-safe (see Fixed above) and KD datasets are tiny by
  construction. Behavior preserved: teacher-position validation still
  fails on the first record without a sidecar entry, with the same error
  message; an empty or missing validation split still disables
  evaluation. A missing train.jsonl now raises a clean BrainforgeError
  instead of a raw FileNotFoundError.

### Removed

- The unused `total_steps` parameter of `_auto_fit` (the warmup warning
  is computed post-fit in `auto_fit_config`).

### Added

- Static AST guard `test_source_lazy_imports.py`: fails on any
  module-level import of torch/transformers/trl/peft/bitsandbytes/
  unsloth in `src/brainforge`, enforcing the AGENTS.md lazy-import rule
  regardless of the installed environment. The scan recurses into
  module-level `if`/`try`/`with`/loop bodies (a `try: import torch`
  at module level is still a module-level import) and exempts only
  `TYPE_CHECKING` blocks and imports inside function/class bodies,
  with self-tests proving both directions.
- Auto-fit for training: `_auto_fit()` in preflight computes a device
  memory budget (CUDA/ROCm/XPU/MPS/CPU agnostic) and automatically
  reduces `max_length`/`batch_size` when activations do not fit, with
  explicit adjustment logging. User-set config values are respected
  (not silently overridden) with a "may not fit" warning. Warmup-steps
  >= total-steps warning added.

### Added

- First benchmark run published: 23 CTF cases generated (12 verified
  tier with curated technique ground truth, 15 accepted), QLoRA on the
  RTX 3080 and task-eval on the post-cutoff holdout. Numbers: base and
  student both 0.00 accuracy at the 6-record scale (prose output, no
  CtfVerdict JSON) - the honest reading and the scale-up lever are in
  `docs/explanation/benchmark.md`.

- CTF flag gate for verified-tier cases: `quality_gate` now receives the
  case (all packs accept the new optional parameter) and the CTF pack
  rejects a judge flag that differs from `metadata.expected_flag`
  whatever the verdict, plus a matching flag under a non-solved verdict;
  cases without `expected_flag` are unaffected.
- First CTF case corpus: `experiments/ctf_ingest/ingest.py` builds 23
  cases from the local CTF corpus (6 lictor lab targets and 6 ROP
  Emporium binaries with known flags; 11 CTF-Archives 2026 Zeekr
  challenges with flag format only). Writeups stay verification
  references and never enter the case inputs.
- Post-training expansion deferral: ADR-0018 records the decision to defer
  the TRL v1.x migration, an on-policy distillation backend (GKD), GRPO on
  the CTF pack, an OpenEnv environment and multi-harness RL until the first
  published benchmark; ROADMAP phase 3 carries each deferred item with its
  prerequisite and `docs/reference/references.md` indexes the deck, blog
  posts and papers behind the map.
- Agentic loss hygiene: `brainforge train run` warns when the training
  split holds `tool` observations or multi-assistant records while
  `training.assistant_only_loss` is off (those tokens would be scored into
  the loss); `docs/reference/training.md` documents when to enable the
  flag and its chat-template requirement.
- CTF task evaluation: `train task-eval` scores the CTF domain
  (verbatim verdict classes solved/partial/abstain/unparseable plus
  `flag_accuracy` against the gate-checked judge flag) alongside the
  security metrics; mixed-domain evaluations are rejected explicitly.
- The first-benchmark specification: `docs/explanation/benchmark.md`
  defines the base-vs-student run on the post-cutoff holdout, its
  metrics, honesty rules (n, single seed, observational) and the
  GPU-host runbook; `train task-eval` accepts a Hub model id for the
  base-model side (report persisted only next to local weights).
### Fixed

- Three defects found by the first real GPU eval run: a 0-record
  validation split (tiny datasets) no longer aborts training (it behaves
  like a missing file, and the empty-train guard stays); the task-eval
  batch builder no longer collapses a multi-prompt batch into ONE chat
  sequence (each prompt gets its own conversation, so predictions match
  records one-to-one); and `train task-eval` now accepts a Hub model id
  for the base-model side (snapshot download fallback in
  `load_student_model`).
## [0.4.0] - 2026-10-24

### Added

- Portable training: `brainforge train check` prefights hardware, model
  compatibility (architecture, chat template, pre-quantized and remote-code
  rejection, gated-model `HF_TOKEN` guidance) and a memory estimate without
  downloading model weights; `--json` and `--strict` modes for CI. Device
  auto-detection covers CUDA/ROCm, Intel XPU, Apple MPS and CPU; TPU/NPU
  fail fast with a `--device cpu` escape. `training.device`,
  `training.quantization` and `training.backend` (new `auto` defaults)
  resolve a per-run plan recorded in `train_summary.json` and `eval.json`.
  `make train-check`, `make install-rocm` and `make install-xpu` targets.
  Docs: new `docs/reference/hardware.md`; README, tutorial, training,
  configuration, CLI and architecture pages updated. CUDA/ROCm/XPU paths are
  implemented and unit-tested but not hardware-validated.
- `huggingface-hub>=0.26` is now a direct dependency (preflight model
  metadata checks; previously available only transitively).
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

- Training and evaluation no longer hard-require CUDA: the device plan
  resolves automatically and `qlora` degrades to plain LoRA where 4-bit is
  unavailable. `train evaluate` and `train task-eval` default to
  `--quantization auto`; `train run` accepts `--device` and
  `--quantization` overrides.
- Rewrite the README for evaluators: dataset-first quickstart, verified
  mechanics, explicit limits, and intent-based docs routing. The docs hub and
  first-model tutorial use the same commands, vocabulary, and qualifiers.
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
- Bump `filelock` to 4.2.4 in the lock, clearing CVE-2026-112907,
  CVE-2026-112913 and CVE-2026-112940 (fixed upstream in 4.2.1).
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
