# Roadmap

Phases are cumulative. The MVP (phase 1) is implemented in this repository.

## Phase 1 - MVP dataset generation (this release)

- [x] JSON config + JSON Schema, environment expansion, judge independence
- [x] Providers: OpenRouter, OpenCode Zen (chat_completions), MLGW, local stub,
      mock with generic structured-output generation
- [x] Model registry, roles, pipeline engine, cost modes
- [x] Domain packs: security, coding, generic fallback
- [x] Local RAG with provenance
- [x] Quality gate, deduplication, dataset JSONL + validation
- [x] Contamination controls: `knowledge_cutoff`, `recitation_risk` tagging,
      post-cutoff holdout split
- [x] Cache, usage logs, CLI, Makefile, tests, CI, docs

## Phase 2 - Train the first students

- [x] QLoRA training on RTX 3080 10GB (4-bit, LoRA rank from config, gradient
      checkpointing) via TRL; student base model configurable
      (default `Qwen/Qwen3-8B`); `train run/evaluate/task-eval/export`
- [x] Full fine-tuning strategy (`--strategy full`) and Unsloth backend
      (opt-in extra) for faster iteration
- [x] Optional knowledge distillation (`pipeline run --kd` sidecar +
      `train run --kd` loss), marked experimental
- [x] GGUF export + `ollama create` registration (`train export --format gguf`)
- [x] Quickstart command chaining the whole loop
- [x] Experimental abliteration command
- [x] `ticket_triage` example pack proving the generic framework
- [ ] First real dataset run: 20-50 cases (CVE + patch + vulnerable + fixed
      code), manual inspection of every accepted record
- [ ] Scale to 500-2,000 examples once quality is validated
- [ ] Golden dataset: 50-200 human-verified cases, including post-cutoff ones
- [x] Benchmark base model vs fine-tuned student, primarily on
      `test_postcutoff.jsonl` to prove analysis rather than recitation
      (first run 2026-10-26: pipeline demonstrated end to end, no gain
      at the 6-record scale - numbers in `docs/explanation/benchmark.md`;
      the 500-2000 record scale-up is the next lever)

## Phase 3 - Measure, ablate, extend

- [ ] Framework decoupling per [ADR-0022](docs/adr/0022-wake-sleep-framework-skeleton.md):
      break the CTF couplings inside generic modules (`task_eval`, `grpo`,
      `rewards`, `consolidation`, `benchmark`, `pipeline/engine`,
      `flag_ground_truth`, `dataset_cmd`) into pack-level hooks
      (`eval_score`, `arbitrate`, `rewards`, `gate_metrics`); relocate the
      CTF-only experiments (`pilot`, `model_only_pilot`, `bench_lab`,
      `final_bench`, `ctf_ingest`) to the CTF pack surface. The loop
      definition becomes domain-free; new domains plug in as one pack.
- [ ] Multi-teacher experiments (A: security+judge, B: +coding, C: +general)
      comparing accuracy, FP/FN, confidence, cost, latency
- [ ] Teacher ablation reporting in the experiments manager
- [ ] Adaptive pipeline: task classifier selecting teachers per case
      (Python -> coding+security, infra -> security, unknown -> general)
- [ ] Local provider backends: llama.cpp, Ollama, vLLM (behind the existing
      `local` provider), student-assisted dataset generation
- [ ] OpenCode Zen `responses` and `anthropic` endpoint styles
- [ ] sqlite-vec index backend as the default at scale
- [ ] Dataset augmentation (analysis -> remediation -> patch analysis ->
      explanation) with strict anti-duplication limits
- [ ] KD quality benchmark: compare `train task-eval` numbers KD vs non-KD
      on identical splits before promoting the feature from experimental
- [ ] Tighten the TRL pin (the lock already resolves 1.12.0 on the v1 API)
      to the version carrying the on-policy trainers and validate SFT/KD
      against it; prerequisite for the distillation and RL items below
      (ADR-0018)
- [x] On-policy distillation backend: the `gkd` strategy (TRL
      `experimental.gkd.GKDTrainer`, generalized-JSD loss against a local
      shared-tokenizer teacher) plus the periodic consolidation cycle with
      replay and a per-round holdout gate; accepted in [ADR-0020](docs/adr/0020-gkd-consolidation-cycle.md)
      with the three-arm benchmark (no KD vs top-k sidecar vs GKD) as the
      sidecar path's promotion/deprecation gate
- [ ] GRPO experiment on the CTF pack with a verifier reward (found flag as
      binary reward); candidate environment: the lictor lab targets. Via the
      pack-level reward hook (ADR-0022): `GRPOTrainer` callable reward
      functions, `loss_type="dapo"` (`epsilon_high=0.28`),
      `scale_rewards=False` (Dr. GRPO), `beta=0.0`.
- [ ] OPD (on-policy distillation v2): reverse-KL per-token dense reward
      over the GRPO trainer (Thinking Machines 2025 / OPD2 recipe, arXiv
      2607.15161); recette SFT puis OPD (`lr=1e-4`, groups_per_batch=512,
      lora_rank=128). Follows the GRPO experiment.
- [ ] OpenEnv environment wrapping the CTF sandbox (Docker target plus
      verifier reward), depends on the GRPO experiment
- [ ] Multi-harness RL through a capture proxy; requires vLLM/SGLang token
      ids and logprobs (llama.cpp is not supported there)

## Phase 4 - Generic backbones and capacity expansion

Deferred ambitions from the Sleep framework ([ADR-0020](docs/adr/0020-gkd-consolidation-cycle.md),
arXiv:2606.03979): BrainForge stays generic, so the backbone capabilities the
framework assumes must land as optional supports, not core assumptions.

- [ ] Mixture-of-Experts student support: a MoE base model behind the existing
      `student_base` config (PEFT expert-layer targeting, router-aware
      training pass); prerequisite for any Sleep-style capacity expansion
- [ ] Sleep-style capacity expansion: at each consolidation round, train a new
      low-rank adapter batch (frozen adapter pool) instead of overwriting the
      previous round; requires the consolidation cycle (phase 3) and an
      adapter merge/selection policy, on the 10GB VRAM budget
- [ ] Adapter pruning/merging at consolidation time (the synthetic-synaptic
      pruning analogue): rank trimming and adapter merging driven by the
      per-round holdout gate
- [ ] Hope/linear-attention backbone support: follow the Transformers
      implementations of the Titans/MIRAS lineage; no custom transformer code
      in BrainForge, so this is a watcher item until upstream lands
- [ ] Abliteration as a first-class sleep-phase pass: port the
      single-direction refusal method (Arditi et al., arXiv 2406.11717) to
      plain transformers hooks (no TransformerLens dependency), promote the
      experimental command per [ADR-0014](docs/adr/0014-abliteration-single-direction.md),
      with a before/after safety eval gate.
- [ ] Weight editing watcher: ROME/MEMIT (arXiv 2202.05262, 2210.07229) as an
      optional sleep-phase pass on a pyvene-style intervention base; scoped
      after the abliteration pass proves the hook pattern.
- [ ] MoE-to-dense distillation watcher: expert-scoring -> grouping -> dense
      FFN conversion + KD (arXiv 2605.28207 recipe) as an export option for
      MoE teachers; depends on the MoE student support above.
- [ ] Final end-to-end benchmark: untrained base vs consolidated student on
      unseen lictor lab challenges, real success rates over repetitions
      (pass@1/pass@k with variance), executed-loop oracle only, published
      with a contamination audit (n-gram overlap, holdout-id exclusion,
      provenance tags); single-seed honesty per the multi-harness RL guide

## Non-goals (standing)

- Mass-producing synthetic examples: quality, provenance and verification beat
  volume.
- Cloud-only workflows: everything must run on one workstation with an
  optional local gateway.

## Domain loop status (ctf pack, 2026-11-17)

Findings from the mr11 end-to-end cycle (commits `b887583` + teacher verdicts
in `experiments/teacher-pilot/`):

- The harness was the bottleneck, not the models: base 1.7B, trained mr7,
  local 4B teacher, paid cloud 27B teacher all scored ~0 on the hard
  families before the harness fix; after it, base and student score
  identically (24/280 clean, overlap exact).
- The facts-first dataset filter fixed the SFT regression
  (`attempts_without_llm_facts`: 267/440 for mr10, 0/440 for mr11) but
  produced no capability gain: same facts emitted, same augure decision,
  same harvest.
- Teacher capability is lab-bound: the free local 4B and the paid cloud 27B
  cover exactly the same 2/21 hard families (fmtstr-eval, setcontext_srop);
  cloud mining is unjustified at this state.
- Next lever is the lictor/augure chain on the 19 uncovered families
  (facts -> technique selection -> exploit execution -> flag), not the
  models. Trace the chain per family from the existing pilot trails, fix
  the most common death stage, re-pilot.
