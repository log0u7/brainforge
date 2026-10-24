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
- [ ] Benchmark base model vs fine-tuned student, primarily on
      `test_postcutoff.jsonl` to prove analysis rather than recitation

## Phase 3 - Measure, ablate, extend

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
- [ ] On-policy distillation backend (TRL `DistillationTrainer`, GKD) and a
      three-arm benchmark: no KD vs top-k sidecar (ADR-0015) vs on-policy
- [ ] GRPO experiment on the CTF pack with a verifier reward (found flag as
      binary reward); candidate environment: the lictor lab targets
- [ ] OpenEnv environment wrapping the CTF sandbox (Docker target plus
      verifier reward), depends on the GRPO experiment
- [ ] Multi-harness RL through a capture proxy; requires vLLM/SGLang token
      ids and logprobs (llama.cpp is not supported there)

## Non-goals (standing)

- Mass-producing synthetic examples: quality, provenance and verification beat
  volume.
- Cloud-only workflows: everything must run on one workstation with an
  optional local gateway.
