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

- [ ] First real dataset run: 20-50 cases (CVE + patch + vulnerable + fixed
      code), manual inspection of every accepted record
- [ ] Scale to 500-2,000 examples once quality is validated
- [ ] Golden dataset: 50-200 human-verified cases, including post-cutoff ones
- [ ] QLoRA training on RTX 3080 10GB (4-bit, low/moderate LoRA rank, gradient
      checkpointing) via TRL; student base model configurable
      (default `Qwen/Qwen3-8B`)
- [ ] Evaluation harness: vulnerability detection, CWE classification, severity,
      false positive/negative rates, code and patch understanding
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

## Non-goals (standing)

- Mass-producing synthetic examples: quality, provenance and verification beat
  volume.
- Cloud-only workflows: everything must run on one workstation with an
  optional local gateway.
