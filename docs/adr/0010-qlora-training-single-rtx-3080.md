# 0010 - Student training: QLoRA 4-bit on a single RTX 3080 10GB

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

The student model (7-9B) must be fine-tuned on the workstation's single RTX
3080 10GB, alongside the rest of the pipeline. Phase 1 ships preparation and
scaffolding; the training loop itself lands in phase 2 and its shape must be
decided now so the dataset tooling targets the right format and the config
schema stays stable.

## Decision Drivers

* 10 GB VRAM is the hard ceiling; the GPU is shared with RAG-free inference
  and everyday use.
* The base model must be configurable, never hardcoded.
* Training stack must be maintained and chat-fine-tune oriented (messages
  format from ADR-0007).
* Local-first: no cloud fine-tuning of vulnerability data.

## Considered Options

* Full fine-tuning of a 7-9B model
* Cloud fine-tuning service
* QLoRA 4-bit on the RTX 3080 via TRL (chosen)
* Inference-only (no training) with retrieval compensation

## Decision Outcome

Chosen option: "QLoRA 4-bit on the RTX 3080 via TRL". Phase 2 will use
Transformers + PEFT + bitsandbytes + TRL with: 4-bit quantization, low to
moderate LoRA rank (config: `lora_rank` 16, `lora_alpha` 32), gradient
checkpointing, gradient accumulation (config: 16) and bf16/fp16 per GPU
support. All parameters live in the `training` config section; `base_model`
defaults to `Qwen/Qwen3-8B`. Heavy dependencies are isolated in the optional
`[training]` extra; phase 1 ships `train prepare` (validate, split, export for
TRL), the `train run`/`evaluate`/`export` stubs raising explicit "phase 2"
errors, and `training/smoke.py` for the GPU CI job.

### Consequences

* Phase 1 cannot train models; the stubs fail loudly with a pointer to the
  roadmap instead of pretending.
* The config schema already carries the training knobs, so enabling the loop
  is additive.
* QLoRA rank/alpha defaults are starting points; experiments (the
  `experiments/` directory) decide final values.

### Confirmation

* `pyproject.toml` `[training]` extra pins the stack.
* `brainforge train prepare` produces `train/validation/test/test_postcutoff`
  JSONL plus `stats.json` (tested in `tests/unit/test_cli.py`).
* `brainforge train run` raises the documented phase-2 error (tested).
* The manual `gpu` CI job runs `brainforge.training.smoke` (1 QLoRA step,
  skipped without CUDA).

## Pros and Cons of the Options

### Full fine-tuning

* Good, because no adapter overhead at inference.
* Bad, because 7-9B full fine-tuning needs far more than 10 GB VRAM even with
  offloading, for marginal gains on a narrow domain.

### Cloud fine-tuning

* Good, because zero local GPU constraints.
* Bad, because vulnerability-focused datasets must not leave the workstation
  (local-first, ADR-0005 rationale applies), and costs scale with
  experimentation.

### QLoRA 4-bit on the 3080 (chosen)

* Good, because 7-9B fits comfortably with adapters, and PEFT/TRL are the
  maintained standard for chat SFT.
* Bad, because quantized base weights cap achievable quality; mitigated by
  the evaluation harness deciding, not assumption.

### Inference-only with retrieval compensation

* Good, because nothing to train.
* Bad, because it contradicts the project's thesis: the point is to measure
  whether verified knowledge improves a small model.

## Links

* [Training reference](../reference/training.md)
* ADR-0007 (dataset format consumed by TRL), ADR-0003 (post-cutoff benchmark)
* `src/brainforge/training/`, `pyproject.toml` `[training]` extra
