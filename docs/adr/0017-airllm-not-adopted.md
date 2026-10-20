# 0017 - AirLLM not adopted for low-VRAM inference or training

* Status: accepted
* Date: 2026-10-20

## Context and Problem Statement

BrainForge fine-tunes a 7-9B student with 4-bit QLoRA on a single RTX 3080
10GB ([ADR-0010](0010-qlora-training-single-rtx-3080.md)), and reaches
teachers, critic and judge through OpenAI-compatible providers
([ADR-0004](0004-provider-abstraction-openai-sdk.md)). AirLLM
(`lyogavin/airllm`) advertises 70B inference on a ~4GB GPU by streaming one
decoder layer from disk at a time, and a LoRA training path for recent
Qwen3.8 models.

The question: does AirLLM lift the 10GB ceiling for this project without
breaking the provider abstraction, training contracts or throughput?

## Decision Drivers

* 10GB VRAM is the hard ceiling, and the GPU is shared with everyday work.
* Dataset generation and task evaluation are throughput-bound: hundreds to
  thousands of structured completions per run.
* Vulnerability data stays local; no cloud training or cloud teacher.
* The provider abstraction is the only route to an LLM API and requires
  structured JSON (plus token logprobs for knowledge distillation).
* Training artifacts must stay consumable by PEFT/TRL, merge and GGUF export,
  knowledge distillation and full fine-tuning.
* Heavy dependencies stay behind optional extras; disk use must be bounded.

## Considered Options

* AirLLM as the inference engine for `train chat` and `train task-eval`.
* AirLLM behind a local OpenAI-compatible provider for pipeline roles.
* AirLLM as a third training backend next to `trl` and `unsloth`.
* Do not adopt AirLLM (chosen).

## Decision Outcome

Chosen option: "Do not adopt AirLLM", because the current 7-9B student
already fits 10GB with 4-bit QLoRA, while AirLLM's gains only materialize for
models that do not fit, at a throughput and integration cost the project
cannot absorb.

### Consequences

* Positive: no new runtime dependency, no layer-shard disk budget, no second
  inference or training contract to maintain.
* Negative: there is no local path to a 27B+ student on 10GB. Accepted: the
  target remains 7-9B until the post-cutoff benchmark justifies otherwise.
* Positive: the rejected designs are recorded, so a future evaluation starts
  from evidence instead of re-running the comparison.

Revisit when any of these becomes true:

1. The student target moves to 27B or larger and must still train locally on
   the 3080.
2. AirLLM ships PEFT-format adapters or a TRL-compatible trainer, so the
   existing merge, eval and GGUF paths keep working.
3. AirLLM exposes an OpenAI-compatible server with token logprobs, and a
   local teacher becomes mandatory at an acceptable throughput.
4. AirLLM training becomes model-agnostic instead of two Qwen3.8 classes.

### Confirmation

* `pyproject.toml` and `uv.lock` contain no `airllm` dependency.
* `TrainingConfig.backend` still accepts only `trl` and `unsloth`
  (`src/brainforge/config/models.py:105`), dispatched in
  `src/brainforge/cli/train_cmd.py:34`.
* `src/brainforge/training/models.py:9` still loads HF or PEFT checkpoints
  only.
* This ADR is registered in `docs/adr/index.md`.

## Pros and Cons of the Options

### AirLLM inference engine for chat and task eval

* Good, because it runs a merged model larger than VRAM at full precision,
  with a small API surface (`AutoModel.from_pretrained`, `generate`).
* Bad, because every forward pass streams all layers from disk: reported
  throughput for 70B-class models is roughly 0.5-2 tokens/s, which makes
  batch task evaluation impractical.
* Bad, because it loads merged weights only. PEFT adapters from `train run`
  must be merged first (`train export --format merged`), adding a mandatory
  export step and a second full checkpoint on disk.
* Bad, because layer shards roughly double checkpoint disk use, and the
  newest model families need model-specific `transformers` builds or
  `flash-attn`.

### AirLLM behind a local provider

* Good, because it would keep teacher calls inside the existing provider
  abstraction and remove cloud dependence.
* Bad, because AirLLM has no HTTP server and no token-logprob API. A custom
  server wrapper would have to reproduce `/v1/chat/completions`, structured
  JSON and logprobs, which is new security and correctness surface.
* Bad, because multi-teacher dataset generation at 0.5-2 tokens/s takes
  hundreds of hours per batch. The pipeline is designed for parallel remote
  calls, not a single streamed local model.

### AirLLM training backend

* Good, because it is the only demonstrated way to train a 27B+ model on a
  10GB card, keeping frozen weights on disk and adapters in VRAM.
* Bad, because it is not HF Trainer or PEFT: `AirLLMLoRA` and
  `AirLLMLoRAQwen4Exp` use a custom `train_step` and save `.pt` adapters, so
  KD loss, full fine-tuning, `merge_and_unload`, GGUF export and the
  `train_summary.json` contract would each need a bypass.
* Bad, because the documented training targets are specific Qwen3.8 models
  (27B dense and Flash-Next), not the configurable `training.base_model`.
* Bad, because it consumes a different JSONL shape and has no evaluation or
  checkpoint-resume contract comparable to the current trainer.

### Do not adopt (chosen)

* Good, because it keeps one inference path and two proven training backends,
  with no throughput, disk or dependency risk.
* Bad, because the project forgoes the only plausible local route to a 27B+
  student. Mitigated by the explicit revisit triggers above.

## Links

* [ADR-0004](0004-provider-abstraction-openai-sdk.md) (provider abstraction)
* [ADR-0010](0010-qlora-training-single-rtx-3080.md) (10GB QLoRA baseline)
* [ADR-0012](0012-unsloth-backend-opt-in.md) (backend isolation precedent)
* [Training reference](../reference/training.md)
* [Provider reference](../reference/providers.md)
* [Roadmap phase 3](../../ROADMAP.md) (local provider backends)
* AirLLM: <https://github.com/lyogavin/airllm>, PyPI `airllm` 4.0.0
