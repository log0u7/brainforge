# Unsloth training backend as an opt-in extra

* Status: accepted
* Date: 2026-10-18

## Context and Problem Statement

The stock trl path (Transformers + PEFT + bitsandbytes + TRL) fits a
7-9B QLoRA on one RTX 3080 10GB but leaves iteration slow. Unsloth's
patched kernels give roughly 2x speed and lower VRAM, which matters for
the quick-pass phase of the workflow. Unsloth is a heavy dependency and
its trainer API differs slightly across versions (the `tokenizer` kwarg
became `processing_class` in trl 1.x).

## Decision Drivers

* Training speed and VRAM are the user's bottleneck, not correctness of
  the trainer.
* The base package and unit suite must keep running without any training
  extra.
* Unsloth's API churns; the wrapper must survive both APIs.

## Considered Options

* Replace trl with Unsloth everywhere.
* Add Unsloth as the default backend, trl as fallback.
* Add Unsloth as an opt-in backend behind a config field and an extra.

## Decision Outcome

Chosen option: "opt-in backend", because it keeps the proven trl path as
the default, isolates the heavy dependency behind `uv sync --extra
unsloth`, and the dispatch is one config field.

### Consequences

* Positive: default install unchanged; speed available on demand.
* Negative: two training paths to keep correct.
* Mitigation: shared `_trainer_kwargs`/`_sft_config`/dataset preparation;
  the backend only swaps model loading and the trainer wrapper; both write
  the same `train_summary.json` contract (which records the backend).

### Confirmation

* `TrainingConfig.backend` (`trl` default) and `train run --backend`.
* `tests/unit/test_unsloth_backend.py` injects a fake unsloth module and
  asserts the call sequence, the clean failure without the extra, and the
  8bit rejection.
* `pyproject.toml` extra `unsloth`; `uv lock` resolves it.

## Pros and Cons of the Options

### Unsloth everywhere

* Good, because one path.
* Bad, because a hard dependency on a fast-moving library and a broken
  default install.

### Unsloth default

* Good, because speed by default.
* Bad, because the default install suddenly needs the extra.

### Opt-in backend

* Good, because zero blast radius on the default path.
* Bad, because users must know the extra exists.

## Links

* [Training reference](../reference/training.md)
* [ADR-0010](0010-qlora-training-single-rtx-3080.md)
