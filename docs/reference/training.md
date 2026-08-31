# Training reference

Training targets: a 7-9B student with QLoRA (4-bit) on a single RTX 3080 10GB
([ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md)). Phase 1 ships the
preparation tooling; the QLoRA loop lands in phase 2.

## Prepare a dataset

```bash
uv run brainforge train prepare datasets/security_dataset.jsonl --output datasets/prepared
```

What it does:

1. Validates every record strictly and fails on the first invalid one (with
   index, id and reasons), so bad data never silently trains.
2. Splits source-grouped 80/10/10, seeded (see
   [dataset format](dataset-format.md)).
3. Exports `train.jsonl`, `validation.jsonl`, `test.jsonl`,
   `test_postcutoff.jsonl` and `stats.json`.

Options: `--train-ratio`, `--val-ratio`, `--seed`, `--output/-o`.

## Configuration

All knobs live in the `training` config section (defaults and semantics in the
[configuration reference](configuration.md)). The base model is
`training.base_model` (default `Qwen/Qwen3-8B`), never hardcoded.

Planned stack: Transformers + PEFT + bitsandbytes + TRL, installed via the
optional extra:

```bash
uv sync --extra training
```

## Phase-2 loop (planned)

- QLoRA: 4-bit quantization, `lora_rank`/`lora_alpha` from config, gradient
  checkpointing, gradient accumulation 16, bf16/fp16 per GPU support.
- The training set is `train.jsonl`; early stopping on `validation.jsonl`.
- Run directory: `experiments/<date>-<name>/` with config snapshot, dataset
  snapshot, metrics and report (created by `pipeline run --experiment`).

## Evaluation plan

- **Primary signal**: `test_postcutoff.jsonl`, the cases teachers cannot have
  memorized. See
  [contamination](../explanation/contamination.md) for why this is the
  headline metric.
- Metrics: vulnerability detection, CWE classification, severity, false
  positive rate, false negative rate, code understanding, patch understanding,
  remediation quality.
- Always compare base model vs fine-tuned student on identical prompts.
- The golden dataset (50-200 human-verified cases, phase 2) is the reference
  point; it never replaces the post-cutoff holdout.

## GPU smoke test

`src/brainforge/training/smoke.py` validates a GPU environment with one QLoRA
forward/backward step on a tiny model:

```bash
uv sync --extra training
uv run python -m brainforge.training.smoke        # skips cleanly without CUDA
BRAINFORCE_SMOKE_MODEL=Qwen/Qwen3-0.6B ...        # override the probe model
```

The manual `gpu` CI job runs this on a self-hosted runner
([CI reference](ci.md)).

## Phase-2 stubs

`brainforge train run`, `train evaluate` and `train export` raise an explicit
"phase 2" error instead of failing silently, and point at the
[ROADMAP](https://gitlab.com/6admin.io/brainforge/-/blob/main/ROADMAP.md).
