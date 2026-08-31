# Training

Training targets: a 7-9B student with QLoRA (4-bit) on a single RTX 3080 10GB.
Phase 1 ships the preparation and the scaffolding; the QLoRA loop lands in
phase 2 (see ROADMAP).

## Prepare a dataset

```bash
uv run brainforge train prepare datasets/security_dataset.jsonl --output datasets/prepared
```

Steps: strict validation of every record (fail fast on the first invalid one),
source-grouped split (80/10/10, seeded), export of `train.jsonl`,
`validation.jsonl`, `test.jsonl`, `test_postcutoff.jsonl` and a `stats.json`
summary including the post-cutoff warning.

## Configuration

```json
"training": {
  "base_model": "Qwen/Qwen3-8B",
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "learning_rate": 0.0002,
  "epochs": 3,
  "batch_size": 1,
  "gradient_accumulation": 16,
  "quantization": "4bit"
}
```

The base model lives in config, never in code. Planned stack: Transformers +
PEFT + bitsandbytes + TRL.

## Evaluation plan

- Primary signal: `test_postcutoff.jsonl` - cases the teachers cannot have
  memorized. Gains here are analysis, not recitation.
- Metrics: vulnerability detection, CWE classification, severity, false
  positive rate, false negative rate, code understanding, patch understanding,
  remediation quality.
- Always compare base model vs fine-tuned student on identical prompts.

## GPU smoke test

`src/brainforge/training/smoke.py` runs one QLoRA forward/backward step on a
tiny model to validate a GPU environment (used by the manual `gpu` CI job; it
exits cleanly when no CUDA device is present). Override the probe model with
`BRAINFORCE_SMOKE_MODEL`.

```bash
uv sync --extra training
uv run python -m brainforge.training.smoke
```

## Stubs

`brainforge train run`, `train evaluate` and `train export` currently raise a
documented "phase 2" error instead of failing silently.
