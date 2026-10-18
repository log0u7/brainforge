# Training reference

Training targets: a 7-9B student on a single RTX 3080 10GB
([ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md)), with an
Unsloth backend for roughly 2x speed and lower VRAM. Three strategies
(QLoRA, full fine-tuning, knowledge distillation) and two backends (trl,
unsloth) compose freely; the dataset preparation is shared by all of them.

All training commands require CUDA and the training extra:

```bash
uv sync --extra training
```

For Unsloth: `uv sync --extra unsloth`.

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

## Strategies

| Strategy | What trains | Quantization | VRAM | Use when |
|---|---|---|---|---|
| `qlora` (default) | LoRA adapters | 4-bit nf4 | ~10GB | first pass, fast iteration |
| `full` | every weight | none (bf16) | model-size dependent | decision model after the dataset is proven |

Selected with `training.strategy` or `train run --strategy {qlora,full}`.
The LoRA rank/alpha/dropout come from config for `qlora`; `full` ignores
them.

## Backends

| Backend | Extra | Notes |
|---|---|---|
| `trl` (default) | `training` | Transformers + PEFT + bitsandbytes + TRL SFTTrainer |
| `unsloth` | `unsloth` | patched kernels, ~2x speed; 4bit or unquantized loads only (8bit fails fast) |

Selected with `training.backend` or `train run --backend {trl,unsloth}`.
Both backends share dataset preparation, splits, resume and the
`train_summary.json` contract (which records the backend and strategy).

## Knowledge distillation (experimental)

Two commands:

1. `pipeline run --kd` records the judge's per-token top-k logprobs into a
   `<dataset>.kd.jsonl` sidecar (requires a provider that serves logprobs;
   the mock provider produces word-level positions for dry runs).
2. `train run --kd` trains with
   `alpha * T^2 * CE(student_T, teacher_topk_T) + (1 - alpha) * CE(gold)`.

Details and limits: [knowledge distillation](../explanation/knowledge-distillation.md).

## Configuration

All knobs live in the `training` config section (defaults and semantics in
the [configuration reference](configuration.md)). The base model is
`training.base_model` (default `Qwen/Qwen3-8B`), never hardcoded; override
per run with `--base-model`.

| Field | Default | Meaning |
|---|---|---|
| `base_model` | `Qwen/Qwen3-8B` | HF model id or local dir |
| `strategy` | `qlora` | `qlora` or `full` |
| `backend` | `trl` | `trl` or `unsloth` |
| `quantization` | `4bit` | `4bit`, `8bit`, `none` (unsloth: 4bit or none) |
| `lora_rank` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | LoRA shape (strategy `qlora`) |
| `learning_rate` | 2e-4 | QLoRA sweet spot; full FT usually wants lower |
| `epochs` | 3 | |
| `batch_size` / `gradient_accumulation` | 1 / 16 | |
| `lr_scheduler_type` | `cosine` | |
| `warmup_steps` | 5 | |
| `max_grad_norm` | 1.0 | |
| `max_length` | 1024 | |
| `packing` | false | rejected for KD |
| `assistant_only_loss` | false | |
| `attn_implementation` | `sdpa` | `sdpa`/`flash_attention_2`/`eager` |
| `eval_batch_size` / `generate_batch_size` | 4 / 4 | |
| `save_steps` | 100 | |
| `seed` | 42 | |
| `output_dir` | `experiments` | |
| `kd_alpha` | 0.5 | KD loss weight |
| `kd_temperature` | 2.0 | KD temperature |
| `kd_topk` | 8 | teacher alternatives per token |

## Train, evaluate, export

```bash
brainforge train run --dataset-dir datasets/prepared \
  --strategy qlora --backend trl --epochs 3
brainforge train evaluate --model experiments/run-...
brainforge train task-eval --model experiments/run-...    # security domain
brainforge train export --model experiments/run-... --format merged
```

- `train run` writes `experiments/run-<timestamp>/` with the adapter,
  checkpoints and `train_summary.json`; `--resume` picks the latest
  checkpoint numerically.
- `train evaluate` computes corpus perplexity on a split (`eval.json`).
- `train task-eval` scores verdict/CWE accuracy against judge ground truth
  (security domain only).
- `train export --format merged` merges the adapter into standalone HF
  weights. `--format gguf` converts via llama.cpp to a quantized GGUF
  (default `Q4_K_M`, `--quant`) plus a `Modelfile`, and with `--ollama
  <name>` registers the model with Ollama. llama.cpp is located via
  `--llama-cpp-path`, the `LLAMA_CPP_DIR` env var, or PATH.
  See [ship to Ollama](../how-to/ship-to-ollama.md).

## Evaluation guidance

- **Primary signal**: `test_postcutoff.jsonl`, the cases teachers cannot
  have memorized. See [contamination](../explanation/contamination.md) for
  why this is the headline metric.
- Always compare base model vs fine-tuned student on identical prompts.
- The post-cutoff holdout stays the reference point across strategies;
  QLoRA-first then full-FT is a workflow, not a single number.

## GPU smoke test

`src/brainforge/training/smoke.py` validates a GPU environment with one
QLoRA forward/backward step on a tiny model:

```bash
uv sync --extra training
uv run python -m brainforge.training.smoke        # skips cleanly without CUDA
BRAINFORCE_SMOKE_MODEL=Qwen/Qwen3-0.6B ...        # override the probe model
uv run python -m brainforge.training.smoke_cpu    # CPU path (tiny random model)
```

The manual `gpu` CI job runs this on a self-hosted runner
([CI reference](ci.md)).
