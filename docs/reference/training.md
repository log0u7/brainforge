# Training reference

Primary target: a 7-9B student on a single RTX 3080 10GB
([ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md)), with an
Unsloth backend for roughly 2x speed and lower VRAM. Four strategies
(QLoRA, full fine-tuning, knowledge distillation, GRPO reinforcement
learning) and three backends (auto, trl, unsloth) compose freely; the
dataset preparation is shared by all of them.

All training commands require the training extra; the device, dtype and
quantization resolve automatically for the detected hardware:

```bash
uv sync --extra training
```

For Unsloth: `uv sync --extra unsloth`.

Run `brainforge train check` before a real run: it validates hardware, model
compatibility and the memory estimate without downloading weights (see
[hardware reference](hardware.md)).

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
| `qlora` (default) | LoRA adapters | `auto`: 4-bit nf4 on CUDA/ROCm, none elsewhere | ~10GB on CUDA | first pass, fast iteration |
| `full` | every weight | none (bf16/fp16 per device) | model-size dependent | decision model after the dataset is proven |
| `grpo` | RL on sampled completions, scored by verifiable rewards | per backend | model-size dependent | domains with a verifiable signal (CTF flags, refusal compliance) |

Selected with `training.strategy` or `train run --strategy {qlora,full,grpo}`.
The LoRA rank/alpha/dropout come from config for `qlora`; `full` ignores
them. `qlora` with quantization `none` (the auto default off CUDA/ROCm)
trains plain LoRA.

## Backends

| Backend | Extra | Notes |
|---|---|---|
| `auto` (default) | `training` | resolves to `trl` (reserved for future selection logic) |
| `trl` | `training` | Transformers + PEFT + bitsandbytes + TRL SFTTrainer |
| `unsloth` | `unsloth` | patched kernels, ~2x speed; CUDA/ROCm only; 4bit or unquantized loads only (8bit fails fast) |

Selected with `training.backend` or `train run --backend {auto,trl,unsloth}`.
Both backends share dataset preparation, splits, resume and the
`train_summary.json` contract (which records the backend, strategy, device,
dtype, quantization and attention implementation actually used).

AirLLM is evaluated and not adopted
([ADR-0017](../adr/0017-airllm-not-adopted.md)): layer streaming lowers VRAM
but not for a 7-9B model that already fits, and its custom training API is
incompatible with the PEFT/TRL, KD and GGUF contracts.

## Reinforcement learning (GRPO)

`strategy: "grpo"` (`train run --strategy grpo`) trains with TRL
`GRPOTrainer`: the model samples several completions per prompt and a
verifiable reward scores them, no labeled corpus required. Rewards are
named callables in `brainforge.training.rewards`:

| Reward | Signal | Scores |
|---|---|---|
| `ctf_flag` | flag found vs `grpo_flags.jsonl` (`id -> expected_flag` sidecar) | exact match 1.0 on the verified tier, format-only 0.3, absent 0 |
| `compliance` | refusal tics after abliteration | compliant answer 1, refusal 0 |

The registry API (`register_reward(name, fn)` / `get_reward(name)`) makes
rewards pluggable, and `compose_reward("ctf_flag,compliance*0.5")` builds
a weighted composition so one GRPO pass trains against several signals.

Resolution order in `train_grpo`: an explicit reward spec >
`training.grpo_reward` > the domain pack's `grpo_reward()` hook >
`BrainforgeError` (no silent default: an unusable reward fails before
training starts). A new domain joins RL training by declaring one reward
in its pack; `CtfPack` declares `ctf_flag`. Phase-3 RL
([ADR-0018](../adr/0018-post-training-expansion-deferred.md)) is
therefore generic rather than CTF-specific.

## Multi-turn data and assistant-only loss

The `agentic` pack emits records that interleave assistant tool calls with
`tool` observations (see the [dataset format](dataset-format.md)). With the
default `training.assistant_only_loss=false`, the SFT loss scores every
completion-side token, tool observations included. Set it to `true` to mask
the loss to assistant turns: it requires a chat template with generation
markers, and TRL swaps in patched training templates for known families
(Qwen3, DeepSeek-V3) when tool calling is enabled. `brainforge train run`
warns when the training split holds tool turns or multi-assistant records
while the flag is off.

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
| `strategy` | `qlora` | `qlora`, `full` or `grpo` |
| `backend` | `auto` | `auto` (trl), `trl` or `unsloth` |
| `device` | `auto` | `auto`, `cuda`, `xpu`, `mps` or `cpu` |
| `quantization` | `auto` | `auto`, `4bit`, `8bit`, `none` (unsloth: 4bit or none; CUDA/ROCm only) |
| `lora_rank` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | LoRA shape (strategy `qlora`) |
| `learning_rate` | 2e-4 | QLoRA sweet spot; full FT usually wants lower |
| `epochs` | 3 | |
| `batch_size` / `gradient_accumulation` | 1 / 16 | |
| `lr_scheduler_type` | `cosine` | |
| `warmup_steps` | 5 | |
| `max_grad_norm` | 1.0 | |
| `max_length` | 1024 | |
| `packing` | false | rejected for KD |
| `assistant_only_loss` | false | mask the loss to assistant turns (agentic multi-turn data) |
| `attn_implementation` | `sdpa` | `sdpa`/`flash_attention_2`/`eager` |
| `eval_batch_size` / `generate_batch_size` | 4 / 4 | |
| `save_steps` | 100 | |
| `seed` | 42 | |
| `output_dir` | `experiments` | |
| `kd_alpha` | 0.5 | KD loss weight |
| `kd_temperature` | 2.0 | KD temperature |
| `kd_topk` | 8 | teacher alternatives per token |
| `grpo_reward` | `""` | reward spec for `strategy: "grpo"` (`reward` or weighted list, e.g. `ctf_flag,compliance*0.5`); empty delegates to the domain pack hook |

## Train, evaluate, export

```bash
brainforge train run --dataset-dir datasets/prepared \
  --strategy qlora --backend trl --epochs 3
brainforge train evaluate --model experiments/run-...
brainforge train task-eval --model experiments/run-...    # security domain
brainforge train export --model experiments/run-... --format merged
```

- `train run` writes `experiments/run-<timestamp>/` with the adapter,
  checkpoints and `train_summary.json` (which records the resolved device,
  dtype, quantization and effective strategy); `--resume` picks the latest
  checkpoint numerically.
- `train evaluate` computes corpus perplexity on a split (`eval.json`);
  `--quantization auto` matches the training-time quantization policy.
- `train task-eval` scores verdict/CWE accuracy against judge ground truth
  (security domain only); `--quantization auto` by default.
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

## Device smoke test

`src/brainforge/training/smoke.py` validates an accelerator environment with
one QLoRA forward/backward step on a tiny model:

```bash
uv sync --extra training
uv run python -m brainforge.training.smoke        # skips cleanly without an accelerator
BRAINFORCE_SMOKE_MODEL=Qwen/Qwen3-0.6B ...        # override the probe model
uv run python -m brainforge.training.smoke_cpu    # CPU path (tiny random model)
```

The manual `gpu` CI job runs this on a self-hosted runner
([CI reference](ci.md)).

## Device support

Which devices train, how dtype and quantization resolve, and the preflight
command: [hardware reference](hardware.md).
