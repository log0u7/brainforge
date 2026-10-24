# Hardware reference

How BrainForge picks a training device, what each device supports, and how to
verify a machine before a real run.

Everything here is implemented and unit-tested, but only CUDA on an RTX 3080
is hardware-validated. Run `brainforge train check` on your machine and
report what you find.

## Preflight

```bash
uv sync --extra training
uv run brainforge train check
```

`train check` inspects the detected device, the model metadata on the Hub (or
a local path), and estimates memory for the selected plan. It never downloads
model weights: only `config.json`, tokenizer files and the chat template are
fetched. It exits 0 when nothing fails and no warning survives `--strict`;
use `--json` for a machine-readable report.

What it checks:

- training deps importable, device available, bf16 support
- backend/quantization/attention valid for the device
- model architecture maps to a causal LM, pipeline tag, chat template present
- no pre-quantized checkpoints, no remote custom code (`auto_map`)
- gated/private models report the `HF_TOKEN` requirement, license surfaced
- max length vs model context, dataset split shape (with `--dataset-dir`)
- memory fit for the resolved plan, free disk space

## Device resolution

`training.device` selects the device. `auto` picks the first available:

1. CUDA (NVIDIA) or ROCm (AMD, `torch.version.hip` set)
2. Intel XPU (`torch.xpu`)
3. Apple MPS (`torch.backends.mps`)
4. CPU

A `torch_xla` (TPU) or `torch_npu` (Ascend NPU) runtime is detected and
rejected with an explicit error; pass `--device cpu` to train there anyway.
An explicit `cpu` always works. CUDA accepts `--device cuda` for either
NVIDIA or AMD builds.

| Device | Dtype (auto) | Quantization (auto) | Status |
|---|---|---|---|
| CUDA | bf16 if supported, else fp16 | 4-bit nf4 | hardware-validated (RTX 3080) |
| ROCm | bf16 if supported, else fp16 | 4-bit nf4 | implemented, not validated |
| XPU | bf16 if supported, else fp16 | none | implemented, not validated |
| MPS | fp16 | none | implemented, not validated |
| CPU | fp32 | none | implemented, not validated |

Rules that hold on every device:

- 4-bit/8-bit quantization (bitsandbytes) only on CUDA/ROCm; anything else
  fails fast with a message, not a crash mid-training.
- `qlora` with quantization `none` degrades to plain LoRA.
- `full` ignores quantization and trains every weight.
- `flash_attention_2` requires CUDA/ROCm; use `sdpa` (default) or `eager`.
- Unsloth requires CUDA/ROCm; use the trl backend elsewhere.

## Installing accelerator torch builds

PyPI `torch` ships CUDA. For AMD or Intel GPUs, replace it after
`uv sync --extra training`:

```bash
make install-rocm    # torch 2.13.0 from the ROCm 7.1 wheel index
make install-xpu     # torch 2.13.0 from the Intel XPU wheel index
```

Both indexes are overridable (`make install-rocm ROCM_INDEX=...`) and were
verified to carry cp312 torch 2.13.0 wheels. On macOS, PyPI torch includes
MPS support. TPU and NPU runtimes are out of scope for this release.

## Memory estimate

`train check` estimates the plan's memory from the model's parameter count
(read from `safetensors` metadata when present):

- 4-bit: ~0.55 bytes/param; 8-bit: ~1.05; otherwise the dtype's bytes/param
- full fine-tuning adds optimizer (AdamW) and gradient state
- LoRA adapters scale with rank, hidden size and layer count
- activations assume gradient checkpointing; ~10% overhead plus a fixed term

The estimate is an estimate, not a guarantee. When free memory is known
(CUDA/XPU), the report compares the two; otherwise it prints the estimate
alone and leaves the judgment to you.
