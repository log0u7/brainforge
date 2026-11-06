# How to expand the facts dataset from historical rollouts

How-to for rebuilding the MR7 facts dataset (`datasets/sft-mr7-facts/train.jsonl`)
from every historical source, training the student on it, and serving it for
the [final benchmark](run-final-benchmark.md). The extraction is deterministic:
same sources, same output. Nothing is lost along the way; every dropped step is
counted and reported.

The protocol this dataset serves: the student must emit a single JSON object
`{"facts": [{"name": "...", "value": "..."}]}` from challenge material alone
(single-turn, schema-constrained decoding). The vuln class is the most
important fact: it decides which exploit technique the engine picks.

## Prerequisites

- The repo checked out, `uv sync` done.
- The lictor case files under `examples/cases/` (they provide the prompt
  material per target; `case_material()` resolves each target to its
  non-eval twin file).
- For training: the GPU host (RTX 3080 10 GB is enough for Qwen3-1.7B
  QLoRA 4-bit), torch with CUDA, and the training extra
  (`uv sync --extra training`).
- For export: llama.cpp built on the GPU host (default `C:\llama.cpp`).

## 1. Extract the union dataset

The extractor mines every `(target, facts)` step from four historical
sources and turns each into a single-turn record:

```sh
python scripts/expand_sft_mr7.py \
  --source-glob \
    'experiments/experiments/rollouts/trajectories/*.jsonl' \
    'experiments/experiments/rollouts-twins/trajectories/*.jsonl' \
    'experiments/experiments/consolidation/round-1/mix/train.jsonl' \
    'datasets/sft-mr6-v2/train.jsonl' \
  --out datasets/sft-mr7-facts/train.jsonl
```

Expected output (58 records):

```
wrote 58 records to datasets/sft-mr7-facts/train.jsonl
dropped: 381 empty-facts steps, 0 without material
  fmtstr: 13
  orw: 8
  ret2func: 5
  ret2plt: 10
  shellcode: 4
  uaf_heap-002: 5
  uaf_train_a: 7
  uaf_train_b: 6
```

What the extractor guarantees:

- **Ground truth vuln class.** The vuln predicate is never taken from the
  trajectory; it comes from `VULN_TRUTH` in `scripts/assemble_sft_mr7.py`
  (hand-verified per target). Trajectory facts are only trusted for the
  supporting predicates (`fmtstr_read`, `seccomp`, ...).
- **Twin aliasing.** Trajectory IDs strip both numeric suffixes, so the uaf
  twin arrives as `uaf_heap`; the alias maps it to `uaf_heap-002`, whose
  case file is the non-eval twin (`ctf-lictor-uaf_heap-002-001.json`). The
  eval target `uaf_heap-001` is never touched.
- **Empty-facts drop.** 381 steps emit zero facts. They are dropped: an
  empty emission teaches the model to emit nothing, the exact bench failure
  the protocol measures. The count is reported so nothing disappears
  silently.
- **Deduplication on (material, facts).** Two targets with identical facts
  are two distinct training samples because their prompt material differs.
  The dedup key is `json.dumps([material, payload], sort_keys=True)`.

## 2. Train the student

```sh
uv run brainforge train run \
  --dataset-dir datasets/sft-mr7-facts \
  --base-model Qwen/Qwen3-1.7B \
  --strategy qlora --quantization 4bit --device cuda \
  --epochs 20
```

Gotcha: total steps = rows x epochs must exceed the warmup (5 steps) by an
order of magnitude. With 58 rows, 20 epochs = 1160 steps and the LR schedule
is meaningful. A 3-epoch run (174 steps) is fine; a 3-epoch run on 6 rows
(3 steps) never leaves warmup and trains nothing.

Expected: final loss below ~0.1, mean token accuracy above ~0.98. The run
directory (`experiments/run-<date>-<id>/`) holds the adapter weights and the
training report.

## 3. Export to GGUF

```sh
uv run brainforge train export \
  --model experiments/run-<date>-<id> \
  --format gguf --output models/export/mr7 --quant Q8_0 \
  --llama-cpp-path C:\llama.cpp
```

Q8_0 matches the base model quantization used by the benchmark, so the only
difference between the two arms is the training.

## 4. Serve the trained student

On the GPU host, register an nssm service (an SSH-launched `start /B`
process dies with the session):

```powershell
nssm install bf-bench-mr7 C:\llama.cpp\llama-server.exe
nssm set bf-bench-mr7 AppParameters -m C:\Users\eppgr\brainforge\models\export\mr7\model-q8_0.gguf --host 0.0.0.0 --port 9934 --alias trained-mr7 -c 8192 --temp 0.0 -ngl 99
nssm start bf-bench-mr7
```

Windows blocks new ports by default; open one rule for the new port:

```powershell
netsh advfirewall firewall add rule name="llama-server 9934" dir=in action=allow protocol=TCP localport=9934
```

Verify from the control machine: `curl http://<gpu-host>:9934/v1/models`.

## 5. Benchmark

Follow [How-to: run the final benchmark](run-final-benchmark.md) with
`TRAINED_URL` pointing at the mr7 service. Read the verdict from the clean
set only; the all-challenges verdict includes challenges mined in training
and is reported as contamination context, never as the score.

## Reproducibility notes

- The extraction reads only committed sources (trajectories, consolidation
  mix, mr6-v2) plus the committed case files. Re-running it reproduces the
  same 58 records byte-for-byte.
- Every dropped step is counted (`empty_steps`, `no_material`) and printed;
  the run log is the provenance record.
- The vuln ground truth lives in `VULN_TRUTH`; adding a new lab target means
  adding one hand-verified entry there plus its case file.
