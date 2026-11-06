# Run the final benchmark (base vs trained on unseen challenges)

How-to for reproducing the [ADR-0021](../adr/0021-final-benchmark-contract.md)
final benchmark: the base model and the trained student solve the same
oracle-verified challenges through the lictor lab loop, and a contamination
audit proves the evaluation set was never seen in training.

Prerequisites:

- A machine that can run the lictor lab Docker stack (a laptop is fine:
  the lab serves the challenge targets, the GPU host only serves models).
- Docker installed, with access to the [lictor repo](https://github.com/log0u7/lictor).
- The trained student exported to GGUF on the GPU host
  (`brainforge train export --format gguf`).
- The base model GGUF (Qwen3-1.7B base, e.g. `Qwen3-1.7B-Q8_0.gguf`).
- The dataset that trained the student (its challenge IDs feed the
  contamination audit). Build it from the raw corpus with
  [How-to: generate a dataset](generate-dataset.md), then prepare it
  with [How-to: manage datasets](manage-datasets.md).

## 1. Start the lictor lab

The lab serves each challenge binary over HTTP (the pilot acquires it
from port 8000) and runs one TCP listener per challenge target (the
exploit connects there to read the flag).

```sh
git clone https://github.com/log0u7/lictor && cd lictor
docker compose -f lab/docker-compose.yml up -d
```

The lab publishes: HTTP `8000` (binaries), `9982-9999` plus `8897-8899`
(the challenge listeners). Verify with `docker ps`.

## 2. Serve the two models

llama.cpp serves one model per instance, so the benchmark needs the
base and the trained student on two separate ports. On the GPU host
(preferred: nssm services):

```powershell
# From repo root on the control machine, or on gpu3080
powershell -ExecutionPolicy Bypass -File scripts/setup_bench_servers.ps1 -Apply
nssm start bf-bench-base
nssm start bf-bench-trained
```

Manual equivalent (must include `--jinja` and match protocol flags):

```cmd
C:\llama.cpp\llama-server.exe -m <path>\Qwen3-1.7B-Q8_0.gguf --host 0.0.0.0 --port 9932 --alias base-qwen3-1.7b -c 16384 -ngl 99 -fa on --parallel 1 --jinja --metrics
C:\llama.cpp\llama-server.exe -m <path>\model-q4_k_m.gguf --host 0.0.0.0 --port 9933 --alias trained-mr6 -c 16384 -ngl 99 -fa on --parallel 1 --jinja --metrics
```

The `--alias` value is the model name the runner routes on: it must
match the `--base-model` / `--trained-model` arguments below. The
preflight checks enforce jinja/chat template support, distinct servers,
healthy latency, and `json_schema` constrained decoding (the fact
request relies on it, and a server that ignores `response_format`
invalidates the measurement).

## 3. Run the benchmark

Preferred entry point (runs preflight, passes attempt cap, enforces
protocol):

```sh
scripts/run_final_bench.sh mr6-v2-seeded 10
```

Environment overrides: `BASE_URL`, `BASE_MODEL`, `TRAINED_URL`,
`TRAINED_MODEL`, `ATTEMPT_CAP`. Manual equivalent:

```sh
uv run python -m brainforge.experiments.final_bench \
  --training-ids datasets/sft-mr6/train.jsonl \
  --training-ids datasets/sft-mr6-v2/train.jsonl \
  --training-ids datasets/eval/grpo-translated.jsonl \
  --base-url http://192.168.1.26:9932/v1 \
  --base-model base-qwen3-1.7b \
  --trained-url http://192.168.1.26:9933/v1 \
  --trained-model trained-mr6 \
  --repetitions 10 \
  --attempt-cap 600
```

The runner:

1. Loads the oracle-verified manifest cases onto the live lab targets.
   Cases with no live listener (the ROP Emporium set in the current
   lab) are skipped and reported, never silently dropped.
2. Collects the training challenge IDs and refuses to run the whole
   benchmark if one leaks into the evaluation list (contamination
   audit).
3. Runs `run_bench_lab`: the base model first, then the trained
   student, one pilot trajectory per (model, challenge, repetition).
   The per-target oracle regex verifies the expected flag on the live
   service output, never a recited training flag.
4. Writes `results.json` (per-challenge breakdown) and `results.md`
   (comparison table) into the workdir.

## 4. Read the results

`results.md` holds the comparison table:

```
| model | successes/attempts | success_rate |
```

The success rate is the honest measurement: a recited training flag on
the wrong challenge counts as a miss, not a success. A contaminated
benchmark (training IDs in the eval list) exits non-zero and the whole
number is void.
