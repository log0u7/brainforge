# CLI reference

Global options: `--config/-c` selects the configuration file (default
`config/config.json`, overridable with `BRAINFORCE_CONFIG`). Output uses
rich tables; failures exit 1 with a reason on stderr.

BrainForge distills a multi-teacher pipeline into a student model: cases
in, verified dataset out, trained model out. Domains are pluggable
(`security` and `coding` are reference packs; `ticket_triage`, `ctf`,
`code_review`, `security_audit` and `agentic` are example packs). See the
[architecture explanation](../explanation/architecture.md).

## config

| Command | Description |
|---|---|
| `brainforge config validate` | validate references, judge independence, dry-build providers |
| `brainforge config providers` | table: name, type, base URL, API style, key set |
| `brainforge config schema --write` | regenerate `config/schema.json` from the models |
| `brainforge config schema --check` | fail when the schema is out of sync |
| `brainforge config path` | print the resolved config path |
| `brainforge config set <dotted.path> <value>` | set a nested value (created as dict when missing, refused when a non-dict is in the way); value parsed as JSON scalar first, raw string otherwise |
| `brainforge config set-model <role> <provider> <model>` | set `models.<role>.provider` and `.model` in one write, preserving sibling keys; role and provider must exist |
| `brainforge providers check-logprobs [--provider P --model M] [--topk N] [--all-configured]` | probe whether a provider serves per-token logprobs for a model (KD teacher signal); `--all-configured` sweeps every provider/model pair into a table |

## models

| Command | Description |
|---|---|
| `brainforge models list` | table: name, provider, model id, family, cutoff |
| `brainforge models test <name>` | tiny request; prints usage and reply, exit 1 on failure |

## roles

| Command | Description |
|---|---|
| `brainforge roles list` | roles with inferred kind, provider, model id, temperature |

## teacher

| Command | Description |
|---|---|
| `brainforge teacher run --role <role> --case <id\|path>` | one role on one case; prints the structured analysis |
| `brainforge teacher ensemble --case <id\|path>` | teachers + judge; prints judge JSON and teacher summaries |

Shared options: `--cases-dir` (default `data/raw`), `--config/-c`.

## pipeline

```bash
brainforge pipeline run security_dataset \
  --input examples/cases \
  --provider mock \
  --no-rag \
  --fresh \
  --experiment first-run \
  --kd
```

| Option | Default | Description |
|---|---|---|
| `name` (argument) | required | pipeline key from `pipelines` |
| `--input/-i` | `data/raw` | cases directory |
| `--output/-o` | `datasets/<pipeline>.jsonl` | dataset output path |
| `--provider` | none | override every provider (e.g. `mock`) |
| `--fresh` | off | bypass the response cache |
| `--rag/--no-rag` | on | use the local RAG index when present |
| `--rag-backend` | `fastembed` | `fastembed` or `hashing` |
| `--experiment <name>` | none | create `experiments/<date>-<name>/` (config, dataset, metrics, report) |
| `--kd` | off | record judge top-k logprobs into `<dataset>.kd.jsonl` (see [KD](../explanation/knowledge-distillation.md)) |
| `--config/-c` | `config/config.json` | configuration file |

Every run also writes a structured state file next to the dataset
(`<output>.state.json`), updated atomically before and after each case, so
long runs stay observable while they happen: top-level `status`
(`running`/`done`), `total_cases`, `done`, `accepted`, `rejected`, `errors`,
`current_case`, plus one entry per case (`id`, `status`, `duration_ms`,
`gate_reasons` count).

| Command | Description |
|---|---|
| `brainforge pipeline status <dataset\|state path>` | print run progress (status, counts, per-case table) from the state file; exit 1 with a clean error when it is missing |
| `brainforge pipeline kd-pass` | run `pipeline run ctf --kd --no-rag` per case file, then assemble a sidecar-aligned KD dataset (`train.jsonl` + `kd.jsonl`) |

`kd-pass` options:

| Option | Default | Description |
|---|---|---|
| `--cases/-i` | required | directory of case JSON files, processed one subprocess per case |
| `--out-dir/-o` | required | output dataset directory |
| `--scratch-dir` | `<out-dir>-parts` | per-case outputs; existing non-empty files are skipped on rerun (resume) |
| `--only-solved` | off | keep only non-fallback (solved) records |
| `--kd-timeout` | `900` | seconds per case subprocess; timed-out or failed cases are dropped and their scratch output deleted |
| `--config/-c` | `config/config.json` | configuration file, forwarded to each case run |

## rag

| Command | Description |
|---|---|
| `brainforge rag index <path> --backend <b>` | clear and rebuild the index from a directory |
| `brainforge rag search "<query>" --k 5 --backend <b>` | search with provenance table |

`--index-dir` defaults to `rag/index` on both.

## dataset

| Command | Description |
|---|---|
| `brainforge dataset build <pipeline>` | build via a pipeline (defaults to the mock provider) |
| `brainforge dataset validate <file.jsonl>` | strict record validation; exit 1 on any invalid record |
| `brainforge dataset inspect <file.jsonl>` | contamination stats, agreement rates, duplicates |
| `brainforge dataset split <file.jsonl>` | source-grouped splits + post-cutoff holdout (`--train-ratio`, `--val-ratio`, `--seed`) |
| `brainforge dataset filter --sidecar <kd.jsonl>` | keep records whose id appears in the sidecar; `--out` writes a filtered JSONL, `--out-dir` writes a training-ready `train.jsonl` + `kd.jsonl` directory |

## train

| Command | Description |
|---|---|
| `brainforge train prepare <file.jsonl> [-o dir]` | validate, split, export for TRL (+ `stats.json`) |
| `brainforge train check` | preflight: hardware, model compatibility and memory estimate before training |
| `brainforge train run` | train the student (QLoRA, full fine-tuning, GRPO or KD distillation); requires the training extra |
| `brainforge train evaluate` | student evaluation: loss + perplexity on a split (training extra); `--base` evaluates the untouched base model instead of an adapter (full precision, no quantization: the reference point) |
| `brainforge train chat` | interactive chat with a trained student (training extra) |
| `brainforge train task-eval` | task-level evaluation (verdict/CWE accuracy for the security domain; verdict/flag accuracy for the CTF domain); `--endpoint URL --model-name M` scores a model served behind an OpenAI-compatible endpoint (e.g. llama-server) instead of loading weights locally |
| `brainforge train export` | export a trained student: merged HF weights or quantized GGUF |

`train check` options:

| Option | Default | Description |
|---|---|---|
| `--model/-m` | config base model | Hugging Face id or local model directory |
| `--strategy` | config | `qlora` or `full` |
| `--quantization` | config | `auto`, `4bit`, `8bit` or `none` |
| `--backend` | config | `auto`, `trl` or `unsloth` |
| `--device` | config | `auto`, `cuda`, `xpu`, `mps` or `cpu` |
| `--max-length` | config | override sequence length for the memory estimate |
| `--batch-size` | config | override training batch size |
| `--lora-rank` | config | override LoRA rank |
| `--dataset-dir` | none | optional prepared split to validate |
| `--json` | off | emit a machine-readable JSON report |
| `--strict` | off | treat warnings as failures |
| `--config/-c` | `config/config.json` | configuration file |

Exits 0 when nothing fails and no warning in `--strict` mode, 1 otherwise.
It never downloads model weights: only `config.json`, tokenizer files and the
chat template are fetched. Use `brainforge train check --json --strict` in CI
before a GPU job.

`train run` options:

| Option | Default | Description |
|---|---|---|
| `--dataset-dir/-d` | `datasets/prepared` | directory with `train.jsonl` (+ optional `validation.jsonl`) |
| `--epochs/-e` | config | override config epochs |
| `--base-model/-b` | config | override config base model |
| `--strategy` | config | `qlora` (4-bit + LoRA), `full` (all weights) or `grpo` (verifiable-reward RL) |
| `--backend` | config | `auto` (trl), `trl` or `unsloth` (needs the `unsloth` extra) |
| `--device` | config | `auto`, `cuda`, `xpu`, `mps` or `cpu` |
| `--quantization` | config | `auto`, `4bit`, `8bit` or `none` |
| `--kd` | off | distillation training; requires `kd.jsonl` in the dataset dir |
| `--output/-o` | `<config.output_dir>/<run>` | output directory |
| `--resume/--no-resume` | off | resume from the latest checkpoint |
| `--config/-c` | `config/config.json` | configuration file |

`train export` options:

| Option | Default | Description |
|---|---|---|
| `--model/-m` | latest run | adapter directory |
| `--output/-o` | `models/export` | export directory |
| `--format/-f` | `merged` | `merged` (HF weights) or `gguf` (quantized GGUF + Modelfile) |
| `--quant` | `Q4_K_M` | GGUF quantization type (`--format gguf`) |
| `--ollama <name>` | none | register the GGUF with Ollama (`--format gguf`) |
| `--llama-cpp-path` | auto | path to a llama.cpp checkout (`--format gguf`) |

## ops

| Command | Description |
|---|---|
| `brainforge ops status` | summarize `datasets/*.state.json` run states (done/total, current case, age, STALE past `--stale-after`) plus optional log tails (`--logs`) |
| `brainforge ops log <path>` | tail the last N non-empty lines (`--lines`, default 12) with an encoding fallback chain (utf-8, utf-16-le, cp1252); error lines are highlighted |

## quickstart

```bash
brainforge quickstart run --domain ticket_triage --input data/raw --provider mock
```

| Option | Default | Description |
|---|---|---|
| `--domain/-D` | required | domain pack; the first configured pipeline for that domain is used |
| `--input/-i` | `data/raw` | cases directory |
| `--provider` | none | provider override (e.g. `mock`) |
| `--base-model/-b` | config | override config base model |
| `--epochs/-e` | 1 | training epochs |
| `--strategy` | `qlora` | `qlora`, `full` or `grpo` |
| `--backend` | `auto` | `auto` (trl), `trl` or `unsloth` |
| `--device` | `auto` | `auto`, `cuda`, `xpu`, `mps` or `cpu` |
| `--quantization` | `auto` | `auto`, `4bit`, `8bit` or `none` |
| `--kd` | off | use the `kd.jsonl` sidecar if present |
| `--output/-o` | `datasets/<domain>.jsonl` | dataset output path |
| `--config/-c` | `config/config.json` | configuration file |

Chains `pipeline run` -> `train prepare` -> `train run` -> `train
evaluate`, then prints next steps. The train step requires the training
extra; with `--provider mock` the whole chain is a zero-cost dry
run that stops cleanly at the training gate.

## abliterate

```bash
brainforge abliterate run --model Qwen/Qwen3-8B \
  --calibration calibration.jsonl --output models/abliterated
```

| Option | Default | Description |
|---|---|---|
| `--model/-m` | required | HF model id or local directory |
| `--calibration` | required | JSONL of `{prompt, label}` with `label` in `harmful`/`benign` |
| `--output/-o` | `models/abliterated` | abliterated model directory |
| `--layers` | `all` | `all` or comma-separated layer indices |

Experimental. Removes a single refusal direction (mean hidden-state
difference between harmful and benign calibration prompts) from attention
output and MLP down projections; writes `ablation_stats.json` with
direction norms before/after and the suppression ratio. Quality is a
heuristic, not a guarantee. See the [KD/abliteration notes](../explanation/knowledge-distillation.md).
