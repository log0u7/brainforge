# Tutorial: build your first student model

In this tutorial you run the complete BrainForge loop on a non-security
domain (IT ticket triage): from one raw case to a distilled dataset, then to
a trained student on a CUDA GPU. The first half uses the mock provider, so it
needs no API keys and no GPU. The second half needs real providers and a GPU.

**What "verified" means here:** schema-valid, gate-checked, and
provenance-complete. It does not mean human-reviewed ground truth, and the
mock labels are not evidence of model quality.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (`mise` is optional)
- The BrainForge repository cloned
- For training: a CUDA GPU and `uv sync --extra training`

## 1. Install

```bash
cd brainforge
uv sync
```

**Checkpoint:** `uv run brainforge --help` prints the command groups
(`config`, `models`, `roles`, `teacher`, `pipeline`, `rag`, `dataset`,
`train`, `quickstart`, `abliterate`).

## 2. Validate the configuration

```bash
uv run brainforge config validate
```

**Checkpoint:** `configuration is valid`. The shipped config includes the
`ticket_triage` pipeline used below.

## 3. Run the pipeline (mock provider)

```bash
uv run brainforge pipeline run ticket_triage \
  --input examples/cases/ticket-password-reset-001.json \
  --provider mock --no-rag
```

The mock provider fills each role's JSON Schema deterministically, so the
teacher, judge, and quality-gate path runs offline. It proves the plumbing,
not the labels. Three ticket examples ship in `examples/cases/` (password
reset, VPN outage, invoice question); this tutorial uses one to keep the
output small.

**Checkpoint:**

```text
ticket_triage: 1 accepted, 0 rejected (1 cases) -> datasets/ticket_triage.jsonl
```

## 4. Inspect the dataset

```bash
uv run brainforge dataset inspect datasets/ticket_triage.jsonl
```

**Checkpoint:** the CLI reports one record, `recitation_risk 1 (100.0%)`, no
post-cutoff cases, and no duplicates. It does not print per-record verdicts;
read the JSONL for those.

**Checkpoint:**

```text
records         1
recitation_risk 1 (100.0%)
post-cutoff     0 (0.0%)
duplicates      0
```

The `general_teacher` role in the ticket standard preset has no knowledge
cutoff, so the record is conservatively tagged `recitation_risk`. The
post-cutoff split is empty, so this run says nothing about generalization.
With the mock provider, all verdicts are mock values.

## 5. Export training splits

```bash
uv run brainforge train prepare datasets/ticket_triage.jsonl \
  --output datasets/prepared-ticket
```

**Checkpoint:** `datasets/prepared-ticket/` contains `train.jsonl`,
`validation.jsonl`, `test.jsonl`, `test_postcutoff.jsonl` and `stats.json`.

```text
prepared dataset -> datasets/prepared-ticket
  train: 1
  validation: 0
  test: 0
  test_postcutoff: 0
  warning: only 0 post-cutoff evaluation cases available (minimum recommended:
20); add more recent sources
```

`test_postcutoff.jsonl` can be empty. The warning means the dataset has too
few cases the teachers could not have memorized; add more recent sources with
declared dates before trusting evaluation numbers.

## 6. Train for real (GPU)

```bash
uv sync --extra training
uv run brainforge quickstart run --domain ticket_triage --input data/raw
```

Configure the providers first (OpenRouter, MLGW, and OpenCode Zen for the
ticket preset); see
[configure providers](../how-to/configure-providers.md). Create `data/raw` and
put real cases there.

`quickstart` chains pipeline, prepare, training, and best-effort evaluation.
**Checkpoint:** a run directory under `experiments/` with the adapter and
`train_summary.json`, then an `evaluated` line when the post-cutoff holdout is
non-empty. A mock run on a GPU still produces mock labels, so use real
providers for a useful student.

## 7. Ship it (optional)

```bash
uv run brainforge train export --model experiments/quickstart-... \
  --format gguf --ollama triage-bot
ollama run triage-bot
```

GGUF export needs a local `llama.cpp` checkout. Distribute the GGUF with
llmp2p (`llmp2p import --ollama models/export`, then `llmp2p seed`). See
[ship to Ollama](../how-to/ship-to-ollama.md).

## What you learned

- A domain pack (here `ticket_triage`) defines teacher and judge schemas,
  prompts, and quality gates; the engine runs any domain the same way.
- The mock provider makes the verification path free and offline, but it
  cannot tell you whether the labels are good.
- `quickstart` chains everything; a first cheap pass is not an evaluation.
  Use the post-cutoff holdout and your own review.
- Judge independence, contamination tagging, and provenance are shared
  machinery, not security-only features. They are also not a guarantee of
  accuracy.

Next: [configure real providers](../how-to/configure-providers.md) and
[knowledge distillation](../explanation/knowledge-distillation.md).
