# Tutorial: from cases to a prepared dataset

In this tutorial you run the dataset path on a non-security domain (IT ticket
triage): from one raw case to a prepared set of training splits. The path uses
the mock provider, so it needs no API keys and no GPU. A final optional section
trains on your hardware (CPU works; the mock labels are not evidence of model
quality).

**What "verified" means here:** schema-valid, gate-checked, and
provenance-complete. It does not mean human-reviewed ground truth, and the mock
labels are not evidence of model quality.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (`mise` is optional)
- The BrainForge repository cloned
- For the optional training section: `uv sync --extra training` (a GPU
  speeds it up, but CPU works for the tiny mock run)

## 1. Install and validate

```bash
cd brainforge
uv sync
uv run brainforge config validate
```

**Checkpoint:** `configuration is valid`. The shipped config includes the
`ticket_triage` pipeline used below.

## 2. Run the pipeline (mock provider)

```bash
uv run brainforge pipeline run ticket_triage \
  --input examples/cases/ticket-password-reset-001.json \
  --provider mock --no-rag \
  --output datasets/ticket_triage.jsonl
```

The mock provider fills each role's JSON Schema deterministically, so the
teacher, critic, judge, and quality-gate path runs offline. It exercises the
plumbing, not the labels. Three ticket examples ship in `examples/cases/`
(password reset, VPN outage, invoice question); this tutorial uses one to keep
the output small.

**Checkpoint:**

```text
ticket_triage: 1 accepted, 0 rejected (1 cases) -> datasets/ticket_triage.jsonl
```

## 3. Inspect the dataset

```bash
uv run brainforge dataset inspect datasets/ticket_triage.jsonl
```

**Checkpoint:**

```text
      Dataset: datasets/ticket_triage.jsonl
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                            ┃ Value      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ records                           │ 1          │
│ recitation_risk                   │ 1 (100.0%) │
│ post-cutoff (primary eval signal) │ 0 (0.0%)   │
│ duplicates                        │ 0          │
└───────────────────────────────────┴────────────┘
```

The CLI calls the post-cutoff count the primary eval signal; it is a holdout
split, not a published benchmark. Here it is empty, so the run says nothing
about generalization.

The `general_teacher` role in the ticket standard has no knowledge cutoff, so
the record is conservatively tagged `recitation_risk`. The record's
`teacher_agreement` is empty because ticket triage does not expose the boolean
verdict signals that the agreement helper understands. With the mock provider,
all verdicts are mock values.

## 4. Prepare training splits

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

`test_postcutoff.jsonl` can be empty. The warning means the dataset has too few
cases the teachers could not have memorized; add more recent sources with
declared dates before trusting evaluation numbers.

## 5. Train for real (optional)

```bash
uv sync --extra training
uv run brainforge train check
uv run brainforge quickstart run --domain ticket_triage --input data/raw
```

Configure providers first. The shipped ticket standard uses an OpenRouter
teacher, an MLGW general critic, and a Zen judge. Configure all three, or pin
`steps` to `ticket_teacher` and `judge` to skip the critic; see
[configure providers](../how-to/configure-providers.md). Create `data/raw` and
put real cases there.

`quickstart` chains pipeline, prepare, training, and best-effort evaluation.
**Checkpoint:** a run directory under `experiments/` with the adapter and
`train_summary.json`, then an `evaluated` line when the post-cutoff holdout is
non-empty. A mock run still produces mock labels, so use real providers for a
useful student. Training, task evaluation, and merged or GGUF export require
the training extra; the device resolves automatically
([hardware reference](../reference/hardware.md)).

## 6. Ship it (optional)

```bash
RUN="$(ls -td experiments/quickstart-* | head -1)"
uv run brainforge train export --model "$RUN" \
  --format gguf --ollama triage-bot
ollama run triage-bot
```

GGUF export needs a local `llama.cpp` checkout. Ollama and llmp2p are optional.
Distribute the GGUF with llmp2p (`llmp2p import --ollama models/export`, then
`llmp2p seed`). See [ship to Ollama](../how-to/ship-to-ollama.md).

## What you learned

- A domain pack (here `ticket_triage`) defines teacher and judge schemas,
  prompts, and quality gates; the engine runs any domain the same way.
- The mock provider makes the verification path free and offline, but it cannot
  tell you whether the labels are good.
- `quickstart` chains everything; a first cheap pass is not an evaluation. Use
  the post-cutoff holdout and your own review.
- Judge independence, contamination tagging, and provenance are shared
  machinery, not security-only features. They are also not a guarantee of
  accuracy.

Next: [configure real providers](../how-to/configure-providers.md) and
[knowledge distillation](../explanation/knowledge-distillation.md).
