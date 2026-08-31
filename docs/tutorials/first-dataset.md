# Tutorial: build your first dataset

In this tutorial you run the complete BrainForge loop, from raw cases to a
validated, split training dataset, without spending a cent on API calls (the
mock provider plays every role). By the end you will have inspected a real
`dataset.jsonl` with contamination stats and teacher/judge agreement, and
exported TRL-ready splits.

No API keys and no GPU are needed. Estimated time: 10 minutes.

## Prerequisites

- Python 3.12+ and [mise](https://mise.jdx.dev) (or any uv 0.12.x)
- The BrainForge repository cloned

## 1. Install

```bash
cd brainforge
mise install
uv sync
```

**Checkpoint**: `uv run brainforge --help` prints the command groups
(`config`, `models`, `roles`, `teacher`, `pipeline`, `rag`, `dataset`,
`train`).

## 2. Validate the configuration

```bash
uv run brainforge config validate
uv run brainforge config schema --check
```

**Checkpoint**: both commands answer green (`configuration is valid`,
`schema.json is in sync`). The shipped `config/config.json` declares four
providers (OpenRouter, Zen, MLGW, mock), four models with knowledge cutoffs,
and judge independence is already satisfied.

## 3. Look at the cases

```bash
ls examples/cases/
cat examples/cases/case-cmd-injection-001.json
```

Three tiny vulnerabilities: a command injection (dated 2026-08, *after* the
teachers' cutoffs), an SQL injection (2026-04, *before*), and a stored XSS
(2026-07, after). Source dates are what powers contamination tagging; the
README in `examples/cases/` explains the format.

## 4. Run the pipeline

```bash
uv run brainforge pipeline run security_dataset --input examples/cases --provider mock
```

**Checkpoint**: output like

```text
security_dataset: 3 accepted, 0 rejected (3 cases) -> datasets/security_dataset.jsonl
```

The mock provider generated schema-valid teacher analyses and a judge verdict
for each case; the quality gate accepted all three. No API was called.

## 5. Inspect the dataset

```bash
uv run brainforge dataset inspect datasets/security_dataset.jsonl
```

**Checkpoint**: a table showing 3 records, `recitation_risk` around 33% (the
SQL injection case predates the cutoffs), the post-cutoff share, and
agreement rates near 100% flagged in red. That red flag is the
[judge correlation](../explanation/judge-correlation.md) safeguard working:
the mock ensemble agrees too much, exactly what the warning exists to catch
with real teachers.

## 6. Look inside a record

```bash
head -1 datasets/security_dataset.jsonl | python3 -m json.tool
```

Notice the shape ([dataset format](../reference/dataset-format.md)):

- `messages`: the user case and the assistant's JSON verdict, what a student
  will train on;
- `metadata`: teachers, judge, source date, `recitation_risk`, RAG provenance,
  gate result. Training never sees this; auditing never does without it.

## 7. Validate and split

```bash
uv run brainforge dataset validate datasets/security_dataset.jsonl
uv run brainforge train prepare datasets/security_dataset.jsonl
```

**Checkpoint**: validation is green, and `datasets/security_dataset_prepared/`
contains `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`test_postcutoff.jsonl` and `stats.json`. A warning about the tiny post-cutoff
holdout is expected with three cases; it tells you the same thing it will tell
you at scale: source more recent material.

## 8. Try the RAG (optional)

```bash
uv run brainforge rag index examples --backend hashing
uv run brainforge rag search "subprocess shell=True" --backend hashing
```

The hashing backend needs no model download. Re-running the pipeline now
injects retrieved chunks into teacher prompts and copies their provenance into
the records.

## Recap

You validated a configuration, generated a three-case dataset with a mock
ensemble, read its contamination and agreement stats, inspected the record
anatomy, and exported training splits. The loop you just ran is the exact
loop you will run with real teachers.

## Where to go next

- Connect real providers:
  [how-to: configure providers](../how-to/configure-providers.md)
- Run for real, with gates, rejected cases and experiment reports:
  [how-to: generate a dataset](../how-to/generate-dataset.md)
- Why the red agreement flag matters:
  [judge correlation](../explanation/judge-correlation.md)
- Why source dates matter:
  [contamination](../explanation/contamination.md)
