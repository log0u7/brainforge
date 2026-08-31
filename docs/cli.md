# CLI reference

Global options: `--config/-c` selects the configuration file (default
`config/config.json`, or `BRAINFORCE_CONFIG`).

## config

| Command | Description |
|---|---|
| `brainforge config validate` | validate references, judge independence, dry-build providers |
| `brainforge config providers` | table of configured providers |
| `brainforge config schema --write` | regenerate `config/schema.json` from the models |
| `brainforge config schema --check` | fail if the schema is out of sync |
| `brainforge config path` | print the resolved config path |

## models

| Command | Description |
|---|---|
| `brainforge models list` | table: name, provider, model id, family, cutoff |
| `brainforge models test <name>` | send a tiny request, print usage and reply |

## roles

| Command | Description |
|---|---|
| `brainforge roles list` | roles with inferred kind, model and temperature |

## teacher

| Command | Description |
|---|---|
| `brainforge teacher run --role <role> --case <id\|path>` | run one role on one case |
| `brainforge teacher ensemble --case <id\|path>` | run teachers + judge, print judge JSON |

Both accept `--cases-dir` (default `data/raw`).

## pipeline

```bash
brainforge pipeline run security_dataset \
  --input examples/cases \
  --provider mock \
  --no-rag \
  --fresh \
  --experiment first-run
```

Options: `--input/-i` (cases directory), `--output/-o` (dataset path, default
`datasets/<pipeline>.jsonl`), `--provider mock` (override every provider),
`--fresh` (bypass cache), `--rag/--no-rag`, `--rag-backend fastembed|hashing`,
`--experiment <name>` (create `experiments/<date>-<name>/` with metrics and
report).

## rag

| Command | Description |
|---|---|
| `brainforge rag index <path> --backend fastembed` | index a directory |
| `brainforge rag search "<query>" --k 5` | search with provenance |

## dataset

| Command | Description |
|---|---|
| `brainforge dataset build <pipeline>` | build via a pipeline (defaults to the mock provider) |
| `brainforge dataset validate <file.jsonl>` | strict record validation |
| `brainforge dataset inspect <file.jsonl>` | contamination stats, agreement rates, duplicates |
| `brainforge dataset split <file.jsonl>` | train/validation/test + post-cutoff holdout |

## train

| Command | Description |
|---|---|
| `brainforge train prepare <file.jsonl>` | validate, split, export for TRL |
| `brainforge train run` | QLoRA training (phase 2 stub) |
| `brainforge train evaluate` | student evaluation (phase 2 stub) |
| `brainforge train export` | model export (phase 2 stub) |
