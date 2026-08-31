# CLI reference

Global options: `--config/-c` selects the configuration file (default
`config/config.json`, overridable with `BRAINFORCE_CONFIG`). Output uses
rich tables; failures exit 1 with a reason on stderr.

## config

| Command | Description |
|---|---|
| `brainforge config validate` | validate references, judge independence, dry-build providers |
| `brainforge config providers` | table: name, type, base URL, API style, key set |
| `brainforge config schema --write` | regenerate `config/schema.json` from the models |
| `brainforge config schema --check` | fail when the schema is out of sync |
| `brainforge config path` | print the resolved config path |

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
  --experiment first-run
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
| `--config/-c` | `config/config.json` | configuration file |

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

## train

| Command | Description |
|---|---|
| `brainforge train prepare <file.jsonl> [-o dir]` | validate, split, export for TRL (+ `stats.json`) |
| `brainforge train run` | QLoRA training (phase 2 stub, explicit error) |
| `brainforge train evaluate` | student evaluation (phase 2 stub) |
| `brainforge train export` | model export (phase 2 stub) |

`train run` reads the `training` config section; see the
[configuration reference](configuration.md).
