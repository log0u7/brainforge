# How to generate a dataset

This guide runs a generation pipeline over your cases, with real providers,
and produces a provenance-complete `dataset.jsonl`. It assumes providers are
configured and tested ([how-to: configure providers](configure-providers.md))
and cases exist under `data/raw/` (format in the
[dataset format reference](../reference/dataset-format.md)).

## 1. Preview with the mock provider

Always start free:

```bash
uv run brainforge pipeline run security_dataset --input data/raw --provider mock
```

Expected: `security_dataset: N accepted, M rejected (K cases) ->
datasets/security_dataset.jsonl`. This validates plumbing (cases load, steps
resolve, gates run) without spending tokens.

## 2. Choose the cost mode

Pipelines have a `mode` in config; override per run with `--mode` by editing
`config.json`:

| Mode | Security preset | Use |
|---|---|---|
| `cheap` | security teacher + judge | triage, experiments |
| `standard` | + coding teacher | default production |
| `maximum` | + general critic | hard cases, flagship datasets |

Teacher ablation: pin explicit `steps` in the pipeline config and remove one
teacher to measure its contribution.

## 3. Run for real

```bash
uv run brainforge pipeline run security_dataset --input data/raw
```

What happens per case: RAG context is fetched (if indexed), each teacher runs
with a cached, logged, structured-output call, the judge produces the
canonical verdict, and the quality gate decides accept/reject.

Options: `--fresh` (bypass cache), `--no-rag`, `--output/-o`,
`--experiment <name>`.

## 4. Watch the run

- `data/logs/usage.jsonl`: one line per call (tokens, latency, cache hits,
  cost when pricing is set).
- The console summary counts accepted vs rejected.

## 5. Inspect rejected cases

```bash
ls data/rejected/security_dataset/
cat data/rejected/security_dataset/case-*.json | jq '.gate_reasons'
```

Each file holds the gate reasons, the judge payload and full metadata. Fix
causes (prompt, schema, source data), not symptoms: a spike of
`unresolved_disagreement` usually means contradictory source data or a too-low
`min_confidence`.

## 6. Save an experiment report

```bash
uv run brainforge pipeline run security_dataset --input data/raw --experiment first-run
```

Creates `experiments/<date>-first-run/` with `config.json` (snapshot),
`dataset.jsonl`, `metrics.json` (contamination ratios, teacher/judge agreement
rates) and `report.md` including correlated-bias warnings.

## 7. Keep the cache on

Re-runs after crashes or gate tweaks replay identical teacher calls from the
cache ([ADR-0009](../adr/0009-sqlite-response-cache.md)). Use `--fresh` only
when you actually want regeneration.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no cases found in ...` | wrong `--input` | check the directory contains case files |
| all records rejected `confidence below threshold` | teacher uncertainty | raise data quality or lower `min_confidence` consciously |
| `ProviderUnavailableError` | gateway down | check `models test <name>`; retry (cache resumes) |
| records all `recitation_risk` | missing `knowledge_cutoff` or source dates | fill them in; see [configure providers](configure-providers.md) |

## Next steps

- [Manage the dataset](manage-datasets.md): validate, inspect, split.
- [Quality gates reference](../reference/quality-gates.md) for the rules.
