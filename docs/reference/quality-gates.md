# Quality gates reference

A record reaches `dataset.jsonl` only after every applicable gate passes.
Gates run at two points: the pipeline engine gates the canonical (judge)
verdict per case, and `dataset validate` re-checks the file independently.

## Generic checks (all domains)

Engine, per record:

- the final structured output parsed and validated against the domain schema
  (otherwise the provider layer already retried and failed loudly);
- `confidence >= pipeline.min_confidence`;
- verdict `insufficient_information` is rejected when
  `pipeline.reject_unresolved` is true (default);
- `recitation_risk` records are rejected when
  `pipeline.reject_recitation_risk` is true (default false, tag-only).

Dataset validation, per file: schema-valid record, user/assistant ordering,
JSON assistant content, provenance keys present, `quality.passed` true.
See [dataset format](dataset-format.md).

## Security pack gates

Teacher (`SecurityTeacherAnalysis`):

- `vulnerability_found: true` requires a non-empty `evidence` list;
- `cwe`, when present, must match `^CWE-\d{1,4}$`;
- `reasoning` must be non-empty;
- `confidence` >= `min_confidence`.

Judge (`SecurityJudgeVerdict`):

- `verdict: confirmed` requires a `cwe` and non-empty `evidence`;
- `reasoning` must be non-empty;
- `confidence` >= `min_confidence`.

## Coding pack gates

Teacher (`CodingTeacherAnalysis`):

- `behavior` must be non-empty;
- `reasoning` must be non-empty;
- `confidence` >= `min_confidence`.

Judge (`CodingJudgeVerdict`):

- `verdict: confirmed` requires at least one issue;
- `reasoning` must be non-empty.

## Generic pack gates

Teacher: non-empty `summary`, confidence threshold.
Judge: non-empty `reasoning`, confidence threshold.
The generic pack is the prototyping fallback
([ADR-0006](../adr/0006-modular-domain-packs.md)); serious datasets deserve a
dedicated pack.

## Rejected records

Rejected cases land in `data/rejected/<pipeline>/<case>.json` with the gate
reasons, the judge payload and the full metadata, so near-misses can be
inspected and the causes fixed (prompt, schema, source data) instead of
guessed.

```bash
uv run brainforge dataset validate datasets/security_dataset.jsonl
ls data/rejected/security_dataset/
```
