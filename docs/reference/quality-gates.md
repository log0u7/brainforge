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
A record carrying `fallback: true` is the one exception: it must carry
`fallback_reason` instead of a passing gate (see
[fallback records](dataset-format.md#fallback-records-teaching-abstention)).
See [dataset format](dataset-format.md).

## Fallback records (opt-in)

With `pipeline.fallback_examples: true`, a rejected case is not dropped: it
becomes an abstention training record whose assistant target is a canonical
`insufficient_information` verdict. The rejected-case audit trail
(`data/rejected/`) is kept either way. A provider failure is not case
unreliability, so it never produces a fallback record; the case is skipped with
a `provider_error` reason and the batch continues.

## Provider failures

A provider or parse failure fails loudly inside the provider layer (two parse
retries, then `ProviderError`), but the engine isolates it per case: the case
is recorded with a `provider_error` reason and the remaining cases still run.
Run summaries report skipped cases in an `errors` counter.

## Escalation (critic retry)

With `pipeline.retry_with_critic: "<critic role>"`, a rejected first pass is
not final: the engine runs that critic role, then the judge again with the
critique in context, and re-applies the gate. The critic never sees the first
judge verdict (only teacher outputs), so it is not anchored by the rejection.
The role must exist and resolve to a critic kind (a name containing "critic"
or "general" does, per `infer_role_kind`); otherwise engine construction
fails. Records produced after a retry carry `metadata.critic_retry: true` and
`metadata.first_pass_reasons`. If the retry also fails, the case is rejected
(or becomes a fallback record when `fallback_examples` is on).

## Dispute arbitration by ground truth

Two experts disagreeing does not mean the example is lost: one of the two
is right, and the ground truth tells which. On verified-tier cases (the
case carries an `expected_flag`), a judge `insufficient_information`
verdict no longer routes straight to rejection when
`reject_unresolved` is set: if the correct flag appears in the teacher
outputs or in the judge's own reasoning, the case is accepted with the
reason `arbitrated_by_ground_truth` and the winning answer feeds
training. When nobody's flag matches, the strict abstention path applies.
Cases without a known flag (tier-B) keep the abstention gate unchanged.

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
