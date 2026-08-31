# How to manage datasets

This guide covers the care and feeding of a generated dataset: validation,
inspection, deduplication-aware statistics and splits. It assumes a dataset
exists, e.g. `datasets/security_dataset.jsonl` from
[generate a dataset](generate-dataset.md).

## 1. Validate

```bash
uv run brainforge dataset validate datasets/security_dataset.jsonl
```

Expected: `dataset is valid (N records)`. On failure, the first ten invalid
records are listed with index, id and reasons, and the exit code is 1 (CI
blocks). Validation rules are in the
[dataset format reference](../reference/dataset-format.md).

## 2. Inspect

```bash
uv run brainforge dataset inspect datasets/security_dataset.jsonl
```

Expected: a table with:

- **records**: total count;
- **recitation_risk**: share of pre-cutoff or undatable cases
  ([contamination](../explanation/contamination.md));
- **post-cutoff (primary eval signal)**: the records teachers cannot recite;
- **agreement `<role>`**: teacher/judge agreement rate; rates >= 95% are
  flagged in red as correlated-bias suspects
  ([judge correlation](../explanation/judge-correlation.md));
- **duplicates**: near-duplicate count detected by the same engine the split
  step uses.

Read this table before any benchmark claim.

## 3. Split

```bash
uv run brainforge dataset split datasets/security_dataset.jsonl --output datasets/security_split
```

Expected: `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`test_postcutoff.jsonl`, `stats.json`. Splits are source-grouped and seeded
(one repository never spans two splits), and the post-cutoff holdout is
exported alongside with a warning when it holds fewer than 20 cases.

Options: `--train-ratio` (0.8), `--val-ratio` (0.1), `--seed` (42).

## 4. Iterate consciously

- Fix rejected causes, then re-run the pipeline: the cache replays completed
  steps for free.
- Add post-cutoff cases when the warning fires; the holdout is the benchmark.
- Re-run `inspect` after every pipeline change; agreement-rate drift is your
  early warning that a model swap changed ensemble behavior.

## 5. Prepare for training

```bash
uv run brainforge train prepare datasets/security_dataset.jsonl --output datasets/prepared
```

Same validation and splits as `dataset split`, plus stats ready for the TRL
loop. Details in the [training reference](../reference/training.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `dataset not found or empty` | wrong path | check the pipeline output path |
| validation fails on `assistant content is not valid JSON` | old records from a previous schema version | regenerate the affected records (`--fresh`) |
| `only N post-cutoff evaluation cases` warning | scarce post-cutoff sources | source recent CVEs/mutations; see [contamination](../explanation/contamination.md) |
| high duplicate count | homogeneous input data | diversify sources; dedup runs at split time |

## Next steps

- [Training reference](../reference/training.md) for the phase-2 loop.
- [ADR-0003](../adr/0003-contamination-controls-postcutoff-holdout.md) for the
  split design.
