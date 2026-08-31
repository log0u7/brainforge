# 0003 - Contamination controls and the post-cutoff holdout

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

Commercial teacher LLMs have almost certainly memorized public CVEs and their
patches. When a teacher analyzes a 2019 OpenSSL vulnerability, it may recite
the advisory instead of analyzing the code. A student fine-tuned on such
examples learns to recite, and benchmarks on memorized cases will report gains
that evaporate on fresh vulnerabilities.

## Decision Drivers

* The student must learn analysis, not recall of public advisories.
* Source data spans different eras; the pipeline must know what the teachers
  could have memorized.
* Benchmark credibility requires evaluation data the teachers cannot recite.
* Pre-cutoff cases still have value (more data); discarding them silently
  would be wasteful.

## Considered Options

* Ignore contamination
* Reject every pre-cutoff case by default
* Tag `recitation_risk`, keep data, and evaluate primarily on a post-cutoff
  holdout
* Rely on private data only

## Decision Outcome

Chosen option: "Tag `recitation_risk`, keep data, and evaluate primarily on a
post-cutoff holdout".

Mechanisms:

1. Every teacher model declares `knowledge_cutoff` (`YYYY-MM`) in `models`.
2. Every case carries a source `date` (git commit date, CVE publish date);
   missing dates are conservative (`unknown_source_date`).
3. Records predating the latest teacher cutoff are tagged `recitation_risk`
   with the reason; tagging never discards silently
   (`reject_recitation_risk: true` opts into exclusion at build time).
4. `brainforge train prepare` / `dataset split` emit `test_postcutoff.jsonl`:
   cases strictly after the latest cutoff. This is the primary evaluation
   signal, with a warning below 20 cases.
5. Experiment reports show contamination ratios so base-vs-student comparisons
   stay honest.

### Consequences

* The burden moves to case sourcing: post-cutoff material is scarcer, which is
  visible (and audible) via the warning.
* Teacher cutoffs must be maintained; a missing cutoff makes *all* records
  risky, which surfaces the gap instead of hiding it.
* Benchmark numbers become comparable across runs: the post-cutoff split is
  stable and interpretable.

### Confirmation

* `tests/unit/test_pipeline_dataset.py` covers tagging (`pre_cutoff_source`,
  `unknown_source_date`, `unknown_teacher_cutoff`) and the holdout split.
* `brainforge dataset inspect` displays contamination ratios.
* Experiment `report.md` includes a contamination note section.

## Pros and Cons of the Options

### Ignore contamination

* Good, because more data is instantly available.
* Bad, because benchmark gains on memorized cases are fake and the student
  recites instead of analyzing.

### Reject pre-cutoff by default

* Good, because the dataset would be contamination-free.
* Bad, because it silently discards most real-world material and hides the
  loss; tagging plus optional exclusion keeps the choice explicit.

### Tag + post-cutoff holdout (chosen)

* Good, because data is preserved, risk is visible per record, and evaluation
  has a defensible baseline.
* Bad, because post-cutoff cases are initially rare; the warning makes the
  gap actionable.

### Private data only

* Good, because teachers cannot have memorized it.
* Bad, because acquiring enough private vulnerability data is impractical at
  MVP scale; it is the golden-dataset complement, not the pipeline's diet.

## Links

* [Contamination explanation](../explanation/contamination.md)
* ADR-0002 for teacher trust, ADR-0007 for the record format
* `src/brainforge/pipeline/engine.py` (`_recitation_risk`),
  `src/brainforge/dataset/split.py`
