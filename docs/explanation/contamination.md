# Contamination: recitation versus analysis

Why BrainForge tracks what its teachers could have memorized, and how that
shapes the datasets and the benchmarks.

## The problem

Commercial teacher LLMs are trained on public data, which includes the CVE
database, advisories, security blog posts and the patches themselves. When a
teacher analyzes a well-known vulnerability, two very different behaviors
produce similar-looking answers:

- **Analysis**: reading the provided code, tracing the data flow, and
  concluding that user input reaches `subprocess.run(..., shell=True)`.
- **Recitation**: recognizing a memorized pattern ("this looks like
  CVE-2024-XXXX") and reproducing the advisory's conclusion.

The output looks the same. The difference shows up only where recitation
stops working: on vulnerabilities published after the teacher's training
cutoff, on mutated code, or on private codebases.

If a student is fine-tuned on recitations, it learns to *name* known
vulnerabilities, not to *find* unknown ones. Worse, benchmarking on the same
memorized corpus reports impressive numbers that measure recall of public
advisories, not vulnerability analysis.

## The mechanisms

BrainForge does not try to make teachers forget. It measures and controls
exposure:

1. **Declared cutoffs**. Every teacher model declares `knowledge_cutoff`
   (`YYYY-MM`) in the configuration. A missing cutoff is treated as "cutoff
   unknown", which makes everything risky rather than everything safe.
2. **Mandatory source dates**. Every case carries a `source.date` (git commit
   date, CVE publish date, advisory date). For plain files, BrainForge queries
   `git log` automatically. A missing date is conservative:
   `unknown_source_date`.
3. **Tagging, not deletion**. A record whose source predates the latest
   teacher cutoff gets `recitation_risk: true` with a machine-readable reason
   (`pre_cutoff_source`, `unknown_source_date`, `unknown_teacher_cutoff`).
   Nothing is silently discarded; `reject_recitation_risk: true` opts into
   exclusion at build time.
4. **The post-cutoff holdout**. `train prepare` and `dataset split` emit
   `test_postcutoff.jsonl`: cases strictly after the latest teacher cutoff.
   These are the cases the teachers *cannot* recite, so they are the primary
   benchmark signal. A warning fires below 20 cases, because a holdout that
   small cannot support meaningful claims.
5. **Visible ratios**. `brainforge dataset inspect` and every experiment
   report show contamination percentages, so "the student improved by 12
   points" can always be read next to "on what kind of cases".

## How to source for this

- Prefer **recent** CVEs and patches (after your teachers' cutoffs) for
  evaluation-oriented cases.
- Use **private or mutated code** when possible: renaming identifiers,
  changing structure and re-hosting the vulnerability in a new context
  degrades memorized associations.
- Keep pre-cutoff cases: they still teach patterns, and the tag lets
  downstream tooling weight them consciously.
- Build the golden dataset (50-200 human-verified cases) with a healthy
  post-cutoff share.

## What "success" looks like

The base model and the fine-tuned student are compared on identical prompts,
with the post-cutoff split as the headline metric. Gains concentrated in the
pre-cutoff split with flat post-cutoff numbers indicate recitation transfer;
gains on post-cutoff cases are the signal the whole pipeline exists to
produce.

## Related

- [ADR-0003](../adr/0003-contamination-controls-postcutoff-holdout.md): the
  decision record with rejected alternatives.
- [Dataset format reference](../reference/dataset-format.md): where
  `recitation_risk` lives in a record.
- [How-to: manage datasets](../how-to/manage-datasets.md): the commands to
  inspect and split.
