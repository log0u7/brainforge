# Final benchmark contract: unseen challenges, oracle-verified runs, contamination audit

* Status: proposed
* Date: 2026-10-30

## Context and Problem Statement

The MR4 benchmark orchestrator (ADR-0020) compares distillation arms on the
frozen post-cutoff holdout. That comparison answers "which arm distills
better", but it does not answer the question the whole program exists for:
**does the trained student solve real targets better than the base model?**
Every task_eval and gate dataset record is drawn from the same challenge
corpus the student was trained on (or split from), so the number is a
memorization measurement, not a capability measurement.

A final benchmark needs a contract that a reader of the blog post can trust:
the evaluation set is unseen, the verification reads the live service output
through the lictor lab loop, the reported success rate is stable across
repetitions, and the contamination audit is explicit.

## Decision Drivers

* The model recitation lesson (2026-10-28): models trained on FLAG{...}
  strings recite them verbatim. Any verification that string-matches a
  training flag is invalid. The pilot already carries per-target
  `expected_flag` and passes `re.escape(expected_flag)` as the run oracle
  (`src/brainforge/experiments/pilot.py:338-340`).
* The pilot's lictor lab loop is the only path where the student actually
  operates a target: facts, decide, run, harvest
  (`src/brainforge/experiments/pilot.py:225-262`).
* The post-cutoff machinery already exists
  (`split_with_postcutoff`, `manager.py:37-38`) but is applied to dataset
  records, not to the challenge corpus itself.
* No repetition mechanism exists anywhere in the codebase (investigation
  2026-10-30): every evaluation is a single pass.
* The GPU (RTX 3080, 10 GB) cannot run teacher and student concurrently
  (ADR-0020); the benchmark must run arms sequentially anyway, so arms X
  repetitions is bounded work.
* The blog post is the deliverable: numbers must be reproducible and
  auditable, not impressive.

## Considered Options

* **A. Reuse task_eval on a held-out challenge split** (endpoint or local
  weights, single pass, flag equality scoring).
* **B. Extend the pilot loop into a benchmark harness**: unseen challenge
  list, per-target oracle regex on live output, N repetitions per challenge,
  contamination audit comparing training IDs with eval IDs.
* **C. Build a new standalone evaluator** outside the pilot, calling lictor
  directly.

## Decision Outcome

Chosen option: **"B. Extend the pilot loop into a benchmark harness"**, because
the pilot already implements everything a fair evaluation needs (oracle regex,
retry semantics, trajectory records) and option A repeats the memorization
error (flag equality on a derived split), while option C duplicates the lab
loop.

### Consequences

* Positive: one evaluation path, live-service verification by construction,
  repetition-aware success rate, contamination audit as a first-class output.
* Negative: the benchmark depends on the lictor host (gpu3080) and cannot run
  in CI; the harness is slower (repetitions multiply run count).
* Neutral: results land as JSON + markdown next to the pilot artifacts, same
  shape as MR4 (`results.json`, `results.md`).

## Pros and Cons of the Options

### Option A (task_eval on held-out split)

* Good: cheap, runs in CI, reuses MR4 plumbing.
* Bad: flag-equality scoring against records from the same corpus the student
  trained on; a reciting model scores well. Contradicts the 2026-10-28
  lesson.

### Option B (pilot-loop harness) - chosen

* Good: verification reads live service output; per-target oracle regex
  prevents recitation scoring; repetitions give a real success rate.
* Bad: host-bound, slow, requires an unseen challenge set to exist first.
* Neutral: the harness is a new module (`brainforge.experiments.bench_lab`)
  orchestrating the existing pilot primitives, not a new loop.

### Option C (standalone evaluator)

* Good: no dependency on pilot's retry/seed logic.
* Bad: reimplements facts/decide/run/harvest; drift risk with the pilot.

## Links

* ADR-0020: GKD + consolidation cycle (MR4 orchestrator this contract
  evaluates).
* ADR-0003: contamination controls, post-cutoff holdout.
* Lesson 2026-10-28: models recite FLAG{...}; never string-match training
  flags.
