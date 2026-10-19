# Graceful failure: fallback examples in SFT and per-case provider isolation

* Status: accepted
* Date: 2026-10-20

## Context and Problem Statement

Reinforcement learning rewards guessing: when success is the only way to earn
reward, a safe fallback is always worth less than a long-shot attempt
([arXiv:2610.20401](https://arxiv.org/abs/2610.20401), GracefulRL). The same
incentive exists in this pipeline's output: a case the ensemble cannot resolve
reliably is currently dropped into `data/rejected/`, and the student never
learns what to do when acting would be unreliable. A student trained only on
confident answers answers every question.

BrainForge has no reinforcement learning loop (SFT and optional logit KD only),
and adding one is out of scale for the project.

## Decision Drivers

* The pipeline already computes the outcome-level reliability signal the paper
  says should drive fallback: quality gate reasons, teacher disagreement,
  confidence below threshold, and the `insufficient_information` verdict.
  No new annotation is needed.
* Every domain pack's verdict literal already accepts
  `insufficient_information`, so an abstention answer is in-schema everywhere.
* A provider failure today aborts the entire batch run
  (`ProviderError` propagates out of `run_pipeline`), so one unreachable
  endpoint loses every remaining case.

## Considered Options

* Only dataset-level SFT with rejected cases dropped (status quo).
* Full GracefulRL: an RL loop with reward reshaping for graceful failure.
* SFT fallback examples plus per-case provider isolation (chosen).

## Decision Outcome

Chosen option: "SFT fallback examples plus per-case provider isolation",
because it reuses signals the pipeline already produces, needs no new training
infrastructure, and degrades gracefully at both levels.

Design points:

* `PipelineDef.fallback_examples` (default `false`, additive) turns rejected
  cases into abstention records instead of dropping them.
* A fallback record's assistant target is a canonical
  `{"verdict": "insufficient_information", "reasoning": <gate reasons>,
  "confidence": 0.0}`; metadata carries `fallback: true` and `fallback_reason`.
* Dataset validation accepts a fallback record without a passing quality gate
  (it requires `fallback_reason` instead); every other record still requires
  `quality.passed`.
* The rejected-case audit trail (`data/rejected/`) is kept unchanged.
* Provider errors are isolated per case: the case is recorded with a
  `provider_error` reason, the batch continues, and no fallback record is
  fabricated (an infrastructure failure is not case unreliability).
* `train task-eval` gains abstention as a third outcome class, with
  `abstain_rate` and `abstain_recall` (of the records the judge could not
  resolve, how many the model refused too).

### Consequences

* Positive: the student learns when not to answer, measured by a metric that
  matches the behavior; generation runs survive flaky providers.
* Negative: SFT teaches a fixed fallback policy; it does not learn the
  paper's optimized fallback boundary. A dataset can be skewed toward
  abstention if most cases are rejected (visible via `abstain_rate`).
* Mitigation: the knob is off by default; `dataset inspect` shows the
  fallback share; a real RL path remains a future, separate decision.

### Confirmation

* `tests/unit/test_graceful_failure.py`: knob default, fallback record shape,
  validation with and without `fallback_reason`, engine emission when enabled
  and absence when disabled, `run_pipeline` counts, provider error isolation,
  and no fallback record from a provider failure.
* `tests/unit/test_task_eval.py`: abstention parsing, prediction and scoring
  (abstain recall/rate, FP on a guess over an abstained case, FN on an
  abstention over a confirmed case).

## Pros and Cons of the Options

### Status quo (drop rejected cases)

* Good, because the dataset contains only verified answers.
* Bad, because the student never sees its own limits, and one provider
  failure kills a whole batch.

### Full GracefulRL (RL with reward reshaping)

* Good, because it learns the optimal fallback boundary from outcome reward.
* Bad, because it needs an RL loop, a reward function, and GPU budget the
  project does not have; it is the literal paper, not the right first step.

### SFT fallback examples plus isolation (chosen)

* Good, because it reuses existing signals, is opt-in, and is measurable.
* Bad, because the fallback policy is fixed by construction, not learned.

## Links

* [arXiv:2610.20401 Training Language Models to Fail Gracefully](https://arxiv.org/abs/2610.20401)
* [Quality gates reference](../reference/quality-gates.md)
* [Dataset format: fallback records](../reference/dataset-format.md)
* [References](../reference/references.md)
