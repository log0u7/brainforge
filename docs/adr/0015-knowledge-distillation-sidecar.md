# Knowledge distillation: teacher top-k sidecar plus custom loss, experimental

* Status: accepted
* Date: 2026-10-18

## Context and Problem Statement

Dataset-level distillation (teachers + judge -> SFT) throws away the
teacher's distribution: the student only sees the sampled answer. When the
teacher is served locally (Unsloth/Ollama) and shares a vocabulary with the
student, the teacher's top-k logprobs per token are a stronger signal,
especially for ambiguous positions where the gold answer is arbitrary
among several good tokens.

## Decision Drivers

* The OpenAI-compatible chat completions API already serves logprobs
  (`logprobs=true, top_logprobs=k`); Ollama serves them too.
* The student and teacher may use different tokenizers, so per-token KD
  needs an alignment strategy.
* Storage matters: per-token top-k doubles dataset size.

## Considered Options

* Only dataset-level distillation (status quo).
* Full online KD (teacher served during training).
* Offline KD: teacher logprobs recorded at generation time into a sidecar,
  consumed by a custom loss at training time.

## Decision Outcome

Chosen option: "offline KD with a sidecar", because generation and
training are already separate phases, the teacher need not run during
training, and the sidecar is inspectable.

Design points:

* The dataset record stores the teacher's exact output (not a
  re-serialization) so sidecar positions align with the student's gold
  tokens.
* Teacher positions map onto student tokens by character midpoint;
  cross-tokenizer is safe, same-tokenizer is exact.
* Top-k tokens absent from the student vocabulary are dropped with the
  teacher distribution renormalized.
* Loss: `alpha * T^2 * CE(student_T, teacher top-k) + (1 - alpha) * CE(gold)`.
* Packing is rejected (alignment needs positions).

### Consequences

* Positive: exact same-vocab distillation; graceful cross-vocab
  degradation instead of failure.
* Negative: marked experimental; no quality benchmark yet; storage cost.
* Mitigation: `kd_topk` default 8; docs state the limits and tell users to
  compare `train task-eval` against the non-KD run.

### Confirmation

* `tests/unit/test_kd.py`: sidecar generation only in `--kd` mode, mock
  positions rebuild the content, alignment math, config validation, the
  missing-sidecar error.
* `tests/integration/test_kd_chain.py` (opt-in): real-tokenizer alignment,
  loss ordering (matching distribution < contradicting), and the CLI chain
  with the mock provider.
* `training/kd.py` docstring documents the alignment and renormalization
  rules.

## Pros and Cons of the Options

### Status quo (dataset-level only)

* Good, because simple and proven.
* Bad, because it discards the teacher's distribution.

### Online KD

* Good, because no storage.
* Bad, because the teacher must run during training; ties training to a
  serving stack.

### Offline KD with sidecar

* Good, because inspectable, decoupled, composable with all backends.
* Bad, because storage and the alignment heuristic.

## Links

* [Knowledge distillation](../explanation/knowledge-distillation.md)
* [Dataset format: KD sidecar](../reference/dataset-format.md)
