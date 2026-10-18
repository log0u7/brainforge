# Abliteration: single-direction refusal removal, experimental

* Status: accepted
* Date: 2026-10-18

## Context and Problem Statement

Red-team and CTF agents need models whose refusal direction has been
removed from the weights. The technique (Arditi et al. 2024): compute a
refusal direction as the mean hidden-state difference between harmful and
benign prompts, then project weight matrices orthogonal to it.

## Decision Drivers

* The user wants a CTF agent; abliteration is part of that pipeline.
* Full implementations (Heretic-style iterative optimization) are
  research-grade and out of scope for a first cut.
* The operation is pure weight surgery: CPU-capable, no training loop.

## Considered Options

* Integrate an existing abliteration library.
* Implement single-direction ablation in-house.
* Defer abliteration entirely.

## Decision Outcome

Chosen option: "in-house single-direction ablation, marked experimental",
because it is ~150 lines with no new dependency, runs on CPU, and the
limitations are documented (fixed 2/3-depth target layer,
mean-over-tokens direction) with a clear upgrade path.

### Consequences

* Positive: no dependency, works offline, stats prove the effect
  (direction norm before/after, suppression ratio).
* Negative: quality is a heuristic; some models need per-model tuning.
* Mitigation: `ablation_stats.json` makes the effect measurable; the docs
  mark it experimental and tell users to evaluate behavior themselves.

### Confirmation

* `tests/integration/test_abliterate_chain.py` (opt-in via
  `BRAINFORCE_IT=1`): on a tiny locally built model, the refusal direction
  shrinks below half its norm, the model round-trips, and layer subsets
  edit only the selected layers.
* `tests/unit/test_abliterate.py`: calibration validation, layer
  selection, CLI wiring.

## Pros and Cons of the Options

### Existing library

* Good, because battle-tested.
* Bad, because a heavy dependency for one linear-algebra operation.

### In-house single-direction

* Good, because small, testable, dependency-free.
* Bad, because naive; not a quality guarantee.

### Defer

* Good, because less surface.
* Bad, because the CTF use case needs it now.

## Links

* [Knowledge distillation / abliteration notes](../explanation/knowledge-distillation.md)
