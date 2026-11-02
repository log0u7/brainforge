# On-policy distillation (GKD) and the periodic consolidation cycle

* Status: proposed
* Date: 2026-11-02

## Context and Problem Statement

BrainForge has one shipped distillation path: the ADR-0015 sidecar, where an
external teacher records top-k logprobs into `kd.jsonl` and the custom
`KDTrainer` consumes them. The 2026-10-28 benchmark measured the path's
structural limits on the 17-record `kd-glm-full` run: perplexity 20.07 on the
post-cutoff holdout, worse than the 12.61 baseline. The loss curve was bimodal
(declining batches near 11-22 mixed with stuck batches near 50-78), traced to
sidecar positions stored as teacher-text tokens and retokenized onto the Qwen
student vocabulary. The same day's task-eval lesson adds a second, independent
defect: training loss fell under teacher forcing while autoregressive
generation stayed identical to the base model (accuracy 0, FPR 92%). Training
on off-policy, teacher-authored sequences does not change the sequences the
student actually emits.

Two adjacent sources propose fixes that BrainForge has deferred but not
forgotten. ADR-0018 deferred the on-policy distillation stage of the
Hugging Face agent-training map; TRL's v1 line now carries
`experimental.gkd.GKDTrainer` (`GKDConfig` with `lmbda`, `beta`, `seq_kd`) and
`DistillationTrainer`, which implement Generalized Knowledge Distillation
(arXiv:2306.13649): the student trains on its own generations against a local
teacher's full next-token distribution, eliminating both the retokenization
noise (shared tokenizer, dense distributions) and the exposure bias (on-policy
sequences). Separately, the Sleep framework (Google Research, arXiv:2606.03979)
frames continual learning as a memory-system problem with two phases: an
active phase that accumulates experience in fragile short-term memory, and a
periodic sleep phase that consolidates it into stable parametric memory.
BrainForge's existing loop already resembles the active phase (pipeline
generation, pilot trajectories, GRPO rollouts); what it lacks is a deliberate,
repeatable consolidation phase that respects the forgetting risk measured on
the 26-record holdout.

## Decision Drivers

* The measured sidecar failure is structural: retokenized teacher tokens and
  off-policy sequences, not a tuning problem. A local teacher sharing the
  student tokenizer removes the first defect; on-policy generation removes the
  second.
* TRL 1.12.0 (already resolved by `uv.lock`, ADR-0018's stated adoption path)
  ships the on-policy trainers in `trl.experimental`. The pin still needs to
  tighten to a version validated against the SFT and KD paths (ROADMAP item).
* Training runs on the single RTX 3080 10GB (ADR-0010). GKD adds a local
  teacher alongside the student; the 4-bit quantized layout (student 4-bit,
  teacher 4-bit, paged/8-bit Adam, gradient checkpointing) must be validated
  by preflight before the backend ships.
* Forgetting is a measured risk, not a hypothetical: every post-SFT gain was
  won at a 16-31 record scale, and consolidation rounds that retrain on new
  data alone can erode the holdout numbers. ADR-0003's contamination
  disciplines apply to every replayed record.
* The Sleep framework is an inspiration for the cycle shape, not an
  architecture to port: its MoE expansion and Hope layers assume training new
  expert capacity offline, which a 10GB workstation does not support and
  BrainForge's dense Qwen students cannot host. BrainForge takes the two-phase
  cycle and the low-rank consolidation idea; the MoE/Hope/parameter-expansion
  stage is recorded on the roadmap as a generic-backbone ambition, not
  implemented here.

## Considered Options

* Replace the sidecar KD path immediately with a GKD backend, and bolt replay
  onto `train run` ad hoc.
* Adopt GKD as a new `train` strategy plus a dedicated `train consolidate`
  cycle (replay mix, round gate, journal), keeping the sidecar path until a
  measured three-arm benchmark justifies deprecating it.
* Keep the sidecar path and only add replay mixing to the SFT strategy.

## Decision Outcome

Chosen option: "adopt GKD as a new strategy plus a consolidation cycle, keep
the sidecar path until benchmarked".

1. A `train` strategy `gkd` (`gkd_teacher`, `gkd_lmbda`, `gkd_beta`,
   `gkd_seq_kd` config fields) wraps `trl.experimental.gkd.GKDTrainer`:
   student rollouts on-policy (`lmbda=1.0` default), generalized-JSD loss
   against a local teacher that shares the student tokenizer, PEFT/LoRA
   compatible, heavy imports lazy per the repo rule.
2. A `brainforge train consolidate` command implements the sleep cycle:
   ingest-trajectories merge (existing), stratified replay mix of previously
   consolidated data (`replay_ratio`, contamination guard re-run on the mix),
   one training pass, and a cheap per-round gate (`train task-eval` on the
   frozen post-cutoff holdout). A failed gate keeps the previous round's
   checkpoint; nothing is overwritten in place. Each round journals provenance,
   records seen, and gate outcome under `experiments/consolidation/`.
3. The sidecar path (`pipeline kd-pass`, `train run --kd`) stays supported and
   marked experimental. The three-arm benchmark (no KD vs top-k sidecar vs
   GKD, identical splits) decides its promotion or deprecation with numbers,
   per the ADR-0018 evidence-first discipline.
4. The consolidation cycle is a generic loop, not a SLEEP reimplementation:
   the credit is the wake/sleep phase split and low-rank consolidation
   (each round's adapter is a compressed, replayable knowledge batch). The
   dreaming and capacity-expansion stages are deferred with named
   prerequisites on the roadmap.

### Consequences

* Good, because the two measured defects of the sidecar (retokenized teacher
  targets, off-policy sequences) are addressed by construction: shared
  tokenizer and student-generated sequences.
* Good, because the cycle protects against the forgetting the holdout
  benchmark exposed, and every round leaves a journal and a gated checkpoint
  instead of silently degrading the student.
* Bad, because GKD needs a local teacher on the 10GB host; the VRAM plan is
  unproven until preflight validates the 4-bit student-plus-teacher layout,
  as in ADR-0019 gate (b).
* Bad, because the consolidation command is a new orchestration surface
  (dataset merge, replay, training, gate) with its own failure modes.
* Mitigation: the smoke (one round on the union dataset, replay ratio 0.3,
  gate on the 26-record holdout) is the acceptance test; no `gkd` strategy
  or `consolidate` command ships without it.
* Neutral for judge independence: the GKD teacher is the training-time teacher
  role, distinct from the dataset judge (ADR-0002); provenance and
  contamination tagging stay binding on every replayed record (ADR-0003).

### Confirmation

* `strategy: "gkd"` appears in `config/schema.json` after regeneration
  (`brainforge config schema --write`), with the smoke run proving the
  trainer loads a teacher on the RTX 3080 without exceeding VRAM.
* `brainforge train consolidate` runs end to end on the union dataset and
  its round journal lands under `experiments/consolidation/`; a forced gate
  failure keeps the previous checkpoint (regression test).
* `docs/reference/cli.md` documents both the new strategy and the command;
  the docs-coverage test enforces it in CI.
* The sidecar path's fate is decided by the three-arm benchmark published in
  `docs/explanation/benchmark.md`; until then `train run --kd` stays usable.

## Pros and Cons of the Options

### Replace immediately (rejected)

* Good, because the sidecar's last run was worse than the baseline; deleting
  it now is defensible.
* Bad, because deprecating without the three-arm benchmark contradicts the
  evidence-first discipline ADR-0018 established, and GKD on BrainForge's
  tiny datasets (6-31 records) is unproven.

### Adopt plus cycle, benchmark-gated (chosen)

* Good, because the contract shift (on-policy teacher, replay, round gate)
  is recorded before code exists, as ADR-0019 did for OPST.
* Good, because the sidecar path remains reproducible for the benchmark arms.
* Bad, because two distillation paths coexist until the benchmark lands.

### Sidecar only with replay (rejected)

* Good, because replay mixing alone is the cheapest forgetting countermeasure.
* Bad, because it keeps the retokenized off-policy signal that measured
  worse than the baseline, and leaves the on-policy stage forever deferred.

## Links

* [ADR-0015](0015-knowledge-distillation-sidecar.md): the sidecar path kept
  until benchmarked.
* [ADR-0018](0018-post-training-expansion-deferred.md): the deferral this ADR
  partially supersedes for the distillation stage.
* [ADR-0019](0019-on-policy-self-training-opst.md): the OPST backend, whose
  gating pattern (vectorized loss, VRAM gate, smoke acceptance) this ADR
  reuses.
* [ADR-0010](0010-qlora-training-single-rtx-3080.md) and
  [ADR-0003](0003-contamination-controls-postcutoff-holdout.md): the hardware
  and contamination disciplines the cycle inherits.
* arXiv:2306.13649 (Agarwal et al., 2023): Generalized Knowledge Distillation,
  the on-policy loss this ADR adopts via TRL.
* arXiv:2606.03979 (Behrouz, Hashemi, Javanmard, Mirrokni, 2026): the Sleep
  framework, the source of the consolidation-cycle shape; its capacity
  expansion (MoE, Hope) is a deferred generic-backbone ambition, not an
  adopted architecture.
* The Google Research blog: <https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/>
