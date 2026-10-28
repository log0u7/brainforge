# On-policy self-training (OPST) adopted as an experimental training backend

* Status: accepted
* Date: 2026-10-28

## Context and Problem Statement

The HuggingFace "train the agent" map orders post-training as SFT, then
distillation, then RL (ADR-0018). BrainForge took the first step: the
verified-dataset pipeline plus QLoRA SFT is measured at -37% perplexity
on the post-cutoff holdout (12.61 base vs 7.94 after SFT on the 1.7B
student).

The second step, distillation, exists only as the ADR-0015 sidecar and
depends on an external teacher that serves logprobs. ADR-0018 deferred
the on-policy distillation trainers and GRPO; since then the first
benchmark is published and the KD loss is vectorized (issue #57, closed).

On-Policy Self-Training (OPST, arXiv 2610.32770) proposes a data-free
path between distillation and RL: no external teacher and no ground
truth. Guided only by coarse topic descriptions, the model generates its
own QA curriculum and a privileged reasoning trace z = (rationale,
answer) from a higher-compute decoding of itself. Training is on-policy
self-distillation: the student samples y given x, an EMA copy of the
student scores y conditioned on (x, z), and the student matches that
scoring with an asymmetric JSD loss (alpha = 0.8, EMA beta = 0.05, N = 7
rounds of about 285 samples each).

The apparent self-distillation paradox is resolved by an information
gap, not a capacity gap: the teacher sees z in context. The signal is
therefore robust to wrong self-generated answers: with 75% of them
replaced by wrong ones, OPST loses 1.9 points and stays 3.3 above the
base model, while SFT on the same data loses 5.7 and falls 4.4 below it.
The paper validates OPST on Qwen3-4B-Base (+6.0 average, GSM8K +13.3),
the same family as our students.

## Decision Drivers

* The data contract changes, not just a flag: OPST stores and reuses a
  privileged trace z per record, runs generation and scoring on the same
  model, and reverses the pipeline order (rollouts first, scoring
  after). An architectural change needs a recorded decision.
* The KD loss vectorization (issue #57) is a hard prerequisite: OPST
  multiplies the scoring passes (per-round generation, verification
  rollouts, teacher scoring conditioned on z).
* Training runs on the single RTX 3080 10GB (ADR-0010): a 1.7B student
  plus an EMA teacher on CPU, paged/8-bit Adam and gradient
  checkpointing is the only plausible layout; naive Adam plus an EMA
  copy on GPU needs 14-25GB.
* The judge-independence discipline (ADR-0002) cannot apply: this loop
  has no external judge. The provenance and contamination disciplines
  (ADR-0003, ADR-0007) must: every self-generated record stays tagged
  and the contamination guard keeps checking against the holdout.

## Considered Options

* Adopt OPST now as an experimental backend, gated on the vectorized KD
  loss and a validated VRAM plan, with a minimal smoke as the acceptance
  test.
* Add the backend experimentally without an ADR.
* Wait for external logprobs-capable teachers and extend the ADR-0015
  sidecar instead.

## Decision Outcome

Chosen option: "adopt now, gated". OPST is adopted as an experimental
training backend (a `train` strategy `opst`) after two gates: (a) the KD
loss vectorization lands (issue #57, closed) and (b) the VRAM plan is
validated on the 10GB RTX 3080 (student on GPU, EMA teacher on CPU,
paged/8-bit Adam, gradient checkpointing).

Relative to the current KD backend, the contract shifts:

1. No external teacher: the generator is the trained model itself, a
   closed loop.
2. The teacher context includes the privileged trace z = (rationale,
   answer): the sidecar contract gains a tagged, reusable per-record z.
3. On-policy ordering: student rollouts first, teacher scoring after,
   reversed versus our off-policy pipeline.
4. The target is an asymmetric JSD (alpha = 0.8) on the full next-token
   distribution; our top-k KD loss is an approximation of it. The top-k
   approximation is acceptable for the smoke; the exact JSD is the
   target.
5. An EMA teacher (beta = 0.05) and a round driver (N = 7, about 285
   samples per round, inter-round dedup) come with the backend.

BrainForge already covers the rest: case generation per topic/domain
(the pipeline), provenance JSONL, the KD top-k loss pattern, and the
reward registry (GRPO is unused in OPST).

The decision's acceptance test is a minimal smoke: one CTF topic, 50-100
self-generated questions, G = 4 rollouts, N = 1 round, the student as
its own generator, JSD on top-k. Success: holdout perplexity improves
without out-of-domain regression.

### Consequences

* Good, because distillation no longer depends on an external teacher or
  ground truth, and the curriculum adapts to the student's capability
  frontier (the consensus gate keeps what it cannot already solve).
* Good, because the paper's robustness result (75% wrong answers still
  improve over base) fits unverified self-generated data.
* Bad, because the implementation costs an estimated 2-4 days on top of
  the KD backend it extends.
* Bad, because the VRAM plan is unproven until gate (b) runs on the real
  host; EMA weights and generation passes multiply memory pressure.
* Neutral for judge independence: no external judge exists in this loop,
  so the discipline is inapplicable here, not satisfied. Provenance
  stays binding: every generated record is tagged, and the contamination
  guard still screens generated questions against the holdout.
* Mitigation: the smoke is the gate; no `opst` strategy ships without
  it.

### Confirmation

* Issue #57 is closed: the KD loss is tensor-vectorized and no longer
  rebuilds the vocab per position, so the multiplied scoring passes have
  a vectorized base.
* The VRAM plan (gate b) is validated by a training run on the RTX 3080
  with the layout above before the strategy merges; the auto-fit
  preflight must accept the configuration.
* The `opst` strategy lands with its own change set, its smoke criteria
  and CI proof; this ADR records the contract that change set implements.

## Pros and Cons of the Options

### Adopt now, gated (chosen)

* Good, because the contract change is decided before code exists, and
  the smoke defines success before the first GPU hour.
* Good, because the paper removes both blockers at once: external
  teacher and ground truth.
* Bad, because it spends 2-4 days on an experimental path while the RL
  items stay deferred.

### Adopt without an ADR (rejected)

* Good, because an opt-in experimental flag could ship faster.
* Bad, because the data contract (privileged z, on-policy ordering, EMA
  teacher, rounds) is architectural: dataset format, training loop and
  provenance all change. Deciding it in a flag's changelog entry hides
  the shift.

### Wait for external logprobs-capable teachers (rejected)

* Good, because it extends ADR-0015 with no new trainer code.
* Bad, because OPST removes the external dependency entirely, and the
  paper validates the data-free loop on Qwen3-4B-Base (+6.0 average,
  GSM8K +13.3), the same family as our students. Waiting preserves a
  dependency we no longer need.

## Links

* [ADR-0015](0015-knowledge-distillation-sidecar.md): the KD backend
  OPST extends.
* [ADR-0018](0018-post-training-expansion-deferred.md): the deferred
  on-policy distillation and RL stages.
* [ADR-0010](0010-qlora-training-single-rtx-3080.md): the training
  hardware.
* [ADR-0002](0002-multi-teacher-ensemble-independent-judge.md) and
  [ADR-0003](0003-contamination-controls-postcutoff-holdout.md): the
  disciplines this decision reshapes.
* Issue #57: <https://github.com/log0u7/brainforge/issues/57> (closed).
* The OPST paper: <https://arxiv.org/html/2610.32770v1>
