# Wake/sleep loop as the framework skeleton: domain-agnostic methods menu

* Status: accepted
* Date: 2026-11-17

## Context and Problem Statement

BrainForge grew through the CTF (lictor/augure) capability loop: teacher
pilot, mining, SFT consolidation, export, bench. That produced a working
end-to-end protocol (mr6..mr11) but left the framework identity ambiguous:
`task_eval`, `grpo`, `rewards`, `consolidation`, `benchmark` and
`pipeline/engine` carry CTF-specific logic (`expected_flag`,
`flag_accuracy`, `grpo_flags.jsonl`, regex `flag{...}`) inside otherwise
generic code. The stated goal is a distillation framework: teacher-agnostic,
model-agnostic (dense, MoE), with a combinable methods menu (SFT, KD, GKD,
GRPO, abliteration, later weight editing), where CTF is one reward domain
among others.

Audit 2026-11-17 (saved in memory) confirmed 11 CTF couplings inside generic
modules, and that ADR-0011 already acté the generic positioning and ADR-0006
already provides the domain-pack mechanism. What is missing is the
mechanical separation: the wake/sleep loop must be defined over domain
packs, not over flags.

## Decision Drivers

* The capability loop (wake: teacher mining; sleep: consolidation SFT/GKD/
  GRPO; wake: export/bench/hot-swap) is the product; domains are plugins.
* GRPO needs a pluggable reward registry (binary CTF flag is one reward,
  not the contract); GKD/OPD need a teacher handle, not a lictor handle.
* Bench metrics must come from the pack (`quality_gate` exists already),
  not from hardcoded columns.
* Abliteration and ROME/MEMIT are model-level operations, domain-free by
  construction; they belong in the sleep phase as optional passes.
* 2026 SOTA (TRL 1.9.2, GKDTrainer in `trl.experimental.gkd`, GRPOTrainer
  with callable reward funcs, OPD as reverse-KL dense reward) matches the
  loop design; no competitor automates teacher hot-swap + wake/sleep e2e.

## Considered Options

* A. Keep CTF logic in the generic modules (current state).
* B. Move generic modules into `domains/ctf/` and duplicate training code.
* C. Break the couplings: add pack-level hooks (eval scoring, arbitrage,
  reward, gate metrics) and relocate CTF-only modules into `domains/ctf/`.

## Decision Outcome

Chosen option: "C", because it preserves the working loop while making the
loop definition domain-free. Concretely:

1. **Wake/sleep contract**: the pipeline engine runs on a `DomainPack` with
   hooks `eval_score`, `arbitrate`, `rewards`, `gate_metrics`. CTF pack
   supplies flag-oracle implementations; a future domain supplies its own.
2. **Relocations**: `dataset/flag_ground_truth.py`, the two CLI commands
   `enrich-flags`/`ingest-trajectories`, and experiments
   `pilot`/`model_only_pilot`/`bench_lab`/`final_bench` + `ctf_ingest` +
   `scripts/` belong to the CTF pack surface (scripts may stay as
   dated one-shots, but import only from `domains/ctf/`).
3. **Methods menu**: `training/` stays generic and gains abliteration as a
   first-class post-consolidation pass (ADR-0014 technique, port of the
   single-direction method without TransformerLens), and the roadmap adds
   OPD (reverse-KL dense reward over GRPOTrainer) and MoE support.
4. **Teacher hot-swap**: the bench/export step (ADR-0021) is the wake
   boundary; the student that fails the gate becomes the curriculum input
   for the next wake, and a stronger teacher model can be swapped in
   without touching domain code.

### Consequences

* Positive: new domains plug in by writing one pack (facts, rewards, eval,
  gate) and optionally experiments; training code stops importing CTF.
* Negative: the mixed modules' tests (`test_grpo_lab`, `test_consolidation`,
  `test_task_eval`) test CTF-incrusted behavior and must be split.
* Mitigation: split tests at the same time as the refactor; keep public
  CLI names stable (`enrich-flags` becomes a `ctf` subcommand).

### Confirmation

* `grep -rn "flag\|lictor\|augure\|ctf" src/brainforge/training/ src/brainforge/pipeline/ src/brainforge/dataset/` returns zero hits outside `domains/ctf/` imports.
* `pytest tests/unit/test_training_gkd.py tests/unit/test_consolidation.py tests/unit/test_task_eval.py` pass after the split.
* A non-CTF pack (mock) can run the full loop in `test_domains.py` style.

## Pros and Cons of the Options

### A. Keep CTF logic in generic modules

* Good, because zero refactor cost.
* Bad, because the framework identity stays CTF-shaped; adding a second
  domain requires editing six core modules.

### B. Move generic modules into domains/ctf/

* Good, because clean separation fast.
* Bad, because it duplicates training code and breaks the 22 ADRs of
  generic positioning; the generic loop is the valuable asset.

### C. Pack-level hooks + relocation (chosen)

* Good, because the loop is defined once over packs; ADR-0011 and ADR-0006
  are honored; SOTA methods (GKD, GRPO, abliteration) slot in unchanged.
* Bad, because it touches six modules and their tests; needs a full
  unit-test run after the split.

## Links

* [ADR-0011](0011-generic-framework-positioning.md) (positioning)
* [ADR-0006](0006-modular-domain-packs.md) (pack mechanism)
* [ADR-0020](0020-gkd-consolidation-cycle.md) (GKD cycle)
* [ADR-0021](0021-final-benchmark-contract.md) (bench contract)
* Audit memory entry: "Audit brainforge: carte core vs CTF, 11 couplages"
* SOTA memory entry: "SOTA distillation 2026 pour brainforge"
