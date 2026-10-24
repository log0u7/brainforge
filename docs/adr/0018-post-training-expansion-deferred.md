# Post-training expansion: TRL v1, on-policy distillation, GRPO and OpenEnv deferred

* Status: accepted
* Date: 2026-10-25

## Context and Problem Statement

The Hugging Face 2026 agent-training research (the "Train the agent" PyTorch
Conf NA deck by Sergio Paniego, the TRL trainer taxonomy, OpenEnv and the
multi-harness RL guide) maps five post-training stages for agents: SFT on
agent traces, on-policy distillation, GRPO with verifiable rewards, RL in
environments, and RL through the harness. BrainForge covers dataset
generation and the SFT stage today, publishes no benchmark yet, and pins
`trl>=0.11` while the trainers this map relies on live in TRL v1.x
(`DistillationTrainer`, `AsyncGRPOTrainer` in `trl.experimental`).

Each next stage would be a real project: a TRL major-version migration, a
second distillation implementation, a new trainer path, a Docker
environment, or a vLLM-based capture proxy. None of it is worth building
before the tool demonstrates its core claim on measured numbers.

## Decision Drivers

* No published benchmark exists: accuracy on the post-cutoff holdout,
  before/after SFT, is the missing evidence (README states this limit).
* The `trl>=0.11` pin is a floor: uv.lock already resolves TRL 1.12.0 and
  the SFT path uses the v1-style API. The on-policy distillation trainer
  landed later in the v1 line, so that stage starts with a pin bump plus
  validation, not a code migration.
* On-policy distillation (GKD) needs a local teacher sharing the student
  tokenizer; the current sidecar path consumes API logprobs instead
  (ADR-0015). The two are complementary, not interchangeable.
* Verifiable-reward RL (GRPO) fits the CTF pack (a found flag is a binary
  reward) but needs a trainer path, an environment and GPU capacity the
  project does not operate yet from the development laptop.
* Training runs on a single RTX 3080 10GB on a separate Windows host
  (ADR-0010); every RL stage adds VRAM and engineering pressure.

## Considered Options

* Adopt the full stack now: bump TRL to v1.x, add an on-policy distillation
  backend, wire GRPO on the CTF pack, package an OpenEnv environment.
* Defer the whole expansion and spend the effort on the first measured
  benchmark (real dataset, base-vs-student task-eval on the post-cutoff
  holdout).
* Defer selectively: migrate TRL now, keep everything else deferred.

## Decision Outcome

Chosen option: "defer the whole expansion", because the benchmark is the
critical path and every expansion item multiplies the codebase that the
benchmark would have to validate. The TRL v1 migration is itself deferred:
the current SFT/QLoRA path works, and migrating before publishing numbers
would couple two moving targets.

The expansion is recorded, not forgotten. Each stage has a roadmap entry
with its dependency, and the study material is indexed in
[references](../reference/references.md).

### Consequences

* Good, because the project's next milestone stays "demonstrated useful",
  not "has more experimental flags".
* Good, because TRL v1.x trainers evolve quickly; adopting later means
  adopting a stabler API surface.
* Bad, because the experimental `--kd` top-k sidecar (ADR-0015) remains the
  only distillation path, with its known structural limits (truncated
  teacher distribution, off-policy sequences, approximate alignment).
* Bad, because BrainForge cannot claim agent-training coverage beyond SFT
  until the deferral is revisited.
* Mitigation: every deferred stage is an unchecked roadmap item with its
  prerequisite named, so revisiting is a planned decision, not an
  archaeology exercise.

### Confirmation

* [ROADMAP](../../../ROADMAP.md) phase 3 carries the deferred items as
  unchecked entries, each naming its prerequisite.
* `pyproject.toml` still declares the `trl>=0.11` floor while `uv.lock`
  resolves 1.12.0; the bump to the on-policy trainers' version lands with
  its own validation change set and CI proof.
* The KD path is still the ADR-0015 sidecar; an on-policy backend would
  arrive with a new ADR superseding this one for that stage.

## Pros and Cons of the Options

### Adopt now

* Good, because the distillation trainer would replace an approximate
  implementation with a principled one immediately.
* Bad, because it stacks a dependency migration, a new trainer, a new
  dataset export and an environment on top of an unbenchmarked core.
* Bad, because GPU capacity (RTX 3080 10GB, remote host) cannot support
  the RL stages' generation load in a validated configuration.

### Defer all (chosen)

* Good, because one metric (post-cutoff holdout, before/after SFT)
  unblocks every later decision.
* Good, because the effort concentrates on data engineering, the project's
  actual differentiator.
* Bad, because the CTF pack's verifiable-reward fit stays unused for now.

### Defer selectively (tighten the pin first)

* Good, because it would shorten the path to the distillation stage.
* Bad, because the bump alone adds a validation burden on every training
  path without adding user-visible value before the benchmark.

## Links

* [ADR-0015](0015-knowledge-distillation-sidecar.md): the current KD path.
* [ADR-0010](0010-qlora-training-single-rtx-3080.md): the training hardware.
* [ROADMAP](../../../ROADMAP.md): the deferred items with dependencies.
* [references](../reference/references.md): the deck, blog posts and papers.
* The "Train the agent" deck:
  <https://huggingface.co/spaces/sergiopaniego/train-the-agent-slides>
* The multi-harness RL guide:
  <https://huggingface.co/spaces/FineEnvs/multi-harness-rl>
