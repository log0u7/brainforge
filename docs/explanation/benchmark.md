# The first benchmark: base vs student on the post-cutoff holdout

BrainForge ships no published model-quality numbers yet (see the README
status table). This page defines the first benchmark, why it is shaped
this way, and the exact runbook to produce it. Until the numbers below
exist, the project's honest claim stays "well built", not "demonstrated
useful".

## What it measures

One question: does the fine-tuned student beat the base model on the
task, on data the teachers cannot have recited?

* **Split**: `test_postcutoff.jsonl` (primary; cases dated after every
  teacher and judge cutoff, so recitation is structurally excluded) and
  `test.jsonl` (secondary).
* **Model pair**: the base model (same id, no adapter) vs the fine-tuned
  student (merged or adapter-loaded), nothing else.
* **Metrics** (from `train task-eval`): accuracy, false positive and
  false negative rates, `cwe_accuracy` (security) or `flag_accuracy`
  (CTF: predicted flag vs the judge flag, which the verified tier
  gate-checks against the known flag), abstention rate and
  `abstain_recall`.
* **Honesty rules**: report `n_records` with every number; one seed;
  results are an observation on one holdout, not a benchmark paper.
  This mirrors the multi-harness guide's own caution about single-seed
  runs.

## Where it runs

The development laptop has no NVIDIA GPU. Training and evaluation run
on the Windows host with the RTX 3080 10GB
([ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md)), reached
either through the `GPU` GitHub Actions workflow (self-hosted runner,
see the [registration how-to](../how-to/register-gpu-runner.md)) or
over SSH (`GPU train via SSH`).

## Runbook

On a machine with provider keys configured:

1. Generate the dataset (real providers; the mock provider fills
   schemas deterministically and proves nothing about quality):

   ```sh
   uv run brainforge pipeline run ctf --input examples/cases \
     --output datasets/ctf-first.jsonl
   ```

   Verified-tier cases (lictor lab, ROP Emporium) carry
   `metadata.expected_flag`: the CTF gate rejects a judge flag that
   does not match it. The mock run therefore rejects every
   verified-tier case by design.

2. Prepare the splits and check the post-cutoff holdout size:

   ```sh
   uv run brainforge train prepare datasets/ctf-first.jsonl \
     --output datasets/prepared-ctf
   ```

3. On the GPU host, train the student (QLoRA, small model first):

   ```sh
   uv run brainforge train run -d datasets/prepared-ctf \
     --base-model Qwen/Qwen3-1.7B
   ```

4. Evaluate both sides on the same split:

   ```sh
   uv run brainforge train task-eval --model Qwen/Qwen3-1.7B \
     -d datasets/prepared-ctf/test_postcutoff.jsonl
   uv run brainforge train task-eval --model experiments/<run> \
     -d datasets/prepared-ctf/test_postcutoff.jsonl
   ```

   Capture the console output of each side: for the Hub-id base model it
   is the only artifact (no `task_eval.json` is written next to remote
   weights); the local student side also gets its report file.

5. Publish the table here with `n_records`, both sides, and the
   holdout size; tick the ROADMAP phase 2 entries it satisfies.

## Dataset tiers behind it

The CTF ingestion (see `experiments/ctf_ingest/README.md`) feeds this
benchmark with two tiers:

* **verified** (12 cases): lictor lab targets and ROP Emporium
  binaries with a known flag; the flag gate enforces the ground truth.
* **unverified** (11 cases): CTF-Archives 2026 dumps with a flag
  format only; judge-verdict agreement is measured, flags are not.

A holdout below 20 cases triggers the documented warning; the first
run may ship with that warning visible rather than padded with
unverifiable material.

## First run results (2026-10-26)

The first end-to-end run on real data. Config, numbers and the honest
reading: **no measurable gain at this scale** - and why that is the
expected outcome of a 6-record training signal, not a tooling failure.

### Config

| Role | Model | Where |
|---|---|---|
| `ctf_teacher` | qwen3.5-9b | OpenRouter API (batch 1), then the same model as a local Q4_K_M GGUF on the RTX 3080 via llama.cpp (batch 2) |
| `coding_teacher` | qwen/qwen3-coder-next | OpenRouter |
| `judge` | z-ai/glm-5.3-flash, 10000-token budget | OpenRouter |

`judge_independence: warn` for this run: every provider was OpenRouter
(the session's Zen key had expired). The provider-level separation
returns with a fresh Zen key; the per-record provenance lists the exact
models of every role either way.

Data: 23 CTF cases (12 verified tier with curated technique ground
truth, 11 Zeekr 2026 tier-B). **15 accepted** (12 verified + 3 tier-B,
65% acceptance); rejections documented per case: unresolved teacher
disagreements on hard forensics, judge structured-output truncation at
low token budgets.

Training: Qwen3-1.7B QLoRA 4-bit on the RTX 3080 10GB. Train split = the
6 ROP Emporium cases (dated 2020, pre-cutoff, `recitation_risk` tagged);
holdout = the 9 post-cutoff cases the teachers cannot have recited.
10 epochs (one optimizer step per epoch: 6 records), warmup 0.

### Numbers

| Model | n | accuracy | FP rate | FN rate | flag_accuracy |
|---|---|---|---|---|---|
| base Qwen3-1.7B | 9 | 0.00 | 0.89 | 0.11 | None |
| student (6 records, 10 steps) | 9 | 0.00 | 0.89 | 0.11 | None |

n = 9, one seed, greedy decoding, 2048-token generation budget:
an observation on one holdout, not a benchmark paper.

### Honest reading

No measurable gain at this scale. Both models answer the challenge in
prose reasoning and never emit the `CtfVerdict` JSON the task scores:
the base model reasons past the budget, and the student - trained on 6
records for 10 steps - degenerates into repetition loops (overfit on a
tiny signal). The training signal was orders of magnitude below what a
1.7B model needs to adopt the output format, exactly the scale-up the
[ROADMAP](../../ROADMAP.md) phase 2 queues (500-2000 records).

What the run did demonstrate:

* the full pipeline works end to end on one workstation: local teacher
  generation, provider-separated adjudication gates, QLoRA training,
  task-level evaluation on a post-cutoff holdout, published numbers;
* the measurement harness refuses to flatter: an unparseable prediction
  is a false positive, not a free pass.

### Defects the run surfaced

* the verified-tier flag gate demanded a flag that lives outside the
  challenge material; replaced by curated technique gating (PR 24);
* `qwen3-coder-480b` was an invalid OpenRouter id: every STANDARD-mode
  run failed on the coding step (PR 25);
* task-eval collapsed a multi-prompt batch into one chat sequence; a
  0-record validation split aborted training; the base-model side could
  not resolve a Hub model id (PR 26);
* `warmup_steps` (5) larger than the total step count (3) silently
  zeroed the learning rate: the first training was mathematically inert.
  Watch for a guard; the rerun used warmup 0.
