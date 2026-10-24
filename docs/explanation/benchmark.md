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
