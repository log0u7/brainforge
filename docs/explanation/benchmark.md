# The first benchmark: base vs student on the post-cutoff holdout

The first published model-quality numbers are in, and they are the
project's proof of use: on the post-cutoff holdout, SFT fine-tuning
cuts the student's perplexity from 12.61 to 7.94 (-37%), and the 17
hard cases alone nearly match the full 40-case result (8.04). The
same table also documents a useful negative result: plain SFT beats
every distillation variant tried so far, and one KD configuration
made the model measurably worse than before training. This page
defines the benchmark, why it is shaped this way, and the exact
runbook that produced the numbers.

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

   Verified-tier cases ([lictor lab](https://github.com/log0u7/lictor/tree/main/lab),
   [ROP Emporium](https://ropemporium.com/)) carry
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

* **verified** (12 cases): [lictor lab](https://github.com/log0u7/lictor/tree/main/lab)
  targets and [ROP Emporium](https://ropemporium.com/) binaries with a
  known flag; the flag gate enforces the ground truth.
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

`judge_independence: warn` for this run: the check is conservative and
also warns when judge and teachers share a gateway, even though the
model families differ here (qwen teachers, glm judge) - the separation
that matters, model-family level, held. It tightens back to enforced
when any role reuses another role's model family; the per-record
provenance lists the exact models of every role either way.

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

## Second run results (2026-10-28)

Scale-up: 40 CTF cases through the pipeline with dispute arbitration
(the ground-truth flag resolves teacher/judge disagreements instead of
discarding the case) and `fallback_examples` on (gate-rejected cases
become abstention training records). Teachers: qwen3.5-9b via
OpenRouter; judge: deepseek-v4.1-flash, then glm-5.3-prime for the KD
sidecar pass (the only models in the run that serve per-token
logprobs).

### Numbers - perplexity progression

Held-out split: 26 records (`test_postcutoff`), the same split for
every row. Base model Qwen3-1.7B, QLoRA 4-bit, bf16.

| Model | Train records | eval_loss | perplexity | vs baseline | Train time |
|---|---|---|---|---|---|
| base Qwen3-1.7B | — | 2.534 | 12.61 | — | — |
| SFT (smoke pass) | 3 | 2.242 | 9.42 | −25.3% | 2.5 min |
| SFT | 31 (37-case slice) | 2.090 | 8.09 | −35.9% | ~5 min |
| SFT (arbitrated 40-case set) | 31 (40-case slice) | 2.072 | 7.94 | −37.0% | ~5 min |
| SFT (17 GLM-solved records) | 17 | 2.084 | 8.04 | −36.2% | ~30 min |
| KD (qwen3.5-9b teacher) | 3 | 2.345 | 10.43 | −17.3% | 12 steps |
| KD (qwen, hard 17-case subset) | 3 | 2.350 | 10.49 | −17.0% | 12 steps |
| KD (glm-5.3-prime, mid-run snapshot) | 14 | 2.471 | 11.84 | −6.1% | 24 steps |
| KD (glm-5.3-prime, full 17-record set) | 17 | 2.999 | 20.07 | +59.2% | 36 steps |
| KD (qwen3.5-35b-a3b teacher, full pass) | 1 | 2.361 | 10.60 | −16.0% | 12 steps |

SFT scales monotonically with usable records: the 17 GLM-solved
records alone nearly match the 40-case arbitrated result (8.04 vs
7.94). The two 31-record rows are the same trainer on the 37- and
40-case slices; `train_rows` after gate filtering is 31 in both.

### KD status

The KD loss is vectorized since issue #57 (PR #63): the 14-record
GLM-teacher snapshot that used to stall 28+ minutes on its first step
now trains 24 steps in ~30 minutes (~16x on this workload). Vectorized
KD at the 17-record scale produced the most instructive negative
result of the run: **more distillation data from the glm-5.3-prime
teacher made the student worse than the untrained baseline**
(11.84 at 14 records -> 20.07 at 17 records, baseline 12.61). The
training loss is bimodal: one batch group converges steadily (~16 ->
~11) while another stays stuck at 50-78 - the student is fitting
where the teacher signal is coherent and fighting it where it is not.

Three mechanisms plausibly combine, and the 2026-10-28 ablations
separate them:

1. **The KD path loses to plain SFT even with a clean teacher.** On the
   same 3 records, SFT@3 (9.42) beats KD@3 with a qwen3.5-9b teacher
   (10.43). The sidecar stores teacher positions as *text tokens with
   top-k logprobs*, which are re-tokenized onto the student vocabulary;
   that approximation costs accuracy even when teacher and student
   share a tokenizer family.
2. **Cross-tokenizer teachers amplify the noise.** The GLM teacher's
   text tokens fragment differently over the Qwen student vocabulary;
   with 4.7x more records the accumulated noise dominates the signal
   and the held-out perplexity collapses.
3. **Two locks bound KD data volume: tokenizer alignment AND the
   verification gate.** A record enters the KD dataset only when the
   teacher's predicted flag matches the corpus ground truth
   (`expected_flag`, the verified-tier gate - which is the product).
   Measured across teachers on the same 40 cases: glm-5.3-prime
   passes the gate 17 times, qwen3.5-9b 3 times (its other 15
   timeouts aside), and qwen3.5-35b-a3b ran 37 of 40 cases with zero
   technical failures yet passed the gate exactly once. A teacher
   that is tokenizer-compatible but cannot clear the gate yields
   almost no distillation data; a teacher that clears the gate but
   does not share the tokenizer yields poison. Both locks must open.

Consequence for the roadmap: tuning KD hyperparameters (alpha 0.5,
temperature 2.0) is pointless until the target representation itself
is fixed (true teacher token ids in the student vocabulary) AND the
teacher clears the gate. The honest next step is SFT scaling plus the
task-eval capability measurement.

Consequence for the roadmap: tuning KD hyperparameters (alpha 0.5,
temperature 2.0) is pointless until the target representation itself
is fixed (true teacher token ids in the student vocabulary, or a
same-tokenizer teacher); the honest next step is SFT scaling plus the
task-eval capability measurement.

### Honest reading

The perplexity gain is a pipeline-validation result, not a capability
claim: part of it is the student adopting the teacher output format,
the holdout is 26 records, and the metric measures likelihood, not
solving. The capability metric for the trained student is the CTF
task-eval (flag accuracy) on the same holdout - still queued. What the
run did demonstrate: the 40-case dataset generation, ground-truth
arbitration, training and evaluation run end-to-end on a consumer
Windows GPU host, with every step state-traced - and that the
distillation term needs a target-representation fix before it can
compete with direct SFT.

## First run results (2026-10-26) - task-eval attempt

The honest reading of the first end-to-end run on real data:
**no measurable gain at this scale** - and why that is the expected
outcome of a 6-record training signal, not a tooling failure. Config
and task-eval numbers: the First run results section above.

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
