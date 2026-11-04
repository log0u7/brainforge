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

| Model | Train records | eval_loss | perplexity | vs baseline | Steps | Wall time (min) |
|---|---|---|---|---|---|---|
| base Qwen3-1.7B | — | 2.534 | 12.61 | — | — | — |
| SFT (smoke pass) | 3 | 2.242 | 9.42 | −25.3% | 12 | 1.0 |
| SFT | 31 (37-case slice) | 2.090 | 8.09 | −35.9% | 48 | 3.9 |
| SFT (arbitrated 40-case set) | 31 (40-case slice) | 2.072 | 7.94 | −37.0% | 48 | 5.5 |
| SFT (17 GLM-solved records) | 17 | 2.084 | 8.04 | −36.2% | 36 | 5.6 |
| KD (qwen3.5-9b teacher) | 3 | 2.345 | 10.43 | −17.3% | 12 | 168.1 |
| KD (qwen, hard 17-case subset) | 3 | 2.350 | 10.49 | −17.0% | 12 | 1.0 |
| KD (glm-5.3-prime, mid-run snapshot) | 14 | 2.471 | 11.84 | −6.1% | 24 | 37.3 |
| KD (glm-5.3-prime, full 17-record set) | 17 | 2.999 | 20.07 | +59.2% | 36 | n/a |
| KD (qwen3.5-35b-a3b teacher, full pass) | 1 | 2.361 | 10.60 | −16.0% | 12 | n/a |

Wall times are the trainer's measured `train_runtime` (the training
loop only, on the RTX 3080 host). The first KD run's wall time includes
the teacher server's cold start; the later 12-step KD runs reused the
warm server. Two logs (KD glm-5.3-prime full, KD qwen3.5-35b) were lost
before wall-time capture, so their cells stay `n/a` - steps are the
comparable unit, and a KD step costs a teacher forward pass that an SFT
step does not. Caution for anyone reading the run dirs directly: the
glm-5.3-prime full run's `train_summary.json` shows a 0.008 s
`train_runtime`, an artifact of a later failed resume that overwrote
the summary tail; its `checkpoint-36/trainer_state.json` carries the
real 36-step loss history.

SFT scales monotonically with usable records: the 17 GLM-solved
records alone nearly match the 40-case arbitrated result (8.04 vs
7.94). The two 31-record rows are the same trainer on the 37- and
40-case slices; `train_rows` after gate filtering is 31 in both.

### KD status

The KD loss is vectorized since issue #57 (PR #63): the 14-record
GLM-teacher snapshot that used to stall 28+ minutes on its first step
now trains 24 steps in 37 minutes measured (~16x on this workload).
Vectorized
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
2. **Cross-tokenizer teachers turn "learning the teacher" into the
   failure mode.** The GLM teacher's text tokens fragment differently
   over the Qwen student vocabulary. The two run sizes' loss curves
   differ qualitatively, not just in noise volume: the 14-record run
   contains no trainable batch group (its minimum loss stays near 42,
   both groups drift down slowly), so the student never absorbs the
   teacher's distribution and its held-out perplexity stays near
   baseline (11.84); the 17-record run contains a group whose loss
   collapses 78 -> 16 -> 11 within the run, and driving that loss down
   means absorbing the misaligned teacher signal - which is exactly
   what wrecks the held-out perplexity (20.07). Fitting the teacher is
   the poisoning.
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

## Third run results (2026-10-29) - first non-zero task-eval

The union SFT run is the first student that answers the holdout in the
scored format: accuracy 0.1923 (5/26), false-positive rate 0.7308.
The base model and every earlier student sat at accuracy 0.0 with a
0.92 false-positive rate, i.e. the model always answered and never
right. The union student answers less often and sometimes right.

### Config

Data: 16-record union (10 KD-GLM + 6 ROP Emporium eval-format), holdout
ids excluded by id. Training: Qwen3-1.7B, QLoRA 4-bit, rank 32,
max_length 4096, 25 epochs, 200 steps, 33 min on the RTX 3080. Eval:
greedy decoding, 2048-token budget, `--no-thinking` (the prompt carries
the empty think block, matching the training render).

| Model | Train records | eval_loss | perplexity | accuracy | FP rate | FN rate | flag_accuracy |
|---|---|---|---|---|---|---|---|
| base Qwen3-1.7B | 0 | 2.534 | 12.61 | 0.00 | 0.92 | 0.08 | None |
| SFT (17 GLM records, contaminated) | 17 | 2.999 | 20.07 | 0.00 | 0.92 | 0.08 | None |
| SFT (union, holdout-excluded) | 16 | 2.984 | 19.77 | 0.1923 | 0.73 | 0.08 | None |
| SFT (union2, flagged ground truth) | 16 | - | - | 0.2692 | 0.65 | 0.08 | 0.0000 |

The union perplexity (19.77) is worse than baseline (12.61) while the
task accuracy is the first non-zero: the training set teaches detailed
verdict JSON (768-2168 token completions) while the holdout ground
truth runs shorter (548-1891), so the likelihood metric punishes the
longer style the task metric rewards. The perplexity column measures
style fit; the task-eval column measures the format the scorer wants.

The contaminated 17-record run trained on 7 ids that appear in the
holdout and still scored 0.00: the format was learned on in-distribution
prompts and the holdout is a different prompt distribution. The union
run fixes the distribution mismatch (recon-style prompts for both) and
scores the first non-zero accuracy. The union2 run adds
`dataset enrich-flags` (verified-tier flags stamped into ground truth
and training targets) and moves accuracy to 0.2692 with the FPR down
to 0.65; flag_accuracy becomes computable for the first time.

### What still does not work

* flag_accuracy is 0.0000 on the union2 run (after
  `dataset enrich-flags` stamped the real lictor flags into the holdout
  ground truth). The reading stays sober: a 1.7B student trained on 16
  records cannot resolve flags by prediction, and the flag must come
  from execution - the student produces the facts, augure decides the
  technique, lictor executes against the local lab target, and the flag
  read from the running service is the ground truth (and the GRPO
  reward).
* 7/26 is one seed on one holdout, not a benchmark paper.
* 0.65 FPR means the model still guesses on most records it cannot
  parse; the abstention path is unused.

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

## The GRPO stage (2026-10-29)

The execution loop closes the reward: the pilot's trajectories carry
the flag read from the running service, `dataset ingest-trajectories`
turns them into the GRPO dataset (6 rows + the `grpo_flags.jsonl`
sidecar), and the trainer scores each generated verdict against the
flag the service actually read - the execution is the ground truth,
never the model. Config: base = the merged SFT student
(`run-1.7b-sft-17b-merged`), reward `ctf_flag`, 6 prompts, 3 epochs,
9 optimizer steps, 376 s (~6 min) on the RTX 3080.

| Step | Reward |
|---|---|
| 1 | 1.000 |
| 2 | 0.417 |
| 3 | 1.000 |
| 4 | 1.000 |
| 5 | 0.500 |
| 6 | 0.917 |
| 7 | 0.500 |
| 8 | 0.917 |
| 9 | 1.000 |

Mean reward 0.81 over the run. The curve opens at 1.0 because the flag
the verdict must report already sits in the prompt's tool observation;
it breathes between 0.42 and 1.0 where a generation misses the flag
format, and closes at 1.0. Honest framing: this stage trains faithful
reporting of the oracle's truth, not exploration - the RL stage where
the model's actions change the environment (emit the facts, decide,
execute, get paid on the result) is the next one. The reward itself is
now real: it comes from an execution, it replays, and every failed
build the pilot records is a zero-reward rollout. Run:
`experiments/run-grpo-1`.

## The lab loop (2026-10-30) - the actions change the environment

The next stage ran for real: the model's completion IS the suggest
step of a closed loop against the lab. `lab_loop` (PR 107) POSTs the
completion to the lictor reward server (the ssh -R tunnel lands on
the operator's box, the lab container serves the targets); the server
runs the SAME pipeline twice - plan, build walk, run - once on the
raw profiler facts (the baseline) and once on the suggest-enriched
facts, and pays the counterfactual: 1.0 when the suggestion was the
difference, 0.5 when the pipeline lands either way, 0.0 when nothing
lands, -0.25 when the schema refuses the completion. The dataset:
7 records (one per runnable target), the prompt carries the
profiler's facts and asks for the source knowledge no ELF can show;
ret2csu is the only decisive record (its facts lack the vuln class,
the suggestion has to supply it). Config: same base, reward
`lab_loop`, 7 prompts, 12 epochs, 9 optimizer steps, ~27 min on the
RTX 3080 plus the local executions.

| Step | Reward mean | Reward std |
|---|---|---|
| 1 | 0.250 | 0.261 |
| 2 | 0.125 | 0.292 |
| 3 | 0.229 | 0.291 |
| 4 | 0.250 | 0.261 |
| 5 | -0.021 | 0.072 |
| 6 | 0.500 | 0.000 |
| 7 | 0.250 | 0.261 |
| 8 | 0.229 | 0.291 |
| 9 | 0.188 | 0.285 |

Reading: the group variance is real (std 0.26-0.29 on most steps:
1.0, 0.5, 0.0 and -0.25 completions inside the same group - the
counterfactual separates the decisive suggestions from the redundant
and the refused ones). Step 5 shows the schema penalty firing
(mean -0.02: a completion paid -0.25). Step 6 is a uniform group
(all completions redundant, std 0) - zero gradient, `frac_reward_zero_std`
1.0, and the loss is exactly 0.0 there, the expected inertness. The
mean does not climb at this scale (9 optimizer steps, 7 prompts): the
run is the loop's proof of life, not a training result. Two defects
it surfaced and fixed: the measurement sweeps run the target until it
segfaults, so without a core limit every crash wrote a coredump and
apport drowned the box (load 120 on 8 cores, fixed server-side with
RLIMIT_CORE 0 and container-side with the compose ulimit); the run's
status line, not the exit code, is the oracle (a budget-exhausted run
exits 0). Run: `experiments/run-20261030-041919-129979` (gpu3080).

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

## The three-arm benchmark (pending GPU run)

The first benchmark left a question open: plain SFT beat every
distillation variant tried, but the sidecar path carried known
structural defects (retokenized teacher targets, off-policy sequences)
and the on-policy backend did not exist yet. With `strategy: gkd`
shipped ([ADR-0020](../adr/0020-gkd-consolidation-cycle.md)), the
comparison can now run properly.

`brainforge train bench` runs every arm on identical splits and the
same frozen post-cutoff holdout, one output directory per arm, and
writes `results.md` + `results.json`:

| Arm | trains with | requirement |
|---|---|---|
| `no-kd` | plain QLoRA SFT | nothing (baseline) |
| `sidecar` | `train run --kd` (ADR-0015 top-k sidecar) | `kd.jsonl` in the dataset dir |
| `gkd` | `strategy: gkd` (ADR-0020 on-policy, generalized JSD) | `training.gkd_teacher` set |
| `lora64` | plain SFT at LoRA rank 64 | nothing (capacity probe) |

The `lora64` arm is opportunistic: the task-eval lesson (train loss
fell under teacher forcing while autoregressive behavior stayed
base-like) left two hypotheses - insufficient LoRA capacity versus
noisy training signal. Rank 16 vs 64 on identical data separates them
cheaply: if rank 64 unlocks accuracy, the student was capacity-bound;
if not, the signal was the problem and on-policy distillation is the
lever.

Honesty rules carry over from the first benchmark: `n_records` on every
row, one seed, results are an observation on one holdout. The GPU run
on the RTX 3080 host is the next step; until it lands, this table is a
contract, not a result.

## Reproducing the final benchmark

The three-arm table above compares distillation arms on the shared
holdout. The final capability measurement - base vs trained student on
unseen, oracle-verified challenges through the lictor lab loop - runs
through the MR5 contract
([ADR-0021](../adr/0021-final-benchmark-contract.md)). The full setup
(lab, models, runner command, contamination audit) is in
[How-to: run the final benchmark](../how-to/run-final-benchmark.md).

## The final benchmark (2026-10-31): base vs trained on unseen challenges

The first MR5 run that measures what ADR-0021 promised. Prerequisite
fix landed first (PR #121): the harness read the pilot record through
keys the real `run_pilot` never produces (`challenge_id`, `outcome`),
so the first gpu3080 run bucketed every solved challenge under a null
key - ret2plt was executed and returned `FLAG{the_data_decides_ret2plt}`
while `results.json` still scored 0/6. The harness now reads the record
id and the verdict flag from the real `build_trajectory` shape, and the
regression tests feed the doubles the real record shape.

### Config

| Component | Value |
|---|---|
| Lab | lictor docker stack, localhost listeners (port 8000 binaries, 9994-9999 targets) |
| Base model | `base-qwen3-1.7b` (llama.cpp Q8_0, GPU host port 9932) |
| Trained student | `trained-grpo17` (the lab-loop GRPO student merged, port 9933) |
| Eval set | 6 oracle-verified lictor challenges, unseen in training |
| Training IDs | the 7 lab-loop GRPO records (`datasets/grpo-lab/train.jsonl`) |
| Repetitions | 1 |

### Numbers

| model | successes/attempts | success_rate |
|---|---|---|
| base | 4/6 | 0.67 |
| trained | 4/6 | 0.67 |

contamination: clean (no training ID in the eval set; the audit holds).

Per challenge (identical for both models): fmtstr ✓, ret2func ✓,
ret2plt ✓, shellcode ✓, orw ✗, uaf_heap ✗.

### Honest reading

Two findings, both negative, both useful:

1. **No capability gain from the GRPO student.** Base and trained score
   identically (4/6). Expected at this scale: 7 training records, 9
   optimizer steps, one honest epoch of the counterfactual loop. The
   GRPO stage trained faithful reporting of the oracle's truth, not
   exploration; the numbers say the transfer has not started.
2. **The 4/6 measures the pipeline, not the model.** On most challenges
   the pilot logged "the LLM emitted no usable facts" - the model failed
   the facts step and the seeded bandit still picked the technique; the
   flag came from lictor executing the built exploit against the live
   service. The benchmark's success metric (verdict flag == per-target
   oracle flag) validates the loop end to end, and it works: but the
   model-side contribution at this scale is near zero. The capability
   claim waits on a trained student that changes the bandit's decision
   - the three-arm benchmark and scale-up come first.

Reproduction: `experiments/final-bench-results/` holds the run's
`results.json` + `results.md`.

## The 17-challenge scale-up run (2026-11-03): the audit holds, the model does not generalize

The 17-challenge run mandated by the MR5 scale-up decision (2026-11-02):
17 oracle-verified lictor challenges (the original 6 + ret2csu + the
ten new lab variants), 3 repetitions per (model, challenge), the
contamination audit hardened by the namespace fix (PR #124).

### Config

| Component | Value |
|---|---|
| Lab | lictor docker stack, 17 live listeners (9982-9991 new + 9992-9999 original) |
| Base model | `base-qwen3-1.7b` (llama.cpp Q8_0, GPU host port 9932) |
| Trained student | `trained-grpo17` (port 9933) |
| Eval set | 17 oracle-verified lictor challenges |
| Training IDs | the 7 lab-loop GRPO records (`datasets/grpo-lab/train.jsonl`), audited through the PR #124 namespace translation |
| Repetitions | 3 |

### Numbers

| model | successes/attempts | success_rate |
|---|---|---|
| base | 13/51 | 0.25 |
| trained | 12/51 | 0.24 |

### Honest reading

1.  **The contamination audit now bites.** The GRPO training IDs carry
    a `lab-*` namespace while the eval manifest names the same targets
    `ctf-lictor-*`; PR #124 normalizes the comparison, and the run
    honestly flags the 6 seen challenges (`fmtstr`, `orw`, `ret2csu`,
    `ret2func`, `ret2plt`, `shellcode`) instead of certifying a
    contaminated eval as clean.
2.  **No capability gain on unseen challenges.** Both models score
    identically on the 11 unseen challenges: 0/33 each. The GRPO
    student did not generalize to the new variants (canary, safe
    linking, SROP, full-ASLR ret2plt, PIE ret2libc).
3.  **The base model matches the student on seen challenges.** Base
    13/18 vs trained 12/18. The only difference is `orw` (1/3 vs 0/3).
    At this scale, the training has not started to transfer.

Reproduction: `experiments/final-bench-results/run-20261103-17ch/`
holds this run's `results.json` + `results.md`.

## The clean-protocol final run (2026-11-03): 11 unseen challenges, audit passes

The 17-challenge run's audit honestly refuses to certify an eval where
6 challenges are the training targets. The final benchmark therefore
uses the 11 uncontaminated challenges (all the `-002` variants plus
`uaf_heap-001`), passed to the runner via `--manifest` filtering. The
audit passes clean and the runner exits 0.

| model | successes/attempts | success_rate |
|---|---|---|
| base | 26/33 | 0.79 |
| trained | 26/33 | 0.79 |

Reading:
1.  **The exploits+rules work, the training does not.** Unseen
    challenges went 0/33 (run D, no exploits for the new targets) to
    26/33 — that is entirely the E1+E2 effect (ronin exploits for the
    new target families + augure rules routing srop/canary/tcache).
    The GRPO student shows no measurable capability gain over base.
2.  **The per-challenge differences wash out.** Trained is better on
    `ret2libc-001` (3/3 vs 2/3) and `srop-001` (2/3 vs 1/3), worse on
    `uaf_heap-002` (0/3 vs 2/3); everything else identical.
3.  **The persistent failure is `uaf_heap-001`** (0/3 both models) —
    the designed handoff, still unresolved at the pilot level.

Reproduction: `experiments/final-bench-results/run-20261103-11ch-clean/`
holds this run's `results.json` + `results.md`.

## The SFT fact-suggestion experiment and the 10-repetition final run (2026-11-03)

After the Étape 0 diagnostic (the suggest LLM is actively harmful when
wrong: bogus facts redirect routing, degenerate repetition poisoned one
fact base, and the GRPO student emitted nothing on the hardest target),
PR #127 capped and deduplicated the suggested facts and PR #128 built a
tiny SFT fact-suggestion dataset (`datasets/sft-facts/train.jsonl`:
6 records, 4 with empty completions, `uaf_heap` excluded because its
material is an eval challenge). Training ran on gpu3080 (60 steps, loss
to 0.0002) and the exported GGUF served as `trained-sft-facts` on port
9933.

PR #129 also made the harness robust after a blocked target
(`shellcode-002` lictor timeout) destroyed an earlier 10-repetition
run: a hung lictor subcommand now produces a failed `timeout` record
instead of killing the whole bench.

The final protocol: the 11 uncontaminated challenges × 2 models ×
10 repetitions = 220 executions, per-target oracle regex on the
running service, contamination audit clean.

### Numbers

| model | successes/attempts | success_rate |
|---|---|---|
| base | 84/110 | 0.76 |
| trained-sft-facts | 61/110 | 0.55 |

### Per-challenge breakdown

| challenge | base | trained-sft-facts |
|---|---|---|
| callme-002 | 7/10 | 8/10 |
| fmtstr-002 | 10/10 | 10/10 |
| orw-002 | 9/10 | 10/10 |
| ret2func-002 | 10/10 | 10/10 |
| ret2libc-001 | 9/10 | 2/10 |
| ret2libc-002 | 8/10 | 6/10 |
| ret2plt-002 | 10/10 | 10/10 |
| shellcode-002 | 10/10 | 5/10 |
| srop-001 | 10/10 | 0/10 |
| uaf_heap-001 | 0/10 | 0/10 |
| uaf_heap-002 | 1/10 | 0/10 |

### Honest reading

1.  **The SFT fact-suggestion pass regressed the student.** The model
    learned to emit fact lines (or nothing) and forgot exploitation:
    `ret2libc-001` 2/10 vs 9/10, `srop-001` 0/10 vs 10/10,
    `shellcode-002` 5/10 vs 10/10. A 6-record SFT dataset (4 empty
    completions) taught the model to talk about facts, not to exploit:
    catastrophic forgetting at textbook scale.
2.  **The Étape 0 verdict is confirmed by execution.** The student
    emits facts that are often wrong (`vuln("fmtstr")` on a ret2libc
    challenge), augure routes on them, and the exploit fails. The base
    model, with no suggested facts, routes correctly on profiler-only
    facts.
3.  **The first clean run's 4/6 = 4/6 tie was a small-sample wash-out,
    not an equilibrium.** At 10 repetitions the base model is a strong
    adversary (0.76) and the trained delta (-0.21) is the exact
    signature of a tiny-dataset SFT.
4.  **`uaf_heap-001` stays 0/10 for both models.** The designed
    handoff remains unresolved at the pilot level; the capability is
    absent from base and student alike.

Reproduction: `experiments/final-bench-results/run-20261103-11ch-10rep-b/`
holds this run's `results.json` + `results.md`. The failed first attempt
(`run-20261103-11ch-10rep`) is kept as the harness-robustness evidence.

### Next cycle (MR6)

The fix is data volume and execution-grounded training, not more
fact-talking: enlarge the SFT dataset to 1000+ records mined from the
logged trails (training-eligible targets only, eval IDs never enter
training), add replay data to prevent forgetting, and train the
use-after-free capability class on dedicated lab twins so the eval
targets stay unseen.

## The MR6 retrain run (2026-11-04): execution-grounded dataset, final 10-rep bench in flight

MR6 executed the fix: a much larger, execution-grounded SFT dataset
replaces the 6-record fact-suggestion dataset. The training data is
mined from logged trails (training-eligible targets only, eval IDs
never enter training), quality-gated, and split 75/25/15
(train/validation/test) into `datasets/sft-mr6/train.jsonl` (115
records).

The retrain ran on gpu3080 as QLoRA 4-bit on Qwen3-1.7B: 12 epochs,
120 steps, 2h25m wall clock. Final metrics:

| metric | value |
|---|---|
| train_loss | 0.0797 |
| eval_loss | 1.465e-04 |
| token accuracy | 1.0 |
| eval_entropy | 8.1e-04 |

Honest reading: eval_loss below train_loss, token accuracy at 1.0 and
near-zero entropy are the signature of overfitting/memorization on a
tiny dataset. That is expected at this scale and says nothing about
exploitation capability. The metric that matters is the execution
benchmark below, not the training curves.

The merged adapter was exported to GGUF Q4_K_M (`models/export-mr6/`)
and served through llama.cpp on port 9933 under the alias
`trained-mr6`. The comparison stays base Q8_0 vs trained Q4_K_M, the
same quantization asymmetry the previous runs assumed.

### Numbers (PENDING: bench `run-20261104-mr6-10rep` still running)

| model | successes/attempts | success_rate |
|---|---|---|
| base | TBD | TBD |
| trained-mr6 | TBD | TBD |

### Per-challenge breakdown (all values TBD, bench in flight)

| challenge | base | trained-mr6 |
|---|---|---|
| callme-002 | TBD | TBD |
| fmtstr-002 | TBD | TBD |
| orw-002 | TBD | TBD |
| ret2func-002 | TBD | TBD |
| ret2libc-001 | TBD | TBD |
| ret2libc-002 | TBD | TBD |
| ret2plt-002 | TBD | TBD |
| shellcode-002 | TBD | TBD |
| srop-001 | TBD | TBD |
| uaf_heap-001 | TBD | TBD |
| uaf_heap-002 | TBD | TBD |

### Honest reading (to be written once the run lands)

To be filled when the run completes. The questions this run answers:
does the execution-grounded dataset recover the regression that the
fact-suggestion SFT caused (`ret2libc-001` 2/10, `srop-001` 0/10,
`shellcode-002` 5/10), does it beat base 0.76, and does `uaf_heap-001`
finally move off 0/10.

Reproduction: `experiments/final-bench-results/run-20261104-mr6-10rep/`
will hold this run's `results.json` + `results.md` (11 clean
challenges × 2 models × 10 repetitions = 220 executions; expected
~4 hours after launch).
