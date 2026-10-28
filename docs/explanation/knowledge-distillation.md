# Knowledge distillation

BrainForge distills a multi-teacher ensemble into one student. This page
explains the two mechanisms and when each applies.

## Dataset-level distillation (default)

The pipeline runs every case through independent teachers, an adversarial
critic and a judge, then keeps only judge-approved records with provenance
and contamination metadata. `train run` fine-tunes the student on those
records.

- **Teachers**: any open models (cloud APIs, local servers, Ollama).
- **Critic**: a devil's advocate that must analyze independently first.
- **Judge**: must be the last step; config load enforces that it shares
  neither provider nor model family with any teacher
  ([judge correlation](judge-correlation.md)).
- **Gate**: per-domain quality rules decide accept/reject; rejected cases
  are quarantined with reasons.

This is the workhorse: it works for any domain, needs no logits, and the
verification machinery (independence, contamination, holdout) applies
unchanged.

## Logit distillation (experimental, `--kd`)

When you want the student to learn the teacher's *distribution* rather than
just its answers, two commands add soft targets:

```bash
brainforge pipeline run ticket_triage --input data/raw --kd
brainforge train run --dataset-dir datasets/prepared --kd
```

1. `--kd` on the pipeline makes the judge's generation call request
   `logprobs=true, top_logprobs=<kd_topk>` through the OpenAI-compatible
   chat completions API (Ollama serves this too). Per generated token, the
   sidecar `<dataset>.kd.jsonl` stores the top-k alternatives with
   logprobs.
2. `train run --kd` trains with

   ```text
   loss = alpha * T^2 * CE(student_T, teacher top-k) + (1 - alpha) * CE(gold)
   ```

   where `CE(student_T, teacher top-k)` is the cross-entropy of the
   student's temperature-scaled distribution against the teacher's
   renormalized top-k distribution, and `CE(gold)` is the ordinary SFT
   term.

### How positions align

The teacher's tokens and the student's tokens need not match (different
tokenizers are fine):

- The dataset record stores the teacher's **exact output** (not a
  re-serialization), so the sidecar positions describe the gold text the
  student sees.
- Each teacher token is attributed to the student token whose character
  span contains its midpoint (character-offset alignment).
- Top-k tokens absent from the student vocabulary are dropped and the
  teacher distribution is renormalized over what remains; positions with no
  usable token contribute nothing to the KD term.

### Limits

- **Vocabulary**: logit distillation is exact when student and teacher
  share a vocabulary (same family, e.g. Qwen teacher into Qwen student).
  Cross-family distillation still trains but the KD term weakens as
  out-of-vocab alternatives are dropped.
- **Storage**: the sidecar roughly doubles dataset size (k = 8 by default).
- **Packing**: rejected; the KD trainer needs per-position alignment.
- **Platform**: the KD dataset path bypasses the datasets builder cache and
  ships targets through the arrow table as JSON, so training runs on
  Windows as well as Linux.
- **Marked experimental**: the plumbing (sidecar, alignment, loss) is
  tested; the training-quality effect is not benchmarked. Compare
  `train task-eval` numbers against the non-KD run before trusting it.

## Abliteration (experimental)

Adjacent to distillation: instead of learning a refusal direction away,
remove it from the weights.

```bash
brainforge abliterate run --model Qwen/Qwen3-8B \
  --calibration calibration.jsonl --output models/abliterated
```

The refusal direction is the mean hidden-state difference between harmful
and benign calibration prompts at a 2/3-depth layer; every attention output
and MLP down projection is projected orthogonal to it. `ablation_stats.json`
reports the direction norm before and after, plus a suppression ratio.

The implementation is a naive single-direction ablation (fixed target
depth, mean-over-tokens direction). It is a heuristic, not a guarantee:
evaluate capability and refusal behavior yourself. Upgrades would be
sweeping the target layer per model or iterative optimization
(Heretic-style).
