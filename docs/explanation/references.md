# Method references

Every method paper below is anchored to the place in the repository that
implements it, so a citation is a map, not a reading list.

The flat inventory of every external source (papers, guides, standards)
lives in [Research and references](../reference/references.md); this page
tells the per-method story for the methods that drive training.

## QLoRA: 4-bit finetuning on a single GPU

Paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
(Dettmers et al., 2023). The method backpropagates through a frozen base
model quantized to 4-bit NF4, shrinks the quantization constants again with
double quantization, and trains LoRA adapters on top.

Where it lives:

- `src/brainforge/training/qlora.py`: `_quantization_config()` builds a
  `BitsAndBytesConfig` with `bnb_4bit_quant_type="nf4"` and
  `bnb_4bit_use_double_quant=True` (an 8-bit fallback path exists).
- The hardware budget this targets is recorded in
  [ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md): one RTX
  3080 10GB.
- Usage and flags: [training reference](../reference/training.md).

## On-policy distillation (GKD) and the sidecar loss

Paper: [On-Policy Distillation of Language Models: Learning from
Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) (Agarwal et al.,
ICLR 2024). GKD trains the student on its own generated sequences with
teacher feedback, using a generalized loss that blends a student-teacher
distribution term with the ordinary gold-label term.

Where it lives, honestly split in half:

- The loss half is implemented, offline: `src/brainforge/training/kd.py`
  trains with `alpha * T^2 * CE(student_T, teacher top-k) +
  (1 - alpha) * CE(gold)`, fed by the `<dataset>.kd.jsonl` sidecar of
  teacher top-k logprobs, aligned cross-tokenizer by character midpoint.
  Decision: [ADR-0015](../adr/0015-knowledge-distillation-sidecar.md).
- The on-policy half (student-sampled sequences through TRL
  `DistillationTrainer`) is deferred, not implemented: see
  [ADR-0018](../adr/0018-post-training-expansion-deferred.md) and
  [ROADMAP](../../ROADMAP.md) Phase 3.

## GRPO with verifiable rewards

Paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024),
which introduced Group Relative Policy Optimization: sample a group of
answers per prompt, use the group mean as its own baseline, and train
against a reward, with no value model.

Where it lives:

- `src/brainforge/training/rewards.py`: the verifiable-reward registry.
  `flag_reward()` scores an exact expected CTF flag at 1.0, a valid flag
  format at 0.3, anything else at 0; it is registered as `ctf_flag`, and
  `compose_reward()` supports weighted specs such as
  `ctf_flag*1.0,compliance*0.5`.
- `src/brainforge/training/grpo.py`: wraps TRL `GRPOTrainer`, wired to the
  CLI as `strategy: "grpo"`.
- Status: shipped before its benchmark (ADR-0018 amendment,
  2026-10-28); the measured CTF experiment is still Phase 3 in
  [ROADMAP](../../ROADMAP.md).

## The post-training curriculum (Hugging Face)

Guide: [A curriculum for post-training agents](https://huggingface.co/blog/huggingface/train-agents)
(Hugging Face, 2026): five post-training stages for agents, each mapped to
a TRL trainer: SFT on agent traces, distillation, RL, RL environments, then
harnesses and scale.

Where it lives: not a module, a plan.

- [ROADMAP](../../ROADMAP.md) is the stage map. Phase 2 covers the SFT
  stage (QLoRA or full fine-tuning on the gate-checked dataset); Phase 3
  lists the remaining stages: the on-policy distillation backend, the GRPO
  experiment, an OpenEnv CTF environment, and multi-harness RL through a
  capture proxy.
- [ADR-0018](../adr/0018-post-training-expansion-deferred.md) records the
  same five-stage map and the selective-deferral decision (GRPO pulled
  forward by the 2026-10-28 amendment, the rest deferred).

## Perturbation Probing: the abliteration roadmap

Paper: [Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN
Behavioral Circuits in Aligned LLMs](https://arxiv.org/abs/2604.27401)
(Liu, Li and Wu, 2026). Two forward passes per prompt map FFN behavioral
circuits without backpropagation; opposition circuits appear where RLHF
suppresses a pretraining tendency, and neuron-level ablation is surgical
(on Qwen3.5-2B, ablating 20 neurons removes multi-turn sycophantic
capitulation).

Where it lives, honestly: as a roadmap, not in the code.

- `src/brainforge/training/abliterate.py` implements the cruder cousin:
  one refusal direction, the mean hidden-state difference between harmful
  and benign calibration prompts at a fixed 2/3-depth layer (Arditi et al.
  2024 style), projected orthogonal to every `self_attn.o_proj` and
  `mlp.down_proj`. Decision:
  [ADR-0014](../adr/0014-abliteration-single-direction.md).
- The paper is the upgrade path for that feature: circuit-targeted,
  neuron-level ablation instead of one global direction, the same intent
  as the module comment ("sweep the target layer per model,
  multi-direction or Heretic-style iterative optimization"). Nothing from
  the paper's method is implemented yet, and the single-direction result
  stays a heuristic: evaluate capability and refusal behavior yourself.
