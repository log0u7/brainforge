# Research and references

External work that shaped BrainForge's design, mapped to what it informs.
arXiv IDs are stable; the "informs" column points at the concrete code or
decision record that implements the idea.

## Research papers

| Source | Informs |
|---|---|
| [arXiv:2610.20401](https://arxiv.org/abs/2610.20401) Training Language Models to Fail Gracefully (Arora et al., 2026) | Graceful failure: the student learns a safe fallback (`insufficient_information`) when acting would be unreliable. See [ADR-0016](../adr/0016-graceful-failure-fallback.md) |
| [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., NeurIPS 2023) | LLM judges are usable but biased (position, verbosity, self-enhancement); motivates the independent judge, adversarial judge prompt, and agreement monitoring. See [ADR-0002](../adr/0002-multi-teacher-ensemble-independent-judge.md) and [judge correlation](../explanation/judge-correlation.md) |
| [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2023) | Agreement across independent samples as a reliability signal; the multi-teacher ensemble and per-teacher `teacher_agreement` metadata |
| [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) Detecting Pretraining Data from Large Language Models (Shi et al., 2024) | Contamination is measurable (Min-K% Prob) and benchmark test data leaks into pretraining; motivates `recitation_risk` tagging and the post-cutoff holdout. See [ADR-0003](../adr/0003-contamination-controls-postcutoff-holdout.md) and [contamination](../explanation/contamination.md) |
| [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) | 4-bit quantized LoRA training that fits a single RTX 3080 10GB. See [ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md) |
| [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) | The LoRA adapter mechanism used by the QLoRA trainer |
| [arXiv:2402.13116](https://arxiv.org/abs/2402.13116) A Survey on Knowledge Distillation of Large Language Models (Xu et al., 2024) | Taxonomy behind the optional teacher-to-student distillation path (algorithm, skill, verticalization). See [ADR-0015](../adr/0015-knowledge-distillation-sidecar.md) |
| [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) Distilling the Knowledge in a Neural Network (Hinton et al., 2015) | The original soft-target distillation formulation the KD temperature/alpha knobs implement |
| [arXiv:2309.07597](https://arxiv.org/abs/2309.07597) C-Pack: Packed Resources for General Chinese Embeddings / BGE (Xiao et al., SIGIR 2024) | The `BAAI/bge-small-en-v1.5` embedding model behind local RAG. See [ADR-0005](../adr/0005-rag-fastembed-numpy-sqlite.md) |
| [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (Agarwal et al., 2023) | Generalized Knowledge Distillation: the student trains on its own sequences against the teacher's full next-token distribution. The principled replacement candidate for the top-k sidecar, deferred. See [ADR-0018](../adr/0018-post-training-expansion-deferred.md) |
| [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024) | Group Relative Policy Optimization: the group of samples as its own baseline, no value model. The verifiable-reward stage deferred behind the benchmark. See [ADR-0018](../adr/0018-post-training-expansion-deferred.md) |

## Guides and blog posts

The 2026 Hugging Face agent-training material that shaped the post-training
map (five stages, same stack: TRL plus OpenEnv). Each entry points at the
stage it informs.

| Source | Informs |
|---|---|
| [Train the agent](https://huggingface.co/spaces/sergiopaniego/train-the-agent-slides) deck (S. Paniego, PyTorch Conf NA 2026) | The stage map itself: SFT on agent traces, distillation, GRPO, RL in environments, RL through the harness, and the "which one to reach for" decision table. See [ADR-0018](../adr/0018-post-training-expansion-deferred.md) |
| [Distillation in 2026](https://huggingface.co/blog/sergiopaniego/distillation-2026) | The distillation stage: on-policy GKD in TRL, teacher and student sharing a tokenizer |
| [Agentic RL in 2026](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) | The GRPO stage: a plain Python reward function over sampled completions |
| [RL environments in 2026](https://huggingface.co/blog/sergiopaniego/rl-environments-2026) | The environment stage: `GRPOTrainer` with an `environment_factory` over an OpenEnv client |
| [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) | The harness stage: capture proxy recording token ids and logprobs, Harbor tasks as verifiers, and single-seed honesty (results presented as observation, not proof) |
| [Agent traces on the Hub](https://huggingface.co/docs/hub/agent-traces) | SFT on recorded agent sessions (multi-turn, tool calls) with assistant-only loss |
| [Agent glossary](https://huggingface.co/blog/agent-glossary) | Shared vocabulary: the agent is a model plus a harness; same weights score differently across harnesses |
| [pi-mono sessions](https://huggingface.co/datasets/badlogicgames/pi-mono) | Raw agent-session JSONL convertible to chat messages, an example trace source for the agentic pack |

## Standards and tools

| Source | Used for |
|---|---|
| [Diátaxis](https://diataxis.fr/) | Documentation structure (tutorials, how-to, reference, explanation) enforced in `CONTRIBUTING.md` |
| [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) | `CHANGELOG.md` format |
| [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) | Version numbering |
| [MADR 3.0](https://adr.github.io/madr/) | Architecture decision record format (`docs/adr/`) |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | OpenAI-compatible provider layer (OpenRouter, Zen, MLGW, local servers) |
| [TRL](https://huggingface.co/docs/trl/) | `SFTTrainer` for supervised fine-tuning; `DistillationTrainer` and `GRPOTrainer` are the trainers of the deferred stages, see [ADR-0018](../adr/0018-post-training-expansion-deferred.md) |
| [OpenEnv](https://huggingface.co/docs/openenv) | Candidate environment interface (Gymnasium-style `reset`/`step` over Docker-isolated environments) for the deferred GRPO stage |
| [Unsloth](https://github.com/unslothai/unsloth) | Optional faster training backend (opt-in extra) |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4-bit/8-bit quantization |
| [fastembed](https://github.com/qdrant/fastembed) | ONNX CPU embeddings for local RAG |
| [Pydantic](https://docs.pydantic.dev/) | Config models, schemas, JSON Schema generation |

## Related

- [Explanation: architecture](../explanation/architecture.md): how the pieces fit.
- [Decision records](../adr/index.md): every choice with its trade-offs.
