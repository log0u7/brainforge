# BrainForge documentation

BrainForge turns local cases into a gate-checked, provenance-tagged JSONL
dataset with a multi-teacher LLM pipeline. It can prepare QLoRA or full
fine-tuning data, train an open student on the best detected device
(CUDA/ROCm, XPU, MPS, CPU), and export it to GGUF or Ollama. Knowledge
distillation is experimental. This site is organized by what you are trying
to do.

Start with the [README quickstart](../README.md#quickstart-no-keys-no-gpu) for
the first no-key run, or read [status and known limits](../README.md#status-and-known-limits)
before planning a deployment.

## Pick your entry point

| I want to... | Go to |
|---|---|
| prepare a dataset without API keys or a GPU | [README quickstart](../README.md#quickstart-no-keys-no-gpu) |
| follow the guided dataset path | [Tutorial: from cases to a prepared dataset](tutorials/first-model.md) |
| train a student on any device | [Training reference](reference/training.md) and [Hardware reference](reference/hardware.md) |
| connect OpenRouter, Zen or MLGW | [How-to: configure providers](how-to/configure-providers.md) |
| index my own documents for retrieval | [How-to: index and search with RAG](how-to/index-and-search-rag.md) |
| generate a dataset with real teachers | [How-to: generate a dataset](how-to/generate-dataset.md) |
| validate, inspect or split a dataset | [How-to: manage datasets](how-to/manage-datasets.md) |
| add a new domain (tickets, CTF, review...) | [How-to: add a domain pack](how-to/add-domain-pack.md) |
| ship a trained model to Ollama | [How-to: ship to Ollama](how-to/ship-to-ollama.md) |
| look up a CLI command or flag | [CLI reference](reference/cli.md) |
| look up a config option | [Configuration reference](reference/configuration.md) |
| look up the record format | [Dataset format](reference/dataset-format.md) |
| understand the quality gate | [Quality gates](reference/quality-gates.md) |
| understand contamination tagging | [Contamination](explanation/contamination.md) |
| understand *why* it is built this way | [Architecture](explanation/architecture.md) |
| read the design decisions and trade-offs | [Decision records](adr/index.md) |
| trace the research behind a design choice | [Research and references](reference/references.md) |
| check what is not supported or not tested | [Status and known limits](../README.md#status-and-known-limits) |

## The system in one paragraph

Cases built from local data (code, CVEs, tickets, documentation) are enriched
with local RAG context, analyzed by independent teacher LLMs, and arbitrated by
a judge. By default the judge must use a different provider from every non-judge
role, plus a different declared model family when both sides declare one. A
quality gate filters the results into a provenance-complete JSONL dataset.
Source dates and teacher knowledge cutoffs drive recitation-risk tagging;
post-cutoff cases form a holdout split, not a benchmark. Prepared splits feed a
student fine-tune on the best detected device (QLoRA first, full FT for the
decision model), exported to GGUF/Ollama and distributable with llmp2p.
Knowledge distillation is
experimental. Dataset-only users can stop before training. Security and coding
are the reference packs; other domains inherit the same machinery. Verified
means gate-checked, not human-reviewed ground truth.

## Documentation map

```text
Tutorials    learning-oriented: guided first run
How-to       task-oriented: solve one concrete problem
Reference    information-oriented: exact facts, tables
Explanation  understanding-oriented: the why and the trade-offs
ADR          decision records: what was chosen, what was rejected
```

Start with the README quickstart, then the tutorial; use Reference to look
things up; read Explanation when a design choice surprises you. Most pages link to their reference and
explanation backing; ADRs include confirmation commands or tests where
available.
