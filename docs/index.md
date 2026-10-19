# BrainForge documentation

BrainForge distills a multi-teacher LLM ensemble into a verified dataset
and trains an open student model on it (QLoRA, full fine-tuning, optional
logit distillation, GGUF/Ollama export). This site is organized by what you
are trying to do.

## Pick your entry point

| I want to... | Go to |
|---|---|
| try the full loop in 10 minutes, no API keys | [Tutorial: build your first model](tutorials/first-model.md) |
| connect OpenRouter, Zen or MLGW | [How-to: configure providers](how-to/configure-providers.md) |
| index my own documents (business history, KB) for retrieval | [How-to: index and search with RAG](how-to/index-and-search-rag.md) |
| generate a dataset with real teachers | [How-to: generate a dataset](how-to/generate-dataset.md) |
| validate, inspect or split a dataset | [How-to: manage datasets](how-to/manage-datasets.md) |
| add a new domain (tickets, CTF, review...) | [How-to: add a domain pack](how-to/add-domain-pack.md) |
| ship a trained model to Ollama | [How-to: ship to Ollama](how-to/ship-to-ollama.md) |
| look up a config option, CLI flag or record field | [Reference](reference/configuration.md) |
| look up the research papers and standards behind the design | [References](reference/references.md) |
| understand *why* it is built this way | [Explanation](explanation/architecture.md) |
| read the design decisions and their trade-offs | [Decision records](adr/index.md) |

## The system in one paragraph

Cases built from local data (code, CVEs, tickets, documentation) are
enriched with local RAG context, analyzed by independent teacher LLMs, and
arbitrated by a judge that shares neither provider nor model family with
them. A quality gate filters the results into a provenance-complete JSONL
dataset, with contamination tagging driven by teacher knowledge cutoffs and
a post-cutoff holdout as the primary benchmark. Prepared splits feed a
student fine-tune (QLoRA first, full FT for the decision model), exported
to GGUF/Ollama and distributable with llmp2p. Security is the reference
domain where this rigor was proven; every other domain inherits it.

## Documentation map

```text
Tutorials    learning-oriented: guided first run
How-to       task-oriented: solve one concrete problem
Reference    information-oriented: exact facts, tables
Explanation  understanding-oriented: the why and the trade-offs
ADR          decision records: what was chosen, what was rejected
```

Start with the tutorial; use Reference to look things up; read Explanation
when a design choice surprises you. Every page is cross-linked: each how-to
points at its reference and explanation backings, and every ADR lists the
tests that confirm it still holds.
