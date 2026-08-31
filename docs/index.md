# BrainForge documentation

BrainForge turns local, real-world material into verified training datasets
using multi-teacher LLM pipelines, then trains small local models (7-9B QLoRA)
on them. This site is organized by what you are trying to do.

## Pick your entry point

| I want to... | Go to |
|---|---|
| try the full loop in 10 minutes, no API keys | [Tutorial: build your first dataset](tutorials/first-dataset.md) |
| connect OpenRouter, Zen or MLGW | [How-to: configure providers](how-to/configure-providers.md) |
| index my own documents for retrieval | [How-to: index and search with RAG](how-to/index-and-search-rag.md) |
| generate a dataset with real teachers | [How-to: generate a dataset](how-to/generate-dataset.md) |
| validate, inspect or split a dataset | [How-to: manage datasets](how-to/manage-datasets.md) |
| add a new specialization (pentest, review...) | [How-to: add a domain pack](how-to/add-domain-pack.md) |
| look up a config option, CLI flag or record field | [Reference](reference/configuration.md) |
| understand *why* it is built this way | [Explanation](explanation/architecture.md) |
| read the design decisions and their trade-offs | [Decision records](adr/index.md) |

## The system in one paragraph

Cases built from local data (code, CVEs, patches) are enriched with local RAG
context, analyzed by independent teacher LLMs, and arbitrated by a judge that
shares neither provider nor model family with them. A quality gate filters the
results into a provenance-complete JSONL dataset, with contamination tagging
driven by teacher knowledge cutoffs and a post-cutoff holdout as the primary
benchmark. Prepared splits feed a QLoRA fine-tune of a small local student.

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
