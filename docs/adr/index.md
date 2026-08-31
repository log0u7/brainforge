# Architecture Decision Records

This directory records the significant architectural decisions of BrainForge,
one file per decision, in [MADR 3.0](https://adr.github.io/madr/) format.

## Rules

1. **Numbered, immutable**: files are `NNNN-short-title.md`; numbers are never
   reused. A reversed decision gets a new ADR that `supersedes` the old one,
   and the old file's status changes to `superseded by ...`.
2. **One decision per file**: keep the scope narrow. Infrastructure trivia and
   routine changes do not need an ADR.
3. **Include the rejected options**: the value of an ADR is the trace of what
   was *not* chosen and why. Fill the "Pros and Cons of the Options" section
   honestly.
4. **Confirmation is mandatory**: every ADR points at the tests, CLI command or
   config field that proves the decision is still honored.
5. **English only**, no em dashes.

## Workflow

1. Copy `template.md` to `NNNN-short-title.md` with the next free number.
2. Fill every section; keep it under ~150 lines.
3. Add the entry to the table below.
4. Open the pull request with the `docs` label.

## Index

| Number | Title | Status |
|---|---|---|
| [0001](0001-json-config-pydantic-schema.md) | JSON configuration validated by pydantic-generated JSON Schema | accepted |
| [0002](0002-multi-teacher-ensemble-independent-judge.md) | Multi-teacher ensemble with an independent judge | accepted |
| [0003](0003-contamination-controls-postcutoff-holdout.md) | Contamination controls and the post-cutoff holdout | accepted |
| [0004](0004-provider-abstraction-openai-sdk.md) | Provider abstraction on the OpenAI SDK with schema-in-prompt structured outputs | accepted |
| [0005](0005-rag-fastembed-numpy-sqlite.md) | Local RAG with fastembed embeddings and a numpy/sqlite store | accepted |
| [0006](0006-modular-domain-packs.md) | Modular domain packs with a generic fallback | accepted |
| [0007](0007-dataset-format-messages-metadata.md) | Dataset format: messages plus separated metadata, RAG kept out of student inputs | accepted |
| [0008](0008-tooling-uv-mise-src-layout.md) | Tooling: uv pinned by mise, src layout, ruff and pytest | accepted |
| [0009](0009-sqlite-response-cache.md) | sqlite response cache keyed on the full request | accepted |
| [0010](0010-qlora-training-single-rtx-3080.md) | Student training: QLoRA 4-bit on a single RTX 3080 10GB | accepted |
