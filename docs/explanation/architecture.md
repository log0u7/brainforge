# Architecture

BrainForge turns local, real-world material into verified training examples
and trains small local models on them. This page explains how the components
fit together and why the boundaries are where they are.

## Component map

```text
src/brainforge/
├── case.py           Case / CaseSource / CaseInput models
├── config/           pydantic config models, loader, env expansion, schema export
├── providers/        Provider abstraction, adapters, cache, observability
├── models/registry   model name -> ModelDef + provider instance
├── roles/registry    role name -> RoleBinding (kind, model, provider)
├── domains/          domain packs: schemas, prompts, quality gates
├── pipeline/         engine, context, judge helpers
├── rag/              loaders, chunking, embeddings, vector store, retrieval
├── dataset/          case builder, writer, validation, gate, dedup, split
├── training/         prepare/export, QLoRA stub, GPU smoke test
└── experiments/      run directories, metrics, reports
```

## The data flow, end to end

1. **Cases** (`dataset/case_builder.py`). JSON case files or plain source
   files become `Case` objects. Every case carries a `CaseSource` whose `date`
   powers contamination tagging (see [contamination](contamination.md)).
2. **RAG context** (`rag/`). The engine queries the local index with the case
   description; retrieved chunks are formatted into teacher prompts and their
   provenance copied into the record. Context is generation-only: it never
   reaches student training inputs.
3. **Steps** (`pipeline/engine.py`). For each role, the engine resolves the
   role binding (kind, model, provider), asks the domain pack for the output
   schema and prompt, and calls the provider through the cache-wrapped
   abstraction.
4. **Judge**. The final judge role receives case, RAG context and all teacher
   analyses, then produces the canonical structured verdict after an
   adversarial re-derivation pass.
5. **Quality gate**. The domain pack's gate checks evidence, CWE format,
   confidence and reasoning; unresolved verdicts and (optionally)
   `recitation_risk` records are rejected.
6. **Dataset**. Accepted records append to `datasets/<pipeline>.jsonl`;
   rejected ones land in `data/rejected/<pipeline>/<case>.json` with reasons.
7. **Splits and training**. `train prepare` validates, splits source-grouped
   and exports TRL-ready JSONL, including the post-cutoff holdout.

## The separation that matters: Provider ≠ Model ≠ Role ≠ Pipeline

| Layer | Answers | Example |
|---|---|---|
| Provider | *How do we reach an LLM?* | `openrouter`, `zen`, `mlgw`, `mock` |
| Model | *Which model, declared once?* | `security` -> deepseek-v4-pro |
| Role | *What function does it serve?* | `security_teacher`, `judge` |
| Pipeline | *Which roles, in which order?* | `security_dataset` |

The pipeline knows role names only. Swapping `deepseek-v4-pro` for another
model is a config edit; no code knows a vendor exists. This is what makes
teacher ablation (ADR-0002) a config change instead of a refactor.

## Layering rules

- `pipeline/` depends on `roles/`, `domains/`, `providers/`, `dataset/`;
  nothing depends back on the pipeline.
- `domains/` never imports vendor SDKs: providers return validated dicts
  (see [provider ADR](../adr/0004-provider-abstraction-openai-sdk.md)).
- `dataset/` stays import-light; the engine imports the writer lazily to keep
  the module graph acyclic.
- `training/` reads datasets; it never generates them.

## Roles in one sentence each

- **RAG**: external context injected into teacher prompts, never into student
  training inputs.
- **Teacher**: produces a structured supervision signal for one domain angle.
- **Critic** (general teacher): independent counter-analysis hunting
  contradictions and missing information.
- **Judge**: canonical verdict; must re-derive conclusions from evidence.
- **Dataset**: verified training knowledge with full provenance.
- **Student**: the small local model trained on that dataset.

## Design invariants

1. RAG is context, teachers generate supervision, the judge validates, the
   dataset is training knowledge, the student is the trained model. Never
   conflate these roles.
2. Judge independence is enforced at config load, not hoped for.
3. Source dates are mandatory for evaluation claims; post-cutoff cases are
   the benchmark signal.
4. Provenance travels with every record.
5. Quality beats volume: gates, deduplication and rejected-case quarantine are
   the default.
