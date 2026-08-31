# Architecture

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

## Data flow

1. **Cases** are built from JSON case files or plain code files
   (`dataset/case_builder.py`). Every case carries a `CaseSource` with a
   `date` used for contamination tagging.
2. **RAG** (optional): the engine queries the local index with the case
   description; retrieved chunks are formatted into teacher prompts and their
   provenance recorded.
3. **Steps**: for each role in the pipeline, the engine resolves the role
   binding, picks the domain pack's output schema and prompt (teacher / critic
   / judge), and calls the provider with structured-output parsing.
4. **Judge**: the final step (when present) receives the case, RAG context and
   all teacher analyses, and produces the canonical verdict.
5. **Quality gate**: the domain pack's gate checks the canonical record
   (evidence, CWE format, confidence threshold, reasoning). Unresolved verdicts
   (`insufficient_information`) and optionally `recitation_risk` records are
   rejected.
6. **Dataset**: accepted records are appended to `datasets/<pipeline>.jsonl`;
   rejected ones land in `data/rejected/<pipeline>/<case>.json` with reasons.

## Layering rules

- `pipeline/` depends on `roles/`, `domains/`, `providers/`, `dataset/` - never
  the reverse.
- `domains/` must not import provider SDKs; it only sees provider results as
  dicts.
- `dataset/` is import-light: the engine imports the writer lazily to keep the
  module graph acyclic.

## Roles in one sentence each

- **RAG**: external context injected into teacher prompts, never into student
  training inputs.
- **Teacher**: generates a structured supervision signal for one domain angle.
- **Critic** (general teacher): independent counter-analysis; hunts
  contradictions and missing information.
- **Judge**: canonical verdict; must re-derive conclusions from evidence.
- **Dataset**: verified training knowledge with full provenance.
- **Student**: the small local model trained on that dataset (phase 2).
