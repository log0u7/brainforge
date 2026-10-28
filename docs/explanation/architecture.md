# Architecture

BrainForge distills a multi-teacher LLM ensemble into a verified dataset
and trains an open student model on it. This page explains how the
components fit together and why the boundaries are where they are.

## Component map

```text
src/brainforge/
├── case.py           Case / CaseSource / CaseInput models
├── config/           pydantic config models, loader, env expansion, schema export
├── providers/        Provider abstraction, adapters, cache, observability, logprobs
├── models/registry   model name -> ModelDef + provider instance
├── roles/registry    role name -> RoleBinding (kind, model, provider)
├── domains/          domain packs: schemas, prompts, quality gates
│                     (security + coding reference packs, ticket_triage,
│                      ctf, code_review, security_audit, agentic, generic)
├── pipeline/         engine, context, judge helpers, KD sidecar capture
├── rag/              loaders, chunking, embeddings, vector store, retrieval
├── dataset/          case builder, writer, validation, gate, dedup, split
├── training/         prepare, hardware plan resolution, preflight, qlora
│                     (trl), unsloth backend, kd trainer, abliteration,
│                     gguf export, smoke probes
└── experiments/      run directories, metrics, reports
```

## The data flow, end to end

1. **Cases** (`dataset/case_builder.py`). JSON case files or plain source
   files become `Case` objects. Every case carries a `CaseSource` whose `date`
   powers contamination tagging (see [contamination](contamination.md)).
2. **RAG context** (`rag/`). The engine queries the local index with the case
   description; retrieved chunks are formatted into teacher prompts and their
   provenance copied into the record. Context is generation-only: it never
   reaches student training inputs. Index your business history here
   (resolved tickets, KB articles) to feed phase 2 of the workflow.
3. **Steps** (`pipeline/engine.py`). For each role, the engine resolves the
   role binding (kind, model, provider), asks the domain pack for the output
   schema and prompt, and calls the provider through the cache-wrapped
   abstraction. In KD mode the judge call additionally requests per-token
   top-k logprobs.
4. **Judge**. The final judge role receives case, RAG context and all teacher
   analyses, then produces the canonical structured verdict after an
   adversarial re-derivation pass.
5. **Quality gate**. The domain pack's gate checks evidence, CWE format,
   confidence and reasoning; unresolved verdicts and (optionally)
   `recitation_risk` records are rejected.
6. **Dataset**. Accepted records append to `datasets/<pipeline>.jsonl`
   (plus `datasets/<pipeline>.kd.jsonl` in KD mode); rejected ones land in
   `data/rejected/<pipeline>/<case>.json` with reasons.
7. **Splits and training**. `train prepare` validates, splits source-grouped
   and exports TRL-ready JSONL, including the post-cutoff holdout.
8. **Training** (`training/`). `train check` prefights hardware, model
   compatibility and memory without downloading weights; `train run`
   resolves a device plan (backend, device, dtype, quantization) then
   dispatches by backend (`trl`/`unsloth`) and strategy (`qlora`/`full`),
   or `--kd` for distillation; `train export` merges and converts to
   GGUF/Ollama ([KD details](knowledge-distillation.md),
   [hardware reference](../reference/hardware.md)).

## The separation that matters: Provider ≠ Model ≠ Role ≠ Pipeline

| Layer | Answers | Example |
|---|---|---|
| Provider | *How do we reach an LLM?* | `openrouter`, `zen`, `mlgw`, `mock`, `local` (Ollama) |
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
- `training/` reads datasets; it never generates them. Heavy dependencies
  (torch, trl, peft, transformers, bitsandbytes, unsloth) are imported
  lazily inside functions so the base package and the unit suite run
  without the training extra.
- `providers/` is the only path to an LLM API; logprobs ride the same
  abstraction (`structured_logprobs` over `complete_logprobs`).

## Roles in one sentence each

- **RAG**: external context injected into teacher prompts, never into student
  training inputs.
- **Teacher**: produces a structured supervision signal for one domain angle.
- **Critic** (general teacher): independent counter-analysis hunting
  contradictions and missing information.
- **Judge**: canonical verdict; must re-derive conclusions from evidence.
  In KD mode its token-level distribution is the distillation signal.
- **Dataset**: verified training knowledge with full provenance.
- **Student**: the local model trained on that dataset, exported to
  Ollama or distributed as GGUF.

## Design invariants

1. RAG is context, teachers generate supervision, the judge validates, the
   dataset is training knowledge, the student is the trained model. Never
   conflate these roles.
2. Judge independence is enforced at config load, not hoped for.
3. Source dates are mandatory for evaluation claims; post-cutoff cases are
   the benchmark signal.
4. Provenance travels with every record.
5. Quality beats volume: gates, deduplication and rejected-case quarantine
   are the default.
6. Security is the reference domain, not the identity: the
   contamination/judge-independence machinery was proven there and every
   other domain inherits it.
