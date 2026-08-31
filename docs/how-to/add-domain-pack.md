# How to add a domain pack

This guide adds a new specialization (for example `pentest` or `code-review`)
as a domain pack. It assumes familiarity with the pipeline concepts
([architecture](../explanation/architecture.md)) and that tests are part of
the work.

## 1. Decide the pack's contract

Before code, write down for your domain:

- what a **teacher** produces (analysis fields, severity scale, evidence);
- what the **judge** produces (verdict enum, required fields when confirmed);
- which gate rules reject a bad record;
- the three **mode presets** (cheap / standard / maximum step lists).

Security and coding packs are the reference implementations
(`src/brainforge/domains/security/`, `src/brainforge/domains/coding/`).

## 2. Create the pack directory

```text
src/brainforge/domains/<name>/
├── __init__.py      # exports a `pack` instance
├── schemas.py       # pydantic output models
└── pack.py          # the DomainPack subclass
```

## 3. Write the schemas

Pydantic models, one per role kind. Conventions that keep the mock provider
and gates useful:

- a `summary` (teacher) / `reasoning` (judge) field;
- a `confidence: float` with `ge=0, le=1`;
- enums for verdicts and severities;
- evidence as a list of a dedicated model;
- optional fields default to `None`.

## 4. Implement the DomainPack

Subclass `DomainPack` from `src/brainforge/domains/base.py`:

- `output_schema(kind)` -> the schema for teacher / critic / judge;
- `system_prompt(role_name, kind)` -> includes the adversarial judge
  instruction ("re-derive conclusions from the evidence, refuse untraceable
  claims");
- `teacher_prompt`, `critic_prompt`, `judge_prompt` -> use the shared
  formatters (`format_case`, `format_rag_context`, `format_previous_results`);
- `quality_gate(kind, data, min_confidence)` -> the rules from step 1;
- `student_input(case)` -> the user message the student will train on (case
  only, never RAG context);
- `recommended_steps` -> the three mode presets, using role names.

## 5. Register it

In `src/brainforge/domains/registry.py`:

```python
from brainforge.domains.pentest.pack import PentestPack

register(PentestPack())
```

## 6. Wire the configuration

Add the roles and pipeline to `config/config.json`:

```json
"pipelines": {
  "pentest_dataset": {"domain": "pentest", "mode": "standard"}
}
```

The judge-independence rule applies to your pipeline exactly as to the
others. Re-export the schema:

```bash
uv run brainforge config schema --write
```

## 7. Test it

- Unit tests for the schemas and every gate rule (see
  `tests/unit/test_domains.py` for the pattern).
- An end-to-end mock run: `brainforge pipeline run pentest_dataset --input
  examples/cases --provider mock`.
- Gate-rejection cases (a confirmed verdict without required fields must be
  rejected).

## 8. Document it

Add the pack to the [quality gates reference](../reference/quality-gates.md)
and, if it introduces new metadata semantics, to the
[dataset format reference](../reference/dataset-format.md). If the pack
changes an architectural assumption, write an ADR
([how-to: write an ADR](write-an-adr.md)).

## Checklist

- [ ] Schemas validated by pydantic, tested
- [ ] Judge prompt contains the adversarial pass
- [ ] Gate rules implemented and unit-tested
- [ ] Mode presets use existing role names
- [ ] Registered + schema re-exported + `config validate` green
- [ ] Mock pipeline produces accepted records
- [ ] Docs updated
