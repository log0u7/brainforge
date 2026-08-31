# BrainForge

BrainForge builds verified training datasets from real data using multi-teacher
LLM pipelines, then trains small local models (7-9B QLoRA) on them.

Start with the [architecture overview](architecture.md), the
[configuration reference](configuration.md), or run the five-minute mock
pipeline from the [README](https://gitlab.6admin.io/6admin.io/brainforge).

## Design invariants

1. **RAG is context, teachers generate supervision, the judge validates, the
   dataset is training knowledge, the student is the trained model.** Never
   conflate these roles.
2. **The pipeline knows roles, not providers.** Providers, models, roles and
   pipelines are separate configuration layers.
3. **Judge independence is enforced.** Erroneous agreement between correlated
   models is the dangerous failure mode.
4. **Source dates are mandatory for evaluation claims.** Post-cutoff cases are
   the primary benchmark signal; pre-cutoff cases are tagged
   `recitation_risk`.
5. **Provenance is part of the data.** Teachers, judge, RAG chunks and gate
   results travel with every record.
