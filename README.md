# BrainForge

[![pipeline status](https://gitlab.com/6admin.io/brainforge/badges/main/pipeline.svg)](https://gitlab.com/6admin.io/brainforge/-/pipelines)
[![coverage](https://gitlab.com/6admin.io/brainforge/badges/main/coverage.svg)](https://gitlab.com/6admin.io/brainforge/-/pipelines)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![code style](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![docs](https://img.shields.io/badge/docs-mkdocs%20material-526CFE)](https://gitlab.com/6admin.io/brainforge/-/pages)

**Turn real-world data into verified training datasets with multi-teacher LLM
pipelines, then train small local models on them.**

BrainForge builds security and coding datasets from real material (vulnerable
code, CVEs, patches, git history, documentation), verifies every example
through independent LLM teachers and a judge, and fine-tunes a 7-9B student
locally with QLoRA on a single RTX 3080 10GB. The engine is domain-agnostic:
specializations plug in as *domain packs* (security and coding ship today).

## Quickstart

```bash
git clone git@gitlab.com:6admin.io/brainforge.git && cd brainforge
mise install && uv sync
```

Validate the shipped configuration, then generate a dataset at zero cost with
the mock provider:

```bash
uv run brainforge config validate
uv run brainforge pipeline run security_dataset --input examples/cases --provider mock
```

```text
security_dataset: 3 accepted, 0 rejected (3 cases) -> datasets/security_dataset.jsonl
```

Inspect what was produced, then export training splits:

```bash
uv run brainforge dataset inspect datasets/security_dataset.jsonl
uv run brainforge train prepare datasets/security_dataset.jsonl
```

Point the teachers at real providers by exporting `OPENROUTER_API_KEY` and
`ZENCODE_API_KEY` and dropping `--provider mock`. Full walkthrough:
[build your first dataset](docs/tutorials/first-dataset.md).

## How it works

```text
                    ┌──────────────────┐
                    │   LOCAL DATA     │
                    │ CVE / Git / Code │
                    │ Docs / Markdown  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │    LOCAL RAG     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │   CASE BUILDER   │
                    └────────┬─────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
     ┌───────────┐     ┌───────────┐     ┌───────────┐
     │ OpenRouter│     │ OpenCode  │     │   MLGW    │
     │ security  │     │    Zen    │     │  general  │
     │  teacher  │     │  coding   │     │  critic   │
     └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
           └─────────────────┼─────────────────┘
                             ▼
                    ┌──────────────────┐
                    │  JUDGE (distinct │
                    │ provider/family) │
                    └────────┬─────────┘
                        QUALITY GATE
                             ▼
                    ┌──────────────────┐
                    │   dataset.jsonl  │
                    │  + provenance    │
                    └────────┬─────────┘
                         QLoRA SFT
                             ▼
                    ┌──────────────────┐
                    │ STUDENT 7-9B     │
                    │  local GPU       │
                    └──────────────────┘
```

## What you get

| Capability | Detail |
|---|---|
| Provider layer | OpenRouter, OpenCode Zen, MLGW, local servers, deterministic mock; sqlite cache, retries, usage logs with cost estimates |
| Multi-teacher ensembles | Independent teachers, adversarial judge, cost modes (cheap / standard / maximum), teacher ablation by config |
| Judge independence | Enforced at config load: distinct provider and model family from every teacher |
| Contamination controls | Per-teacher `knowledge_cutoff`, source dates, `recitation_risk` tagging, post-cutoff holdout as the primary benchmark |
| Local RAG | fastembed (ONNX, CPU) or hashing embeddings, sqlite + numpy store, full chunk provenance |
| Quality gates | Per-domain rules (evidence, CWE format, confidence), rejected-case quarantine, exact + near-duplicate detection |
| Datasets | JSONL with chat messages + separated provenance metadata, source-grouped splits, TRL-ready export |
| Training prep | Validate / split / export; QLoRA 4-bit scaffolding for phase 2 on one RTX 3080 10GB |
| Tooling | `brainforge` CLI (typer + rich), Makefile, 112 tests, GitLab CI with docs and a manual GPU smoke job |

## Principles

1. **Separation of concerns**: provider ≠ model ≠ role ≠ pipeline; the
   pipeline never knows a vendor exists.
2. **Independent verification**: the judge shares neither provider nor model
   family with any teacher, because erroneous agreement is the dangerous
   failure mode.
3. **Contamination awareness**: the student is benchmarked on cases the
   teachers cannot have memorized; gains must mean analysis, not recitation.
4. **Provenance everywhere**: teachers, judge, RAG chunks and gate results
   travel with every record.
5. **Quality over volume**: verified, deduplicated, source-grouped data beats
   mass-produced synthetic examples.

## Providers

| Provider | Type | Role in the default config |
|---|---|---|
| [OpenRouter](https://openrouter.ai) | `openrouter` | security + coding teachers |
| [OpenCode Zen](https://opencode.ai/docs/zen/) | `zen` | judge |
| MLGW (local gateway) | `mlgw` | general critic |
| Mock | `mock` | zero-cost demos and tests |

## Documentation

| Section | Content |
|---|---|
| [Tutorial](docs/tutorials/first-dataset.md) | first dataset in 10 minutes, no keys |
| [How-to guides](docs/how-to/configure-providers.md) | providers, RAG, generation, dataset management, domain packs, ADRs |
| [Reference](docs/reference/configuration.md) | configuration, providers, CLI, dataset format, quality gates, RAG, training, CI |
| [Explanation](docs/explanation/architecture.md) | architecture, contamination, judge correlation |
| [Decision records](docs/adr/index.md) | 10 MADR ADRs behind the design |

## Development

```bash
make install     # uv sync
make test        # pytest (unit)
make lint        # ruff check + format check
make config      # config validate + schema check
make docs-serve  # local documentation
```

Contributions welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[ROADMAP](ROADMAP.md) (phase 2: QLoRA training and the golden dataset).

## License

[Apache-2.0](LICENSE).
