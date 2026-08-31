# BrainForge

Agentic tooling for building security & coding datasets from real data with
multi-teacher LLM pipelines, and training small local language models on
consumer GPUs.

BrainForge turns real-world material (vulnerable code, CVEs, patches, git
history, documentation) into verified training examples, then measures whether
that knowledge actually improves a small local student model (7-9B, QLoRA on a
single RTX 3080 10GB).

Security is the first domain, coding the second. The engine is generic: any
specialization (audit, pentest, code review, other trades) plugs in as a
**domain pack**.

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
     │           │     │  teacher  │     │           │
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

## Core principles

- **Provider ≠ Model ≠ Role ≠ Pipeline**: providers describe access, models are
  declared once, roles assign a function, pipelines orchestrate roles. The
  pipeline never knows about providers.
- **Multi-teacher verification**: independent teachers analyze each case, a
  judge (on a *different* provider and model family) produces the canonical
  verdict, and a quality gate rejects weak records.
- **Contamination awareness**: every case carries a source date; every teacher
  declares a knowledge cutoff. Cases predating the cutoffs are tagged
  `recitation_risk`, and evaluation is primarily run on the **post-cutoff
  holdout**, so gains reflect analysis rather than memorized CVEs.
- **Judge independence**: `config validate` enforces that the judge does not
  share a provider or model family with any teacher, because erroneous
  agreement is the dangerous failure mode, not disagreement.
- **Provenance everywhere**: every record keeps its teachers, judge, RAG chunks
  (source, hash, score) and quality-gate result.
- **Quality over volume**: deduplication (exact + near-dup), confidence
  thresholds, and rejected-case quarantine in `data/rejected/`.

## Quickstart

Requirements: Python 3.12+, [mise](https://mise.jdx.dev) (or uv directly).

```bash
mise install                 # installs the pinned uv
uv sync                      # install dependencies into .venv

# validate the shipped example configuration
uv run brainforge config validate

# generate a dataset without spending a cent (mock provider)
uv run brainforge pipeline run security_dataset --input examples/cases --provider mock

# inspect it: contamination stats, teacher/judge agreement, duplicates
uv run brainforge dataset inspect datasets/security_dataset.jsonl

# validate, split (train/validation/test + post-cutoff holdout)
uv run brainforge dataset validate datasets/security_dataset.jsonl
uv run brainforge train prepare datasets/security_dataset.jsonl
```

Point the teachers at real providers:

```bash
export OPENROUTER_API_KEY="sk-or-..."     # security + coding teachers
export ZENCODE_API_KEY="..."              # judge (OpenCode Zen)
uv run brainforge pipeline run security_dataset --input examples/cases
```

MLGW (`http://localhost:8080/v1`) serves local models; its development API key
is the intentionally fake `deadbeef`.

## Configuration

JSON only, validated against `config/schema.json` (generated from pydantic
models). See [docs/configuration.md](docs/configuration.md). Secrets live in
environment variables, never in git: `${VAR}` or `${VAR:default}` placeholders
are expanded at load time.

## Documentation

Full docs live in [`docs/`](docs/) (mkdocs-material):

- [Architecture](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [Providers](docs/providers.md) - OpenRouter, OpenCode Zen, MLGW, mock, local
- [Domain packs](docs/domain-packs.md) - security, coding, writing your own
- [RAG](docs/rag.md)
- [Dataset format & contamination](docs/dataset.md)
- [Training](docs/training.md)
- [CLI reference](docs/cli.md)
- [CI](docs/ci.md)

Build them locally with `uv run mkdocs serve`.

## Development

```bash
make install    # uv sync
make test       # pytest (unit)
make lint       # ruff check + format check
make config     # validate config + schema check
make docs-serve # local docs
```

Contributions welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[ROADMAP](ROADMAP.md).

## Status

`0.0.0` - MVP. Dataset generation pipeline is complete and tested; QLoRA
training is scaffolded (see ROADMAP phase 2). Apache-2.0 licensed.
