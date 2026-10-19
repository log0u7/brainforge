# BrainForge

[![CI](https://github.com/log0u7/brainforge/actions/workflows/ci.yml/badge.svg)](https://github.com/log0u7/brainforge/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![code style](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)

**Every other fine-tuning tool trusts a single teacher. BrainForge never
does.**

BrainForge distills a multi-teacher LLM ensemble (independent teachers, an
adversarial critic, an independent judge) into a verified dataset, then
trains an open student model on it: QLoRA for a fast first pass, full
fine-tuning for the decision model, optional logit distillation, GGUF/Ollama
export to ship it. Domains are pluggable: `security` and `coding` are the
reference packs (where the rigor was proven), `ticket_triage`, `ctf`,
`code_review`, `security_audit` and `agentic` show non-security and
agentic-shaped domains, and any domain without a pack falls back to a generic
one.

Born from a security problem: you cannot trust one LLM's verdict, models
memorize CVEs, benchmarks lie. The machinery those problems forced (judge
independence enforced at config load, contamination tagging, post-cutoff
holdout) is domain-agnostic, and every new domain inherits it by default.
That is what makes a ticket-triage model trustworthy rather than "65%,
maybe".

## Quickstart: your first student in one command

```bash
git clone git@github.com:log0u7/brainforge.git && cd brainforge
mise install && uv sync
uv run brainforge config validate
```

Zero-cost dry run of the whole chain (mock provider, no GPU):

```bash
uv run brainforge quickstart run --domain ticket_triage \
  --input examples/cases --provider mock
```

```text
quickstart domain=ticket_triage pipeline=ticket_triage
dataset 3 accepted, 0 rejected (3 cases) -> datasets/ticket_triage.jsonl
preparing splits
quickstart failed: no CUDA device available; QLoRA training requires a GPU
```

The dataset step runs anywhere; training needs a CUDA GPU and the training
extra (`uv sync --extra training`). With real providers:

```bash
export OPENROUTER_API_KEY=... ZENCODE_API_KEY=...
uv run brainforge quickstart run --domain ticket_triage --input data/raw
```

Full walkthrough: [build your first model](docs/tutorials/first-model.md).

## How it works

```text
                     ┌──────────────────┐
                     │   LOCAL DATA     │
                     │ tickets / CVEs / │
                     │ code / docs      │
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │    LOCAL RAG     │  (business history,
                     └────────┬─────────┘   KB, tickets resolved)
                              ▼
                     ┌──────────────────┐
                     │   CASE BUILDER   │
                     └────────┬─────────┘
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌───────────┐     ┌───────────┐     ┌───────────┐
      │ teacher 1 │     │ teacher 2 │     │  critic   │
      │ (any open │     │ (any open │     │(devil's   │
      │  model)   │     │  model)   │     │ advocate) │
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
                     │  + kd.jsonl      │ (optional logits)
                     └────────┬─────────┘
              QLoRA ──▶ full FT ──▶ GGUF / Ollama
                              ▼
                     ┌──────────────────┐
                     │ STUDENT (open    │
                     │  model, local)   │
                     └──────────────────┘
```

## What you get

| Capability | Detail |
|---|---|
| Provider layer | OpenRouter, OpenCode Zen, MLGW, local servers (Ollama included), deterministic mock; sqlite cache, retries, usage logs with cost estimates |
| Multi-teacher ensembles | Independent teachers, adversarial critic, independent judge, cost modes (cheap / standard / maximum), teacher ablation by config |
| Judge independence | Enforced at config load: distinct provider and model family from every teacher |
| Contamination controls | Per-teacher `knowledge_cutoff`, source dates, `recitation_risk` tagging, post-cutoff holdout as the primary benchmark |
| Local RAG | fastembed (ONNX, CPU) or hashing embeddings, sqlite + numpy store, full chunk provenance |
| Quality gates | Per-domain rules (evidence, CWE format, confidence), rejected-case quarantine, exact + near-duplicate detection |
| Domain packs | `security` and `coding` reference packs, `ticket_triage`/`ctf`/`code_review`/`security_audit`/`agentic` example packs, generic fallback for anything else |
| Training | QLoRA (4-bit + LoRA, one RTX 3080 10GB), full fine-tuning, Unsloth backend (opt-in extra), optional knowledge distillation from teacher logprobs |
| Export | merged HF weights, quantized GGUF + Modelfile, `ollama create` registration; distribute GGUF with [llmp2p](https://github.com/log0u7/llmp2p) |
| Abliteration | experimental refusal-direction removal for uncensored/red-team models |
| Quickstart | `brainforge quickstart run --domain <pack>`: cases to trained student in one command |
| Tooling | `brainforge` CLI (typer + rich), Makefile, 267 tests, GitHub Actions CI and a manual GPU smoke job |

## Two-phase workflow

1. **Quick pass**: two teachers (any open models, local or API), a judge,
   one LoRA run. Minutes to a working model, e.g. 65% on ticket triage.
2. **Decision model**: index your business history in RAG, regenerate the
   dataset through the same verified pipeline, then `train run --strategy
   full` for the full fine-tune that lands the number that matters.

Between the two, `dataset inspect` tells you whether the gain is analysis
or recitation.

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

Any OpenAI-compatible endpoint works, including Ollama on
`http://localhost:11434/v1` via the `local` provider type.

## Documentation

| Section | Content |
|---|---|
| [Tutorial](docs/tutorials/first-model.md) | first student model in minutes, no keys |
| [How-to guides](docs/how-to/configure-providers.md) | providers, RAG, generation, dataset management, domain packs, shipping to Ollama, ADRs |
| [Reference](docs/reference/configuration.md) | configuration, providers, CLI, dataset format, quality gates, RAG, training, CI, research references |
| [Explanation](docs/explanation/architecture.md) | architecture, contamination, judge correlation, knowledge distillation |
| [Decision records](docs/adr/index.md) | 15 MADR ADRs behind the design |

## Development

```bash
make install     # uv sync
make test        # pytest (unit)
make lint        # ruff check + format check
make config      # config validate + schema check
```

Integration tests are opt-in: `BRAINFORCE_IT=1 uv run pytest -m integration`
(needs `uv sync --extra training`). GPU smoke: `make train-smoke-cpu` for the
CPU path, `make train-smoke` on a CUDA runner.

Contributions welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[ROADMAP](ROADMAP.md).

## License

[Apache-2.0](LICENSE).
