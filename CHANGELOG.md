# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions CI mirroring the GitLab pipeline: ruff lint, gitleaks secret
  scan over full history, unit tests with coverage, config validation plus
  mock pipeline run, and package build published as a 7-day artifact.
- Manual `GPU smoke` workflow (`workflow_dispatch`) targeting a self-hosted
  runner with the `gpu` label for the training smoke test.
- Documentation restructured following Diataxis: guided tutorial, six how-to
  guides, eight reference pages (including new quality-gates and
  dataset-format references), and three explanation pages (architecture,
  contamination, judge correlation).
- Architecture Decision Records in `docs/adr/` (MADR 3.0): index, blank
  template, and ten decision records covering configuration, ensemble and
  judge independence, contamination controls, provider abstraction, RAG,
  domain packs, dataset format, tooling, cache and training targets.
- Coverage badge: GitLab coverage regex on the `test` job.

### Changed

- README rewritten: badges, 30-second quickstart with expected output,
  feature table, principles, provider overview and Diataxis docs routing.
- mkdocs navigation reorganized into Tutorials / How-to guides / Reference /
  Explanation / Decision records; Pages job builds with `--strict`.

### Removed

- Legacy flat doc pages (`docs/architecture.md`, `docs/configuration.md`,
  `docs/providers.md`, `docs/domain-packs.md`, `docs/rag.md`,
  `docs/dataset.md`, `docs/training.md`, `docs/cli.md`, `docs/ci.md`)
  migrated into the new structure.

## [0.0.0] - 2026-08-31

Initial MVP release.

### Added

- JSON configuration system with pydantic models, `${VAR:default}` environment
  expansion, JSON Schema export/check, and enforced judge independence
  (judge provider and model family must differ from every teacher).
- Provider layer: generic OpenAI-compatible provider with retries, timeouts and
  structured-output parsing; OpenRouter, OpenCode Zen (`chat_completions`
  style), MLGW, local (Ollama-compatible) adapters; deterministic mock provider
  with generic JSON-Schema instance generation; sqlite response cache keyed on
  provider/model/role/messages/params/RAG context.
- Observability: per-call JSONL usage logs (tokens, latency, retries, cache
  hits, estimated cost from optional pricing config).
- Model and role registries with kind inference (teacher/critic/judge).
- Domain packs: `security` (CWE/severity/evidence schemas, security teacher,
  coding teacher, adversarial judge, quality gates) and `coding` (code analysis
  schemas); generic fallback pack for future domains.
- Pipeline engine: role steps, pipeline context, cost modes (cheap/standard/
  maximum), optional teacher ablation via explicit steps, unresolved-verdict
  rejection, `recitation_risk` tagging from case source dates vs teacher
  knowledge cutoffs.
- Local RAG: text/markdown/code/JSON loaders, markdown-aware chunker, fastembed
  (ONNX, CPU) embeddings with deterministic hashing fallback backend, sqlite +
  numpy vector store with full retrieval provenance.
- Dataset tooling: case builder (JSON cases and plain code files, git commit
  dates), JSONL writer with teacher/judge/RAG provenance, record validation,
  quality gate, exact + near-duplicate detection (Jaccard shingles),
  source-grouped train/validation/test splitting plus post-cutoff holdout.
- Training preparation: validate/split/export for TRL; QLoRA, evaluate and
  export stubs (phase 2); GPU smoke module for the manual CI job.
- Experiments: dated run directories with config snapshot, dataset copy,
  metrics (contamination stats, teacher/judge agreement rates) and a generated
  markdown report including correlated-bias warnings.
- CLI (`brainforge`): `config`, `models`, `roles`, `teacher`, `pipeline`,
  `rag`, `dataset`, `train` command groups with rich output.
- Documentation: README, CONTRIBUTING, ROADMAP, mkdocs-material site.
- CI: GitLab pipeline with lint (ruff), secrets (gitleaks), tests, config
  validation, build, mkdocs Pages, and a manual GPU-tagged training smoke job.
- Example cases spanning a teacher knowledge cutoff (command injection,
  SQL injection, stored XSS) and a working example configuration.
