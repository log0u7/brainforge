# 0006 - Modular domain packs with a generic fallback

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

BrainForge is not a security-only tool: the same generate-verify-export loop
applies to coding, pentest reporting, code review and future trades. The
domain-specific part is exactly the part that changes per trade (output
schemas, prompts, gate rules), while the engine (providers, roles, RAG,
dataset) is shared. The architecture must let new trades plug in without
touching the core.

## Decision Drivers

* Security and coding are MVP domains; pentest, review and others are planned.
* Adding a domain must not require engine changes (Open/Closed).
* Output schemas and quality gates are code (typed, testable), not free text.
* The engine must never branch on `if domain == "security"`.

## Considered Options

* Monolithic security-only implementation
* Prompts and schemas as loose config files only
* Separate Python packages per domain (entry points)
* Domain packs in-repo with a registry and a generic fallback (chosen)

## Decision Outcome

Chosen option: "Domain packs in-repo with a registry and a generic fallback".
`domains/base.py` defines the `DomainPack` interface (output schemas per role
kind, system/teacher/critic/judge prompts, quality gate, student input,
mode-prescribed steps). `domains/registry.py` resolves a pipeline's `domain`
to a pack; unknown names fall back to a `GenericPack` (free-form schema, basic
gate), so new trades can be prototyped in config alone and hardened later.

### Consequences

* Security and coding specifics (CWE checks, verdict enums, evidence
  requirements) live in one place each, fully unit-tested.
* The generic fallback can produce weakly-gated records; it is a prototyping
  tool, and serious datasets should use a dedicated pack.
* Packs live in-repo for MVP; the interface is designed so extraction into
  separate packages remains possible later without engine changes.

### Confirmation

* `tests/unit/test_domains.py` asserts pack registration, schemas, mode
  presets and every gate rule.
* `tests/unit/test_pipeline_dataset.py` runs the engine end-to-end through the
  security pack without any domain-specific branch in `pipeline/`.

## Pros and Cons of the Options

### Monolithic security-only

* Good, because the MVP ships faster.
* Bad, because "coding" already exists in the plan and every new trade would
  fork the engine.

### Config-only prompts and schemas

* Good, because no Python is needed for a new trade.
* Bad, because JSON Schema strings are unvalidated, gates become pseudo-code
  in config, and tests cannot cover domain semantics.

### Separate packages per domain (entry points)

* Good, because third parties could ship packs.
* Bad, because packaging and release overhead is unjustified while the pack
  set is two; the interface keeps the door open.

### In-repo packs + registry + fallback (chosen)

* Good, because adding a domain is one directory, one registration line and
  tests; the engine stays domain-blind.
* Bad, because the fallback pack can look like a real domain; mitigated by
  docs steering serious use to dedicated packs.

## Links

* [Domain pack how-to](../how-to/add-domain-pack.md)
* ADR-0007 (what a pack emits), ADR-0003 (gates and contamination interact)
* `src/brainforge/domains/`
