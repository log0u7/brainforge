# Generic framework positioning: security is the reference domain, not the identity

* Status: accepted
* Date: 2026-10-18

## Context and Problem Statement

BrainForge was built around security datasets. The user wants a generic,
easy-to-use framework: take any open model (HF / Unsloth / Ollama), distill
it, fine-tune it (QLoRA then full FT), optionally abliterate it, ship it.
Two narratives compete: keep the security-first identity and bolt generic
features on the side, or reposition the tool as a domain-agnostic framework
where security is one pack among others.

## Decision Drivers

* The under-the-hood machinery is already domain-agnostic: `GenericPack`
  fallback via `get_or_generic`, role-based config, the provider
  abstraction. The cost of repositioning is narrative plus example packs,
  not architecture.
* The verification machinery (judge independence, contamination tagging,
  post-cutoff holdout) is domain-agnostic in value: a ticket-triage
  decision model needs the same guarantees a security model needs.
* Existing security users must not lose anything.

## Considered Options

* Keep the security-first identity; generic features stay secondary.
* Pure generic repositioning: security demoted to a plain example pack,
  origin story erased.
* Reconciliation: security stays the reference domain pack (hardest ground
  truth, where the rigor was proven), the identity becomes "verified
  distillation framework for open models".

## Decision Outcome

Chosen option: "reconciliation", because it keeps the security pedigree as
a credibility asset for the generic pitch (the elevator line: every other
fine-tuning tool trusts a single teacher, BrainForge never does) while
every new domain inherits security-grade verification by default.

### Consequences

* Positive: generic domains get judge independence and contamination
  controls for free; the security pack keeps working unchanged.
* Negative: two narratives must be kept coherent in docs and marketing.
* Mitigation: `docs/reference/security-pack.md` documents security as the
  reference pack; the README carries the origin paragraph and the pivot.

### Confirmation

* `tests/unit/test_docs_coverage.py` fails when a CLI command is
  undocumented (the drift this repositioning had to fix).
* `ticket_triage` proves the framework runs non-security domains
  (`tests/unit/test_ticket_triage_pack.py`).
* `brainforge quickstart run --domain ticket_triage` chains the whole flow.

## Pros and Cons of the Options

### Security-first identity

* Good, because zero repositioning risk for existing users.
* Bad, because the generic framework the user asked for stays a silent
  fallback; the narrative debt is paid later anyway.

### Pure generic, origin erased

* Good, because simplest single narrative.
* Bad, because it throws away the credibility the security pedigree buys.

### Reconciliation

* Good, because the origin story justifies the rigor generically.
* Bad, because the docs must hold both stories at once.

## Links

* [Architecture](../explanation/architecture.md)
* [ADR-0002](0002-multi-teacher-ensemble-independent-judge.md)
* [ADR-0006](0006-modular-domain-packs.md)
