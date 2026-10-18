# The security domain pack (reference)

The `security` pack is the reference implementation of a BrainForge domain
pack: the hardest ground truth (verifiable CWE identifiers, exploitability,
post-cutoff holdout), and where the contamination and judge-independence
machinery was proven. It is a domain pack like any other, not the product
identity; `ticket_triage` and the generic fallback show the framework is
domain-agnostic.

## What it provides

| Piece | Content |
|---|---|
| Teacher schema | `SecurityTeacherAnalysis`: vulnerability flag, CWE, severity, preconditions, exploitability, false-positive likelihood, evidence, reasoning |
| Judge verdict | `SecurityJudgeVerdict`: confirmed / rejected / insufficient_information, CWE, severity, evidence, reasoning, remediation, disagreement |
| Quality gate | confidence floor; confirmed verdicts require CWE + evidence; teacher vulnerability claims require evidence; CWE format enforced (`CWE-\d{1,4}`); missing reasoning rejected |
| Recommended steps | cheap: security teacher + judge; standard: + coding teacher; maximum: + general critic |
| Student input | the case question (or description) plus fenced code |

## Pipelines in the shipped config

| Pipeline | Domain | Mode |
|---|---|---|
| `security_dataset` | security | standard |
| `security_dataset_maximum` | security | maximum |
| `coding_dataset` | coding | explicit steps |

## Evaluation specifics

`train task-eval` scores verdict + CWE accuracy against judge ground truth
and is restricted to the security domain (the metric only exists for its
schema). All domains can still use `train evaluate` (loss + perplexity)
and the post-cutoff holdout.

## The generic fallback

Any unregistered domain name gets `GenericPack` via `get_or_generic`:
confirmed / rejected / insufficient_information verdicts, generic analysis
schema, confidence-gated quality gate. Use it for a first pass, then write
a dedicated pack when the schema matters (see
[add a domain pack](../how-to/add-domain-pack.md)).
