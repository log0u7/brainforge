# 0007 - Dataset format: messages plus separated metadata, RAG kept out of student inputs

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

Every accepted pipeline run must produce a record that (a) trains a chat
student directly, (b) carries full provenance for audits and reproduction, and
(c) does not leak teacher-side context into what the student learns to expect.
The same file must be loadable by TRL and readable by the validation and
deduplication tools.

## Decision Drivers

* Direct TRL/SFT compatibility: chat records, one JSON object per line.
* Provenance is mandatory: teachers, judge, RAG chunks, gate results, source
  dates.
* The student learns to analyze code it is *given*, not to query a retriever.
* Metadata must never pollute the training text.

## Considered Options

* Messages only (no metadata)
* Flat text (prompt/completion pair, metadata in a sidecar file)
* Messages + metadata in the same JSON object, RAG context excluded from
  `messages` (chosen)
* Messages including the RAG context (RAG-augmented student)

## Decision Outcome

Chosen option: "Messages + metadata in the same JSON object, RAG context
excluded from `messages`". A record is `{id, domain, messages, metadata}`;
`messages` holds exactly the user input (case, never RAG) and the assistant
canonical verdict (JSON string). `metadata` carries source, `source_date`,
`recitation_risk`, teacher/judge provenance, `teacher_agreement`, RAG chunk
provenance and the quality-gate result. Validators (`dataset validate`) reject
records missing provenance keys or with non-JSON assistant content.

### Consequences

* The student is trained on the same distribution it will see at inference:
  case in, structured analysis out, no retriever required.
* Auditing a record answers "who produced this, from what evidence" without
  touching logs.
* Downstream tools (dedup, splits, contamination stats) all read `metadata`
  instead of re-parsing text.
* If a RAG-augmented student is ever wanted, the change is additive (a second
  field or a build-time flag), not a format break.

### Confirmation

* `tests/unit/test_pipeline_dataset.py::test_build_record_shape` and the
  validation tests assert the exact shape.
* `tests/unit/test_cli.py::test_train_prepare` exports the split files for
  TRL from real pipeline output.
* The CI `config` job runs the mock pipeline and produces such a dataset.

## Pros and Cons of the Options

### Messages only

* Good, because it is minimal.
* Bad, because no provenance: un-auditable, and contamination tooling cannot
  work (ADR-0003 depends on metadata).

### Flat text + sidecar metadata

* Good, because legacy trainer compatibility.
* Bad, because two files drift apart; chat multi-message records and JSON
  verdicts fit awkwardly.

### Messages + metadata, RAG excluded (chosen)

* Good, because training and audit concerns are separated in one atomic line.
* Bad, because records are larger than raw text (acceptable for JSONL).

### RAG context inside `messages`

* Good, because the student could learn to exploit provided context.
* Bad, because it trains a retrieval-dependent student while the deployment
  target is a standalone 8B model, and it leaks teacher-side context into the
  training distribution.

## Links

* [Dataset format reference](../reference/dataset-format.md)
* ADR-0003 (recitation_risk lives in metadata), ADR-0005 (RAG provenance)
* `src/brainforge/dataset/writer.py`, `src/brainforge/dataset/validation.py`
