# Domain packs

A domain pack packages everything that is specific to one specialization:
output schemas, prompts, recommended pipeline steps and the quality gate. The
core engine stays generic; packs register by name and are selected per pipeline
via `domain`.

Bundled packs: `security` and `coding`, plus a `generic` fallback (used
automatically for unknown domains) so new trades can be tested with prompts
only, then hardened into a real pack.

## Pack interface

`src/brainforge/domains/base.py`:

```python
class DomainPack(ABC):
    name: str
    description: str
    recommended_steps: dict[Mode, list[str]]

    def output_schema(self, kind: RoleKind) -> type[BaseModel]: ...
    def system_prompt(self, role_name: str, kind: RoleKind) -> str: ...
    def teacher_prompt(self, role_name, case, rag_context) -> str: ...
    def critic_prompt(self, role_name, case, rag_context, previous_results) -> str: ...
    def judge_prompt(self, case, rag_context, previous_results) -> str: ...
    def quality_gate(self, kind, data, min_confidence=0.5) -> GateResult: ...
    def student_input(self, case) -> str: ...
    def steps_for_mode(self, mode) -> list[str]: ...
```

## Security pack

Teacher output (`SecurityTeacherAnalysis`): summary, `vulnerability_found`,
`vulnerability_type`, `cwe`, `severity`, confidence, preconditions,
exploitability, false-positive likelihood, evidence list, reasoning.

Judge output (`SecurityJudgeVerdict`): `verdict`
(`confirmed` / `rejected` / `insufficient_information`), CWE, severity,
confidence, evidence, reasoning, remediation, `disagreement`.

Gate rules (selection): a claimed vulnerability requires evidence; a CWE must
match `CWE-\d{1,4}`; a confirmed verdict requires a CWE and evidence; reasoning
is mandatory; confidence must clear `min_confidence`.

Mode presets: `cheap` = security teacher + judge; `standard` = + coding
teacher; `maximum` = + general critic.

## Coding pack

Teacher output (`CodingTeacherAnalysis`): summary, behavior explanation,
dataflow notes, issues, remediation, patch suggestion, confidence, reasoning.
Judge output (`CodingJudgeVerdict`): verdict, issues, reasoning, remediation,
confidence, disagreement. The gate rejects confirmed verdicts without issues
and teacher analyses without a behavior explanation.

## Writing a new pack

1. Create `src/brainforge/domains/<name>/` with `schemas.py` and `pack.py`
   (subclass `DomainPack`).
2. Register it in `src/brainforge/domains/registry.py`.
3. Define the roles it needs in `config.json` and reference the pack:
   `"pipelines": {"my_pipeline": {"domain": "<name>"}}`.
4. Add unit tests covering the schemas and gate rules (see
   `tests/unit/test_domains.py`).

Rules of thumb: schemas must be strictly validated by pydantic; prompts must
demand evidence traced to the provided code; the judge prompt must include the
adversarial re-derivation instruction.
