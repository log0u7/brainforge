# Dataset format and safeguards

## Record shape (`dataset.jsonl`)

```json
{
  "id": "case-cmd-injection-001",
  "domain": "security",
  "messages": [
    {"role": "user", "content": "Find vulnerabilities\n\n```\n...code...\n```"},
    {"role": "assistant", "content": "{\"verdict\": \"confirmed\", \"cwe\": \"CWE-78\", ...}"}
  ],
  "metadata": {
    "pipeline": "security_dataset",
    "domain": "security",
    "language": "python",
    "source": {"type": "git", "repository": "example-tools", "commit": "a1b2c3d", "date": "2026-08-10"},
    "source_date": "2026-08-10",
    "recitation_risk": false,
    "recitation_reason": null,
    "teachers": [{"role": "security_teacher", "kind": "teacher", "provider": "openrouter", "model": "deepseek-v4-pro", "family": "deepseek"}],
    "judge": {"role": "judge", "kind": "judge", "provider": "zen", "model": "glm-5.2", "family": "glm"},
    "teacher_agreement": {"security_teacher": true},
    "rag": {"used": false, "chunks": []},
    "quality": {"passed": true, "reasons": []},
    "created_at": "2026-08-31T15:20:00+00:00"
  }
}
```

The assistant content is the canonical structured verdict (JSON). Metadata is
kept out of the training text: the student sees only `messages`.

## Case builder

Cases come from JSON files (`source` + `input` + `metadata`) or plain source
files. For plain files BrainForge queries `git log` for the commit date. A
missing date is treated conservatively: the record is tagged `recitation_risk`
with reason `unknown_source_date`.

## Contamination: recitation vs analysis

Commercial teacher models have likely memorized public CVEs and their patches.
Left unchecked, the pipeline would harvest recitations and the student would
learn to recite, not analyze. Mechanisms:

- Each teacher model declares `knowledge_cutoff` (`YYYY-MM`).
- Every case carries a source date; a record is tagged `recitation_risk: true`
  when its date precedes the latest teacher cutoff (`pre_cutoff_source`),
  when a teacher lacks a cutoff (`unknown_teacher_cutoff`), or when the date is
  unknown.
- Tagging never silently discards data; `reject_recitation_risk: true` opts
  into exclusion at build time.
- `dataset split` emits `test_postcutoff.jsonl`: cases strictly after the
  latest cutoff. This is the primary evaluation signal, and it warns when fewer
  than 20 post-cutoff cases exist.
- `dataset inspect` and experiment reports show contamination ratios so
  base-vs-student gains can be interpreted honestly.

## Judge correlation

The plan's disagreement handling covers teachers contradicting each other; the
more dangerous failure is **erroneous agreement**: a teacher and judge sharing
a provider or model family can confirm each other's shared blind spots.

- `config validate` enforces judge independence by default: the judge role must
  use a different provider *and* a different declared `family` from every other
  role. Opt out per config with `judge_independence: warn | off`.
- Every record stores `teacher_agreement`; experiment reports flag rates >= 95%
  as correlated-bias warnings.
- The judge prompt includes an adversarial pass: re-derive conclusions from the
  evidence, refuse any teacher claim that is not traceable to the case.

## Quality gate and deduplication

Generic checks: schema-valid messages, assistant content is JSON, provenance
keys present, gate passed. Domain packs add their own rules (evidence, CWE
format, confidence). Unresolved verdicts (`insufficient_information`) are
rejected by default (`reject_unresolved`).

Deduplication combines exact sha256 on normalized text with near-duplicate
Jaccard similarity over 5-word shingles (threshold 0.85). Splits are grouped by
source (repository/path), so one repository never appears in both train and
test.

## Splits

```bash
uv run brainforge train prepare datasets/security_dataset.jsonl
# -> train.jsonl, validation.jsonl, test.jsonl, test_postcutoff.jsonl, stats.json
```
