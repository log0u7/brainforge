# Dataset format reference

One JSON object per line (`dataset.jsonl`), produced by pipeline runs and
consumed by validation, deduplication, splitting and training. Decision:
[ADR-0007](../adr/0007-dataset-format-messages-metadata.md).

## Record schema

```json
{
  "id": "case-cmd-injection-001",
  "domain": "security",
  "messages": [
    {"role": "user", "content": "Find vulnerabilities\n\n```\n...code...\n```"},
    {"role": "assistant", "content": "{\"verdict\": \"confirmed\", \"cwe\": \"CWE-78\", ...}"}
  ],
  "metadata": { }
}
```

| Field | Type | Rules |
|---|---|---|
| `id` | string | deterministic `case-<sha1[:10]>` from the case builder |
| `domain` | string | domain pack name |
| `messages` | list | first message `user`, last message `assistant`, non-empty contents |
| `messages[-1].content` | string | valid JSON: the canonical structured verdict |
| `metadata` | object | provenance, see below |

The student sees only `messages`. Metadata is for tooling and audit.

## `metadata` keys

| Key | Type | Description |
|---|---|---|
| `pipeline` | string | pipeline that produced the record |
| `domain` | string | domain pack |
| `language` | string \| null | case language |
| `source` | object | `{type, repository, commit, path, url, date}` from the case |
| `source_date` | string \| null | ISO date driving contamination tagging |
| `recitation_risk` | bool | true when pre-cutoff or unknown date/cutoff |
| `recitation_reason` | string \| null | `pre_cutoff_source`, `unknown_source_date`, `unknown_teacher_cutoff` |
| `teachers` | list | one entry per teacher role: `{role, kind, provider, model, family}` |
| `judge` | object \| null | same shape, absent for judge-less runs |
| `teacher_agreement` | object | role -> bool: teacher signal vs judge verdict |
| `rag` | object | `{used, chunks: [{source, doc_id, chunk_id, hash, score}]}` |
| `quality` | object | `{passed, reasons}` from the quality gate |
| `created_at` | string | ISO timestamp |

## Validation rules (`brainforge dataset validate`)

- record parses against the schema;
- first message is `user`, last is `assistant`, roles in
  {system, user, assistant}, no empty content;
- assistant content parses as JSON;
- metadata contains `source`, `teachers`, `quality`, `recitation_risk`;
- `metadata.quality.passed` is `true`.

Invalid records are listed with index, id and reasons; the command exits 1.

## Case files (input format)

```json
{
  "id": "optional-stable-id",
  "source": {"type": "git", "repository": "repo", "commit": "abc", "path": "tools/x.py", "date": "2026-08-10"},
  "input": {"code": "...", "description": "...", "question": "optional"},
  "metadata": {"language": "python"}
}
```

Plain source files (`.py`, `.c`, `.js`, ...) are auto-converted: code becomes
`input.code`, language comes from the extension, `source.date` is the file's
last git commit date, and the id is derived from source + content.

## Splits

`train prepare` / `dataset split` emit:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: source-grouped 80/10/10
  (seeded); one repository never spans two splits;
- `test_postcutoff.jsonl`: records with `recitation_risk: false` from all
  splits; the primary evaluation signal
  ([contamination](../explanation/contamination.md));
- `stats.json`: split sizes and the post-cutoff warning.
