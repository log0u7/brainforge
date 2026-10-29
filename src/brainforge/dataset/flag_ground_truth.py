"""Stamp verified-tier ground-truth flags into dataset records.

The CTF pipeline verifies teacher flags against
``metadata.expected_flag`` (the gate) but the stored record's assistant
JSON keeps ``flag: null``: the ground truth never reaches the training
targets or the holdout, so ``flag_accuracy`` is uncomputable. This
module copies the case's expected flag into the record's assistant JSON
and tags ``metadata.flag_verified`` for provenance.
"""

import json

from brainforge.errors import BrainforgeError


def enrich_flags(records: list[dict], cases: list[dict]) -> list[dict]:
    """Copy ``expected_flag`` from verified-tier cases into record answers.

    Tier-B cases (no ``expected_flag`` string) are skipped: their flag is
    format-only. A record that already carries a different flag is a data
    defect and raises instead of being silently overwritten.
    """
    by_id = {case.get("id"): case for case in cases}
    out: list[dict] = []
    for record in records:
        case = by_id.get(record.get("id"))
        expected = case.get("metadata", {}).get("expected_flag") if case else None
        expected_flag = expected.strip() if isinstance(expected, str) else ""
        if not expected_flag:
            out.append(record)
            continue
        messages = record.get("messages") or []
        if not messages:
            raise BrainforgeError(f"record '{record.get('id')}' has no messages")
        content = messages[-1].get("content", "")
        try:
            data = json.loads(content[content.find("{") :])
        except (ValueError, TypeError) as exc:
            raise BrainforgeError(
                f"record '{record.get('id')}' has unparseable assistant JSON: {exc}"
            ) from exc
        current = data.get("flag")
        if current and current != expected_flag:
            raise BrainforgeError(
                f"record '{record.get('id')}' claims flag {current!r} but the case "
                f"ground truth is {expected_flag!r}; refusing to overwrite"
            )
        if not current:
            data["flag"] = expected_flag
            messages[-1]["content"] = json.dumps(data, ensure_ascii=False, indent=2)
        metadata = dict(record.get("metadata") or {})
        metadata["flag_verified"] = True
        record = {**record, "messages": messages, "metadata": metadata}
        out.append(record)
    return out
