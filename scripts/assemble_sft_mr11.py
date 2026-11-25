#!/usr/bin/env python3
"""Assemble the MR11 SFT dataset: the teacher-mined trajectories plus
the facts-first-filtered mr10 base.

Two filters act at the source:
1. FLAG-VERIFIED: only mining records whose outcome actually landed the
   flag (metadata.flag_verified - the flag was read from the live
   target, so a win is execution-verified).
2. FACTS-FIRST (levier 2, anti-regression): only trajectories whose
   FIRST assistant call is the ``facts`` action. The measured
   facts-skipping rate was learned from the data; the filter acts at
   the source. Applied to the mining records AND retroactively to the
   mr10 base (rows that skip the facts step are dropped, not
   re-generated).

The mining records pass through the mr6 provenance enrichment (the
trainer's validation gate requires source/teachers/quality/
recitation_risk) and the mr6 class-fact gate (a uaf claim is only
legal on a uaf-class target). Records are deduplicated on the exact
content (the historical convention keeps same-id rows: the class
balance relies on them).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import assemble_sft_mr6 as mr6


def first_action_is_facts(record: dict) -> bool:
    """The first assistant message must be the facts call itself."""
    for message in record.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        try:
            parsed = json.loads(message.get("content") or "")
        except ValueError:
            return False
        return isinstance(parsed, dict) and parsed.get("action") == "facts"
    return False


def uaf_class_legal(record: dict) -> bool:
    """The mr6 class-fact gate: a uaf claim only on a uaf-class target."""
    rid = str(record.get("id", ""))
    if any(marker in rid for marker in mr6.UAF_CLASS_TARGET_MARKERS):
        return True
    for message in record.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        try:
            payload = json.loads(content)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("action") == "facts":
            for predicate in (payload.get("action_input") or {}).get("predicates") or []:
                if predicate.startswith(mr6.UAF_CLASS_FACT):
                    return False
        if mr6.UAF_CLASS_FACT in content:
            return False
    return True


def record_key(record: dict) -> str:
    """The exact-content dedup key: id + messages. Same-id distinct runs
    both survive (the class balance relies on them); only IDENTICAL
    trajectories against the same target collapse."""
    return json.dumps({"id": record.get("id"), "messages": record.get("messages")}, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="the mining's trajectories dir (per-seed JSONL)",
    )
    parser.add_argument("--base", type=Path, default=Path("datasets/sft-mr10-facts/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("datasets/sft-mr11-facts/train.jsonl"))
    args = parser.parse_args()

    kept: list[dict] = []
    seen: set[str] = set()
    mining_kept = dropped_flag = dropped_facts = dropped_class = dropped_dup = 0
    if not args.trajectories.is_dir():
        raise SystemExit(f"--trajectories must be a directory: {args.trajectories}")
    files = sorted(args.trajectories.glob("*.jsonl"))
    print(f"mining files: {len(files)}")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not (record.get("metadata") or {}).get("flag_verified"):
                dropped_flag += 1
                continue
            if not first_action_is_facts(record):
                dropped_facts += 1
                continue
            if not uaf_class_legal(record):
                dropped_class += 1
                continue
            record = mr6._fill_empty_messages(record)
            record = mr6._enrich_metadata(record)
            record.setdefault("metadata", {})["mr11_source"] = "teacher-mining"
            key = record_key(record)
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            kept.append(record)
            mining_kept += 1

    base_in = base_kept = 0
    if args.base.exists():
        for line in args.base.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            base_in += 1
            if not first_action_is_facts(record):
                dropped_facts += 1
                continue
            if not uaf_class_legal(record):
                dropped_class += 1
                continue
            record = mr6._fill_empty_messages(record)
            record.setdefault("metadata", {})["mr11_source"] = "mr10-base"
            key = record_key(record)
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            kept.append(record)
            base_kept += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for record in kept:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"mining: kept {mining_kept} (dropped: {dropped_flag} unflagged,"
        f" {dropped_class} class-fact, {dropped_dup} identical)"
    )
    print(f"base:   {base_kept}/{base_in} rows kept (facts-first filter)")
    print(f"mr11 total: {len(kept)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
