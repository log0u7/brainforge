#!/usr/bin/env python3
"""Assemble the MR11 SFT dataset.

Two filters act at the source:
1. FLAG-VERIFIED: only mining records whose outcome actually landed the
   flag (metadata.flag_verified) - a recited flag is not a success.
2. FACTS-FIRST (levier 2, anti-regression): only trajectories whose FIRST
   assistant call is the ``facts`` action. The measured facts-skipping
   rate was learned from the data; the filter acts at the source.

The mr10 base (247 rows) passes through the same facts-first filter
retroactively: rows that skip the facts step are dropped, not
re-generated. Records are deduplicated by id (mining wins over base).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="the mining's trajectories dir (per-seed JSONL)",
    )
    parser.add_argument(
        "--base", type=Path, default=Path("datasets/sft-mr10-facts/train.jsonl")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("datasets/sft-mr11-facts/train.jsonl")
    )
    args = parser.parse_args()

    kept: list[dict] = []
    seen_ids: set[str] = set()
    mining_kept = dropped_flag = dropped_facts = dropped_dup = 0
    files = sorted(args.trajectories.glob("*.jsonl"))
    print(f"mining files: {len(files)}")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rid = record.get("id") or ""
            if not (record.get("metadata") or {}).get("flag_verified"):
                dropped_flag += 1
                continue
            if not first_action_is_facts(record):
                dropped_facts += 1
                continue
            if rid in seen_ids:
                dropped_dup += 1
                continue
            seen_ids.add(rid)
            record.setdefault("metadata", {})["mr11_source"] = "teacher-mining"
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
            rid = record.get("id") or ""
            if rid in seen_ids:
                dropped_dup += 1
                continue
            seen_ids.add(rid)
            record.setdefault("metadata", {})["mr11_source"] = "mr10-base"
            kept.append(record)
            base_kept += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for record in kept:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"mining: kept {mining_kept} (dropped: {dropped_flag} unflagged,"
        f" {dropped_dup} duplicate ids)"
    )
    print(f"base:   {base_kept}/{base_in} rows kept (facts-first filter)")
    print(f"mr11 total: {len(kept)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
