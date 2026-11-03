#!/usr/bin/env python3
"""Assemble the MR6 SFT dataset from its four sources.

Sources (all JSONL, chat-record shape):

  - the trails mined from the benchmark run dirs
    (``scripts/build_sft_trails.py`` output);
  - the fresh seed-swept rollout trajectories
    (``scripts/run_rollouts.py`` output);
  - the decision-verified fact records
    (``scripts/build_verified_facts.py`` output);
  - the replay corpora (the MR1-3 teacher records) - the anti-forgetting
    mix that the tiny 6-record SFT lacked.

Contracts enforced here (the MR5 failure modes, made structural):

  1. CONTAMINATION: any record whose ID carries a clean-protocol eval
     challenge marker is dropped AND counted; the CLI exits nonzero
     when one was found, so a contaminated training file cannot pass
     unnoticed.
  2. EMPTY-SHARE CAP: the MR5 dataset was 4/6 empty completions and
     trained a model that says nothing; the empty share is capped
     against the kept total.
  3. PER-TARGET CAP: 40 seeds per target is useful variety, but one
     target must not dominate the mix.

Run:  python scripts/assemble_sft_mr6.py \
        --source datasets/sft-trails/train.jsonl \
        --source-glob 'experiments/rollouts/trajectories/*.jsonl' \
        --source datasets/sft-verified-facts/train.jsonl \
        --source datasets/ctf-distill.jsonl \
        --out datasets/sft-mr6/train.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# The 11 clean-protocol eval challenges, as ID markers: any record ID
# containing one of these strings is eval-derived (the ID namespaces
# differ - ctf-lictor-*, lab-*, sft-*- - but the substring survives the
# translation, which is exactly the PR #124 lesson applied in reverse).
EVAL_ID_MARKERS = (
    "callme-002",
    "fmtstr-002",
    "orw-002",
    "ret2func-002",
    "ret2libc-001",
    "ret2libc-002",
    "ret2plt-002",
    "shellcode-002",
    "srop-001",
    "uaf_heap",
)


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def assemble(
    sources: list[Path],
    out: Path,
    per_target_cap: int,
    empty_completion_share: float,
) -> dict:
    """Merge, gate, cap, write. Returns the summary the CLI prints."""
    records: list[dict] = []
    contaminating = 0
    for path in sources:
        for record in _load(path):
            rid = str(record.get("id", ""))
            if any(marker in rid for marker in EVAL_ID_MARKERS):
                contaminating += 1
                continue
            records.append(record)

    # per-target cap: the record ID is the target key (the mined and
    # rollout records carry lab-<target>-001, the fact records carry
    # sft-verified-facts-<target>)
    counts: dict[str, int] = {}
    capped: list[dict] = []
    for record in records:
        key = str(record.get("id", ""))
        if counts.get(key, 0) >= per_target_cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        capped.append(record)

    # empty-share cap: non-empty records are kept in full, empties are
    # admitted only up to the share of the kept total
    def _empty(record: dict) -> bool:
        messages = record.get("messages") or []
        return not (messages and (messages[-1].get("content") or "").strip())

    non_empty = [r for r in capped if not _empty(r)]
    empties = [r for r in capped if _empty(r)]
    # e <= share * (n + e)  =>  e <= share * n / (1 - share); a share
    # of 1.0 means empties are not capped, and an all-empty dataset is
    # refused (that is the MR5 failure mode, not a dataset)
    if empty_completion_share >= 1.0:
        max_empties = len(empties)
    elif not non_empty:
        max_empties = 0
    else:
        max_empties = int(empty_completion_share * len(non_empty) / (1 - empty_completion_share))
    kept_empties = empties[:max_empties]
    final = non_empty + kept_empties

    if final:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final),
            encoding="utf-8",
        )
    return {
        "records": len(final),
        "contaminating": contaminating,
        "empties_dropped": len(empties) - len(kept_empties),
        "per_target": counts,
    }


def _expand(source_args: list[str]) -> list[Path]:
    """``--source`` takes a path or a glob pattern (the rollout
    trajectories arrive as one JSONL per seed)."""
    paths: list[Path] = []
    for arg in source_args:
        matches = sorted(glob.glob(arg))
        paths.extend(Path(m) for m in matches) if matches else paths.append(Path(arg))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="JSONL path or glob pattern (repeatable)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-target-cap", type=int, default=24)
    parser.add_argument("--empty-completion-share", type=float, default=0.2)
    args = parser.parse_args()

    sources = _expand(args.source)
    summary = assemble(
        sources=sources,
        out=args.out,
        per_target_cap=args.per_target_cap,
        empty_completion_share=args.empty_completion_share,
    )
    print(
        f"[assemble] records: {summary['records']}, contaminating dropped: "
        f"{summary['contaminating']}, empties dropped: {summary['empties_dropped']}"
    )
    for key, count in sorted(summary["per_target"].items()):
        print(f"[assemble]   {key}: {count}")
    # a contaminated training file must fail loud, not train on poison
    return 1 if summary["contaminating"] else 0


if __name__ == "__main__":
    sys.exit(main())
