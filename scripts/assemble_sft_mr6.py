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

# The Emporium multi-file target: the pilot's single-binary acquire
# 404s on it, so every record it ever produced is an acquire-failure
# artifact (zero records in every measured run). Its rollout noise
# never crosses into a training file.
EXCLUDED_TARGETS = ("ret2csu",)


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _verdict_flag(record: dict) -> str | None:
    """The flag the record's final assistant verdict carries (None when
    the verdict is missing, unparseable, or flagless - the failed
    rollout)."""
    import json as _json

    messages = record.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return None
    try:
        verdict = _json.loads(messages[-1].get("content") or "")
    except _json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict):
        return None
    return verdict.get("flag") or None


def _target_of(record: dict) -> str:
    """The lab target name from the record ID (``lab-fmtstr-001``,
    ``ctf-lictor-orw-001``, ``sft-verified-facts-ret2plt``); unknown IDs
    group under ``unknown``."""
    import re as _re

    rid = str(record.get("id", ""))
    match = _re.search(r"(?:lab|ctf-lictor|sft-verified-facts)-([a-z0-9_]+?)(?:-\d+)?$", rid)
    return match.group(1) if match else "unknown"


def _enrich_metadata(record: dict) -> dict:
    """Fill the provenance keys the trainer's validation gate requires
    (``source``, ``teachers``, ``quality.passed``, ``recitation_risk``)
    without overwriting provenance a record already carries (the replay
    corpora ship their own, including ``recitation_risk: true`` for the
    pre-cutoff sources). The lab records' facts are MEASURED at run
    time, so their recitation risk is none and the quality gate passes
    on the execution-verified flag."""
    metadata = dict(record.get("metadata") or {})
    if "source" not in metadata:
        # the per-target path also diversifies the split's group key
        # (one shared repository string would hash every lab record
        # into a single bucket and starve the validation split; the
        # path identifies the target, so it is the group)
        metadata["source"] = {
            "type": "lictor-lab",
            "repository": None,
            "commit": None,
            "path": f"lab/targets/{_target_of(record)}",
            "url": None,
            "date": "2026-11-03",
        }
    if "teachers" not in metadata:
        pilot = metadata.get("pilot") or {}
        model = pilot.get("model")
        if model:
            metadata["teachers"] = [
                {
                    "role": "exploit_student",
                    "kind": "student",
                    "provider": "llamacpp",
                    "model": model,
                    "family": "qwen",
                }
            ]
        elif metadata.get("verified_by") == "decision":
            metadata["teachers"] = [
                {
                    "role": "fact_teacher",
                    "kind": "teacher",
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-pro",
                    "family": "deepseek",
                }
            ]
        else:
            metadata["teachers"] = []
    if "quality" not in metadata:
        metadata["quality"] = {
            "passed": True,
            "evidence": "flag_verified_by_execution"
            if _verdict_flag(record)
            else "decision_verified",
        }
    if "recitation_risk" not in metadata:
        metadata["recitation_risk"] = False
    record["metadata"] = metadata
    return record


def _fill_empty_messages(record: dict) -> dict:
    """The validator forbids empty message content, and the best SFT
    records (a successful run that routed from profiler facts ALONE)
    carry an empty facts-step TOOL observation. Stamp a truthful marker
    on empty tool messages only: an empty ASSISTANT completion is the
    MR5 poison and stays subject to the drop/cap gates."""
    messages = record.get("messages") or []
    for message in messages:
        if message.get("role") == "tool" and not (message.get("content") or "").strip():
            message["content"] = "(the LLM emitted no usable facts)"
    return record


def assemble(
    sources: list[Path],
    out: Path,
    per_target_cap: int,
    empty_completion_share: float,
    keep_failed: bool = False,
) -> dict:
    """Merge, gate, cap, write. Returns the summary the CLI prints."""
    records: list[dict] = []
    contaminating = 0
    failed_dropped = 0
    for path in sources:
        for record in _load(path):
            rid = str(record.get("id", ""))
            if any(marker in rid for marker in EVAL_ID_MARKERS):
                contaminating += 1
                continue
            if any(marker in rid for marker in EXCLUDED_TARGETS):
                continue
            if not keep_failed and _verdict_flag(record) is None:
                # a failed trajectory is not SFT material: cloning it
                # teaches the model to fail (MR5 lesson). The failures
                # stay in their source JSONLs for the GRPO zero-reward
                # sidecar.
                failed_dropped += 1
                continue
            records.append(_enrich_metadata(_fill_empty_messages(record)))

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
        "failed_dropped": failed_dropped,
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
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="retain failed rollouts (for the GRPO sidecar, not SFT)",
    )
    parser.add_argument("--empty-completion-share", type=float, default=0.2)
    args = parser.parse_args()

    sources = _expand(args.source)
    summary = assemble(
        sources=sources,
        out=args.out,
        per_target_cap=args.per_target_cap,
        empty_completion_share=args.empty_completion_share,
        keep_failed=args.keep_failed,
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
