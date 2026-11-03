#!/usr/bin/env python3
"""Mine the benchmark run dirs into an SFT trajectory dataset (MR6).

The benchmark runs (``experiments/final-bench-results/run-*``) hold a
per-target trail (``trail-<target>.jsonl``), the merged fact base
(``target.facts``) and the run manifest (``run.json``). For the
TRAINING-ELIGIBLE lab originals only (the 6 targets the GRPO lab loop
already trained on), this script reconstructs one agentic trajectory
record per target: user challenge, facts step, decision step, run
outcome - the same shape ``build_trajectory`` emits, so the mined
records and the fresh rollout records train the same behavior.

CONTAMINATION CONTRACT (memory dc8e61c1): the 11 clean-protocol eval
challenges NEVER enter the training set. An eval-named directory is
skipped and counted; an unknown directory is skipped and reported for
the operator to classify. If the mined output would be empty the CLI
exits nonzero so a silently-empty dataset cannot pass unnoticed.

Run:  python scripts/build_sft_trails.py \
        --run-dir experiments/final-bench-results/run-20261103-17ch \
        --out datasets/sft-trails/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The training-eligible lab originals: exactly the 6 targets the GRPO
# lab loop trained on (translated to lab-<name>-001 IDs). Everything
# else in a run dir is either a clean-protocol eval challenge or an
# unclassified target - neither may become a training record.
TRAINING_ELIGIBLE = {"fmtstr", "orw", "ret2csu", "ret2func", "ret2plt", "shellcode"}

# The 11 clean-protocol eval challenges, as run-dir directory names.
EVAL_DIR_NAMES = {
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
    "uaf_heap-002",
}


def _final_outcome(run_json: dict) -> dict:
    """The run manifest to the outcome shape ``build_trajectory`` reads:
    the overall status and the flag the run actually read (any
    successful attempt carries it; the oracle already matched it)."""
    flag = next(
        (a.get("flag") for a in run_json.get("attempts", []) if a.get("success")),
        None,
    )
    return {"status": run_json.get("status", "failed"), "flag": flag}


def _final_decision(trail_path: Path) -> dict:
    """The last plan entry of the trail is the decision that produced
    the run: its selected technique and the applicable ranking."""
    decision: dict = {}
    for line in trail_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") == "plan":
            decision = {
                "technique": entry.get("selected"),
                "applicable": entry.get("applicable", []),
            }
    return decision


def mine_run_dirs(run_dirs: list[Path], out: Path) -> dict:
    """Mine every run dir into one JSONL of agentic trajectory records.

    Returns the summary the CLI prints: records written, eval
    directories skipped, unknown directories reported by name."""
    from brainforge.experiments.pilot import build_trajectory

    records: list[dict] = []
    skipped_eval = 0
    skipped_unknown: list[str] = []
    for run_dir in run_dirs:
        for target_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            name = target_dir.name
            if name in EVAL_DIR_NAMES:
                skipped_eval += 1
                continue
            if name not in TRAINING_ELIGIBLE:
                skipped_unknown.append(name)
                continue
            trail_path = target_dir / f"trail-{name}.jsonl"
            run_json_path = target_dir / "run.json"
            if not trail_path.exists() or not run_json_path.exists():
                skipped_unknown.append(name)
                continue
            facts_path = target_dir / "target.facts"
            facts_steps = (
                [
                    line.strip()
                    for line in facts_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if facts_path.exists()
                else []
            )
            run_json = json.loads(run_json_path.read_text(encoding="utf-8"))
            records.append(
                build_trajectory(
                    challenge_id=f"lab-{name}-001",
                    student_input=(
                        f"Solve this CTF challenge against the local lab target {name}."
                    ),
                    facts_steps=facts_steps,
                    decision=_final_decision(trail_path),
                    outcome=_final_outcome(run_json),
                    model="base-qwen3-1.7b (mined from trail)",
                )
            )
    if records:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        out.write_text(text, encoding="utf-8")
    return {
        "records": len(records),
        "skipped_eval": skipped_eval,
        "skipped_unknown": skipped_unknown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = mine_run_dirs(args.run_dir, args.out)
    print(
        f"[mine] records: {summary['records']}, "
        f"eval skipped: {summary['skipped_eval']}, "
        f"unknown skipped: {summary['skipped_unknown']}"
    )
    # a silently-empty dataset is worse than a loud failure: if nothing
    # was mined the operator must know before training on it
    return 0 if summary["records"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
