#!/usr/bin/env python3
"""Drive the MR6 SFT dataset expansion rollouts against the live lab.

One pilot run per seed over the TRAINING-ELIGIBLE lab targets only
(the 6 GRPO lab-loop originals): each seed changes lictor's bandit
draws so the trajectories differ across repetitions. The trajectory
JSONL of every seed lands in the out directory for the dataset
assembler to merge with the mined trails and the replay records.

CONTAMINATION CONTRACT: the clean-protocol eval challenges (the -002
variants, uaf_heap, srop, callme-002, ret2libc) are excluded here at
the target-mapping level; a manifest case whose binary is not in the
allowlist never becomes a training rollout.

Run:  python scripts/run_rollouts.py \
        --provider-url http://192.168.1.26:9932/v1 \
        --model base-qwen3-1.7b \
        --repetitions 40 --seed-start 10 \
        --workdir experiments/rollouts \
        --out-dir experiments/rollouts/trajectories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The training-eligible lab originals (same allowlist as the trails
# miner): every other manifest case is an eval challenge or
# unclassified and never becomes a training rollout.
TRAINING_ELIGIBLE_BINARIES = {
    "fmtstr",
    "orw",
    "ret2func",
    "ret2plt",
    "shellcode",
    # the MR6 training twins: the uaf capability class as training
    # material (the eval targets uaf_heap/-002 stay unseen by training)
    "uaf_train_a",
    "uaf_train_b",
    # ``ret2csu`` is deliberately absent: it is an Emporium multi-file
    # target (the binary needs ``libret2csu.so``), the pilot's
    # single-binary acquire 404s on it, and it produced ZERO records
    # in every measured run. Not trainable in the current loop.
}


def build_rollout_targets(manifest: Path, cases_dir: Path) -> list[tuple]:
    """Map the verified manifest onto the training-eligible lab targets.

    Reuses the final-bench loader (same port table, same case files)
    and keeps only the allowlisted binaries."""
    from brainforge.experiments.final_bench import load_verified_challenges

    challenges, _skipped = load_verified_challenges(manifest, cases_dir)
    return [c for c in challenges if c[2] in TRAINING_ELIGIBLE_BINARIES]


def run_rollouts(
    targets: list[tuple],
    workdir: Path,
    provider: Any,
    model: str,
    repetitions: int,
    seed_start: int,
    pilot_fn: Any = None,
    out_dir: Path | None = None,
) -> dict:
    """One pilot run per seed; each run's trajectories land in its own
    JSONL so a crashed seed costs one file, not the sweep."""
    if pilot_fn is None:
        from brainforge.experiments.pilot import run_pilot as pilot_fn
    out_dir = out_dir or workdir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"runs": 0, "flagged": 0, "per_seed": {}}
    for offset in range(repetitions):
        seed = str(seed_start + offset)
        output = out_dir / f"trajectories-seed-{seed}.jsonl"
        records = pilot_fn(
            targets,
            workdir / f"rep-seed-{seed}",
            provider,
            model,
            output=output,
            seed=seed,
        )
        flagged = 0
        for record in records:
            messages = record.get("messages") or []
            if not messages:
                continue
            verdict = json.loads(messages[-1]["content"])
            if verdict.get("flag"):
                flagged += 1
        summary["runs"] += 1
        summary["flagged"] += flagged
        summary["per_seed"][seed] = flagged
        print(f"[rollout] seed {seed}: {flagged}/{len(records)} flagged")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/ctf_ingest/manifest.jsonl"),
    )
    parser.add_argument("--cases-dir", type=Path, default=Path("examples/cases"))
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--provider-api-key", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=10)
    parser.add_argument("--workdir", type=Path, default=Path("experiments/rollouts"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    targets = build_rollout_targets(args.manifest, args.cases_dir)
    if not targets:
        print("no training-eligible targets: aborting", file=sys.stderr)
        return 1
    print(f"rollout targets: {[t[2] for t in targets]}")

    from brainforge.experiments.final_bench import _build_provider

    provider = _build_provider("rollouts", args.provider_url, args.provider_api_key)
    summary = run_rollouts(
        targets=targets,
        workdir=args.workdir,
        provider=provider,
        model=args.model,
        repetitions=args.repetitions,
        seed_start=args.seed_start,
    )
    print(f"[rollout] done: {summary['flagged']} flagged across {summary['runs']} seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
