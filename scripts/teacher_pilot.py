"""Teacher-capability pilot: measure how many HARD challenges a candidate
teacher model solves, WITHOUT mining any dataset (diagnostic only, the
trajectories are diagnostic evidence, never training data).

Targets: the clean-protocol eval challenges where the 1.7B base scored
0/10 (the curriculum gap). Running the teacher on them is a measurement;
the eval set stays untouched for the student bench (no record ever
enters an SFT dataset - the whole point is to decide WHICH teacher covers
WHICH family before the real mining on training twins).

Run:  python /tmp/teacher_pilot.py \
        --provider-url http://127.0.0.1:9934/v1 \
        --model teacher-qwen3-4b \
        --repetitions 10 \
        --workdir /tmp/teacher-pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")

from brainforge.experiments.bench_lab import _record_flag, _run_attempt_capped
from brainforge.experiments.final_bench import (
    _build_provider,
    load_verified_challenges,
)

# The clean-failed families: the 1.7B base scored 0/10 on all of these
# (run-20261116-mr7-10rep-10rep, integrity-verified).
HARD_CHALLENGE_IDS = [
    "ctf-lictor-brop-001",
    "ctf-lictor-jop-001",
    "ctf-lictor-one_gadget-001",
    "ctf-lictor-setcontext_srop-001",
    "ctf-lictor-ret2vdso-001",
    "ctf-lictor-got_partial_overwrite-001",
    "ctf-lictor-ret2partial_overwrite-001",
    "ctf-lictor-uaf_heap-001",
    "ctf-lictor-callme-eval-001",
    "ctf-lictor-callme-eval2-001",
    "ctf-lictor-fmtstr-eval-001",
    "ctf-lictor-fmtstr-eval2-001",
    "ctf-lictor-orw-eval-001",
    "ctf-lictor-orw-eval2-001",
    "ctf-lictor-ret2func-eval2-001",
    "ctf-lictor-ret2libc-eval-001",
    "ctf-lictor-ret2libc-eval2-001",
    "ctf-lictor-ret2plt-eval-001",
    "ctf-lictor-ret2plt-eval2-001",
    "ctf-lictor-srop-eval-001",
    "ctf-lictor-srop-eval2-001",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/teacher-pilot"))
    parser.add_argument("--attempt-cap", type=float, default=600.0)
    parser.add_argument("--seed-base", type=int, default=7)
    args = parser.parse_args()

    challenges, _ = load_verified_challenges(
        Path("experiments/ctf_ingest/manifest.jsonl"), Path("examples/cases")
    )
    by_id = {entry[0]: entry for entry in challenges}
    targets = [by_id[c] for c in HARD_CHALLENGE_IDS if c in by_id]
    expected_flags = {entry[0]: entry[4] for entry in targets if len(entry) > 4}
    provider = _build_provider("teacher", args.provider_url, None)
    print(
        f"teacher pilot: {len(targets)} hard challenges x {args.repetitions} reps",
        flush=True,
    )

    pairs = [(rep, entry) for rep in range(args.repetitions) for entry in targets]
    tally: dict[str, dict] = {}

    def attempt(pair):
        rep, entry = pair
        started = time.monotonic()
        records, timed_out = _run_attempt_capped(
            lambda e, wd, p, m, seed=None: __import__(
                "brainforge.experiments.pilot", fromlist=["run_pilot"]
            ).run_pilot(e, wd, p, m, seed=seed),
            [entry],
            args.workdir / f"rep{rep}" / entry[0],
            provider,
            args.model,
            args.attempt_cap,
            seed=str(args.seed_base + rep),
        )
        elapsed = time.monotonic() - started
        flag = _record_flag(records[0]) if records else None
        cid = entry[0]
        ok = bool(cid in expected_flags and flag == expected_flags[cid])
        status = "TIMEOUT" if timed_out else ("FLAG" if ok else "miss")
        print(
            f"[teacher {rep + 1}/{args.repetitions}] {cid}: {status} (elapsed {elapsed:.0f}s)",
            flush=True,
        )
        return cid, ok

    with ThreadPoolExecutor(max_workers=4) as pool:
        for cid, ok in pool.map(attempt, pairs):
            slot = tally.setdefault(cid, {"flag": 0, "miss": 0})
            slot["flag" if ok else "miss"] += 1

    print("\n=== TEACHER CAPABILITY VERDICT ===", flush=True)
    solved = []
    for cid in sorted(tally):
        s = tally[cid]
        n = s["flag"] + s["miss"]
        mark = " <== COVERED" if s["flag"] > 0 else ""
        if s["flag"]:
            solved.append(cid)
        print(f"  {cid:40} {s['flag']:>2}/{n}{mark}", flush=True)
    print(
        f"\nteacher covers {len(solved)}/{len(tally)} hard families",
        flush=True,
    )
    Path(args.workdir / "teacher-verdict.json").write_text(
        json.dumps({"covered": solved, "tally": tally}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
