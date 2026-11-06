#!/usr/bin/env python3
"""Expand the MR7 facts dataset from the rollout trajectories.

The majority-vote assembler (assemble_sft_mr7.py) collapses every
trajectory into ONE record per target: 17 records total. The 40
trajectory files carry far more material: every facts step is a
(prompt, facts) pair. This extractor turns each step into a single-turn
record, deduplicates identical pairs, and drops the empty-facts steps
(an empty emission teaches the model to emit nothing, the exact bench
failure the protocol now measures).

Run:  python scripts/expand_sft_mr7.py \\
        --source-glob 'experiments/experiments/rollouts/trajectories/*.jsonl' \\
        --out datasets/sft-mr7-facts/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from assemble_sft_mr7 import VULN_TRUTH, case_material

from brainforge.experiments.pilot import FACT_PROMPT, LLM_ALLOWED

# the trajectory regex strips both numeric suffixes, so the uaf twin
# arrives as "uaf_heap"; VULN_TRUTH and the case files key it as the
# twin id (the eval target uaf_heap-001 is never touched).
TARGET_ALIASES = {"uaf_heap": "uaf_heap-002"}


def _target_of(record: dict) -> str | None:
    """The lab target a trajectory record was collected against."""
    rid = str(record.get("id", ""))
    match = re.search(r"ctf-lictor-(\S+?)(?:-\d+)?-\d+$", rid)
    if match:
        return match.group(1)
    for message in record.get("messages") or []:
        if message.get("role") != "user":
            continue
        m = re.search(r"lab target (\S+)\.", str(message.get("content") or ""))
        if m:
            return m.group(1)
    return None


def _facts_of(record: dict) -> list[tuple[str, str]]:
    """The (name, value) pairs of the facts step, LLM predicates only."""
    facts: list[tuple[str, str]] = []
    for message in record.get("messages") or []:
        if message.get("role") != "assistant" or '"action": "facts"' not in str(
            message.get("content") or ""
        ):
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        for predicate in (payload.get("action_input") or {}).get("predicates") or []:
            match = re.match(r'^(\w+)\("(.*)"\)\.$', str(predicate))
            if match and match.group(1) in LLM_ALLOWED:
                facts.append((match.group(1), match.group(2)))
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-glob",
        nargs="+",
        default=[
            "experiments/experiments/rollouts/trajectories/*.jsonl",
            "experiments/experiments/rollouts-twins/trajectories/*.jsonl",
            "datasets/sft-mr6-v2/train.jsonl",
        ],
    )
    parser.add_argument("--cases-dir", default="examples/cases")
    parser.add_argument("--out", default="datasets/sft-mr7-facts/train.jsonl")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    seen: set[str] = set()
    records: list[dict] = []
    empty_steps = 0
    no_material = 0
    per_target: dict[str, int] = {}

    for pattern in args.source_glob:
        for path in sorted(Path().glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                target = _target_of(record)
                if target is not None and target not in VULN_TRUTH:
                    target = TARGET_ALIASES.get(target, target)
                if target is None or target not in VULN_TRUTH:
                    continue
                facts = _facts_of(record)
                if not facts:
                    empty_steps += 1
                    continue
                material = case_material(target, cases_dir)
                if not material:
                    no_material += 1
                    continue
                # the vuln class is ground truth, never trust the trajectory
                payload = {"facts": [{"name": "vuln", "value": VULN_TRUTH[target]}]}
                emitted = {name: value for name, value in facts}
                for name, value in emitted.items():
                    if name != "vuln":
                        payload["facts"].append({"name": name, "value": value})
                # the prompt material differs per target: two targets with
                # identical facts are two distinct training samples
                key = json.dumps([material, payload], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "messages": [
                            {"role": "user", "content": FACT_PROMPT.format(material=material)},
                            {"role": "assistant", "content": json.dumps(payload)},
                        ]
                    }
                )
                per_target[target] = per_target.get(target, 0) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} records to {out}")
    print(f"dropped: {empty_steps} empty-facts steps, {no_material} without material")
    for target, count in sorted(per_target.items()):
        print(f"  {target}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
