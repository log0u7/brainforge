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
TRAINING_ELIGIBLE = {
    "fmtstr",
    "orw",
    "ret2func",
    "ret2plt",
    "shellcode",
    # ``ret2csu`` excluded: Emporium multi-file target, the
    # single-binary acquire 404s on it, zero records in every
    # measured run.
}

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


def _suggested_facts(suggest_events: list[dict]) -> list[str]:
    """Suggested predicates to augure fact lines: ``["vuln", ["uaf"]]``
    becomes ``vuln("uaf").`` (strings quoted, numbers bare - the
    ``target.facts`` convention: ``gadget("ret_align", 4198422).``)."""
    lines: list[str] = []
    for event in suggest_events:
        for predicate, values in event.get("suggested") or []:
            rendered = []
            for value in values or []:
                rendered.append(f'"{value}"' if isinstance(value, str) else str(value))
            if rendered:
                lines.append(f"{predicate}({', '.join(rendered)}).")
    return lines


def _plan_decision(events: list[dict], index: int) -> tuple[dict | None, list[str]]:
    """The decision that produced the winning outcome at ``events[index]``:
    the nearest preceding plan event carrying a selection. Returns
    ``(decision, suggested_fact_lines)`` where the suggestion lines are
    the suggested predicates emitted before that plan (the facts state
    reconstruction material) and the decision dict carries the plan's
    provenance evidence under ``_provenance_evidence``."""
    decision: dict | None = None
    suggestions: list[dict] = []
    for event in events[:index]:
        if event.get("type") == "suggest":
            suggestions.append(event)
        if event.get("type") == "plan" and event.get("selected"):
            decision = {
                "technique": event.get("selected"),
                "applicable": event.get("applicable", []),
                "_provenance_evidence": (event.get("explain") or {}).get("provenance") or {},
            }
    if decision is None:
        return None, []
    return decision, _suggested_facts(suggestions)


def _evidence_facts(decision: dict) -> list[tuple[str, str]]:
    """The (predicate, value) pairs the winning plan's provenance rule
    actually matched - the facts that MUST be in the reconstruction."""
    pairs: list[tuple[str, str]] = []
    for entries in (decision.get("_provenance_evidence") or {}).values():
        for entry in entries:
            for pair in entry.get("evidence") or []:
                if len(pair) == 2:
                    pairs.append((str(pair[0]), str(pair[1])))
    return pairs


def mine_sacrificed_wins(
    run_dirs: list[Path],
    out: Path,
    target: str,
    challenge_id: str,
    expected_flag: str | None = None,
) -> dict:
    """Mine ONE record per executed win from a SACRIFICED target.

    The target was moved from eval into training (the sacrifice ledger
    in ``datasets/eval/README.md``): its measured wins are real
    executed-success trajectories on the live target, the strongest
    SFT material available. Records keep the eval-namespaced
    ``challenge_id`` on purpose: once they land in the training set,
    the bench protocol's contamination derivation drops the challenge
    from the eval list automatically.

    ``expected_flag`` is the sacrificed case's oracle flag: when the
    run.json carries no successful attempt (the pilot budget ran out
    while the lictor repetitions won) the trail wins still read the
    oracle - the bench lab verified them - so the oracle fills the
    verdict and the record reads solved like every other SFT record.

    Facts reconstruction: the final on-disk ``target.facts`` is a later
    fresh harvest that may LACK the facts the winning plan used (they
    were suggested mid-run and later overwritten). The record's facts
    step = final facts + every suggested predicate emitted before the
    winning plan, deduplicated. A win is kept only if every evidence
    pair of its plan provenance is present in the reconstruction.
    """
    from brainforge.experiments.pilot import build_trajectory

    records: list[dict] = []
    skipped_no_plan = 0
    skipped_evidence_missing = 0
    wins_seen = 0
    for run_dir in run_dirs:
        target_dir = run_dir / target
        trail_path = target_dir / f"trail-{target}.jsonl"
        if not trail_path.is_file():
            continue
        events = []
        for line in trail_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        facts_path = target_dir / "target.facts"
        facts_final = (
            [
                line.strip()
                for line in facts_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if facts_path.is_file()
            else []
        )
        run_json = (
            json.loads((target_dir / "run.json").read_text(encoding="utf-8"))
            if (target_dir / "run.json").is_file()
            else {}
        )
        flag = next(
            (a.get("flag") for a in run_json.get("attempts", []) if a.get("success")),
            None,
        )
        for index, event in enumerate(events):
            if event.get("type") != "outcome" or not (
                event.get("success") and event.get("executed")
            ):
                continue
            wins_seen += 1
            decision, suggested = _plan_decision(events, index)
            if decision is None:
                skipped_no_plan += 1
                continue
            reconstructed = list(dict.fromkeys(facts_final + suggested))
            # the plan's provenance evidence must be explainable by the
            # reconstruction, otherwise the facts state is untrustworthy
            evidence_pairs = _evidence_facts(decision)
            fact_set = set(reconstructed)
            missing = [
                (p, v)
                for p, v in evidence_pairs
                if f'{p}("{v}").' not in fact_set and f"{p}({v})." not in fact_set
            ]
            if evidence_pairs and missing:
                skipped_evidence_missing += 1
                continue
            outcome = {"status": "success", "flag": flag or expected_flag}
            # _provenance_evidence is internal bookkeeping: strip it
            # before the record is built, it must not reach training data
            clean_decision = {k: v for k, v in decision.items() if not k.startswith("_")}
            record = build_trajectory(
                challenge_id=challenge_id,
                student_input=(f"Solve this CTF challenge against the local lab target {target}."),
                facts_steps=reconstructed,
                decision=clean_decision,
                outcome=outcome,
                model="base-qwen3-1.7b (mined from trail)",
            )
            record["metadata"]["provenance"] = f"executed-success-{event.get('run')}"
            record["metadata"]["mined_from"] = str(run_dir)
            records.append(record)
    if records:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        out.write_text(text, encoding="utf-8")
    return {
        "wins": wins_seen,
        "records": len(records),
        "skipped_no_plan": skipped_no_plan,
        "skipped_evidence_missing": skipped_evidence_missing,
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
