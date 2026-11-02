"""The consolidation cycle: ingest, replay mix, one training pass, a gate.

Implements the sleep cycle of ADR-0020 (wake/sleep shape credited to the
Sleep framework, arXiv:2606.03979). One round:

1. *Ingest* the pilot's execution-verified trajectories into the
   cumulative dataset (merged by id, the sidecar carries the flag the
   service read - never the model's guess).
2. *Replay mix*: a stratified sample of the prior corpus (one record per
   source group until the ``replay_ratio`` share of the mix is old data),
   excluding contaminated records (``recitation_risk`` true) and any id
   that appears in the gate holdout - the contamination guard re-run.
3. *Train*: one pass of the configured strategy over the mix, into a
   round-specific directory. Nothing is overwritten in place.
4. *Gate*: the cheap per-round gate (task-eval on the frozen post-cutoff
   holdout). A failed gate keeps the previous round's checkpoint: the
   ``current.json`` pointer is only updated when the gate passes.
5. *Journal*: provenance, records seen, gate outcome and the checkpoint
   path land in ``round-<N>.json`` under the consolidation output dir.

Heavy dependencies are NOT imported here: the trainer and the gate are
injected callables, so the orchestration is testable without torch.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brainforge.errors import BrainforgeError

#: metrics that gate acceptance (regression = any of these got worse)
GATE_METRICS = ("flag_accuracy", "cwe_accuracy", "accuracy")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def gate_ok(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    """Cheap per-round gate: not worse than the previous round.

    A metric missing on either side is ignored: the holdout may carry a
    metric the earlier gate never computed (and vice versa - a different
    domain split computes different keys). The first round has no
    previous - accepted.
    """
    for metric in GATE_METRICS:
        prev = previous.get(metric)
        curr = current.get(metric)
        if prev is None or curr is None:
            continue
        if curr < prev:
            return False
    return True


def _record_source(record: dict[str, Any]) -> str:
    source = record.get("metadata", {}).get("source", {})
    if isinstance(source, dict):
        for key in ("repository", "path", "url", "type"):
            value = source.get(key)
            if value:
                return str(value)
    return "unknown"


def _replay_sample(
    prior: list[dict[str, Any]],
    new_count: int,
    replay_ratio: float,
    excluded_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Stratified replay: ``replay_ratio`` share of the mix is prior data.

    ``replay_ratio`` r with ``n`` new records wants ``r/(1-r) * n`` replayed
    records (so the mix holds exactly an r share of old data), capped by
    what the prior corpus can offer after the contamination guard.
    Stratification is round-robin over source groups: every source keeps
    coverage instead of one dominant group filling the quota.
    """
    if replay_ratio <= 0.0 or new_count == 0 or replay_ratio >= 1.0:
        if replay_ratio >= 1.0 and replay_ratio != 1.0:
            raise BrainforgeError(f"replay_ratio must be in [0, 1); got {replay_ratio}")
        if replay_ratio >= 1.0:
            target = len(prior)
        else:
            return []
    else:
        target = min(len(prior), int(new_count * replay_ratio / (1 - replay_ratio)))

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in prior:
        rid = record.get("id")
        if rid in excluded_ids:
            continue
        groups.setdefault(_record_source(record), []).append(record)
    for group in groups.values():
        rng.shuffle(group)

    replayed: list[dict[str, Any]] = []
    while len(replayed) < target and any(groups.values()):
        for source in list(groups):
            if len(replayed) >= target:
                break
            if groups[source]:
                replayed.append(groups[source].pop())
    return replayed


def consolidate(
    config: Any,
    trajectories: Path,
    dataset_dir: Path,
    output_dir: Path,
    replay_ratio: float = 0.3,
    gate_dataset: Path | None = None,
    train_fn: Callable[..., dict[str, Any]] | None = None,
    gate_fn: Callable[[Path, Path], dict[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one consolidation round; returns the round journal as a dict.

    Nothing is silently defaulted: without a gate dataset the gate cannot
    run, and a round without a gate is a training run, not a sleep cycle.
    """
    from brainforge.dataset.flag_ground_truth import ingest_trajectories

    if not trajectories.is_file():
        raise BrainforgeError(f"trajectories file not found: {trajectories}")
    if gate_dataset is None or not gate_dataset.is_file():
        raise BrainforgeError(
            "no gate dataset: the per-round gate needs the frozen post-cutoff holdout"
        )

    output_dir = Path(output_dir)
    dataset_dir = Path(dataset_dir)

    # --- previous state -------------------------------------------------
    current_path = output_dir / "current.json"
    current: dict[str, Any] = {}
    if current_path.is_file():
        current = json.loads(current_path.read_text(encoding="utf-8"))
    previous_round = current.get("round", 0)
    previous_gate = current.get("gate", {})
    round_number = previous_round + 1

    # --- 1. ingest ------------------------------------------------------
    new_records = [
        json.loads(line)
        for line in trajectories.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases: list[dict[str, Any]] = []
    cases_dir = trajectories.parent / "cases"
    if cases_dir.is_dir():
        for path in sorted(cases_dir.glob("*.json")):
            cases.append(json.loads(path.read_text(encoding="utf-8")))
    # the pilot's execution-verified flag rides the trajectory's assistant
    # verdict (JSON with a "flag" key); cases from the lab dir take
    # precedence, trajectory-extracted flags fill the gap
    known_case_ids = {case.get("id") for case in cases}
    for record in new_records:
        if record.get("id") in known_case_ids:
            continue
        messages = record.get("messages") or []
        if not messages or messages[-1].get("role") != "assistant":
            continue
        content = messages[-1].get("content", "")
        try:
            data = json.loads(content[content.find("{") :])
        except (ValueError, TypeError):
            continue
        flag = data.get("flag")
        if isinstance(flag, str) and flag.strip():
            cases.append({"id": record.get("id"), "metadata": {"expected_flag": flag.strip()}})
    rows, flags = ingest_trajectories(new_records, cases)
    if not rows:
        raise BrainforgeError("no usable trajectories: nothing to consolidate")

    # merge into the cumulative dataset (id-keyed, like the CLI command)
    train_path = dataset_dir / "train.jsonl"
    merged: dict[str, dict[str, Any]] = {
        row["id"]: row for row in _read_jsonl(train_path) if row.get("id")
    }
    prior_count = len(merged)
    fresh = sum(1 for row in rows if row.get("id") in merged)
    for row in rows:
        if row.get("id"):
            merged[row["id"]] = row
    _write_jsonl(train_path, list(merged.values()))

    flags_path = dataset_dir / "grpo_flags.jsonl"
    known_flags: dict[str, dict[str, Any]] = {
        entry["id"]: entry for entry in _read_jsonl(flags_path) if entry.get("id")
    }
    for entry in flags:
        known_flags[entry["id"]] = entry
    _write_jsonl(flags_path, list(known_flags.values()))

    # --- 2. replay mix ---------------------------------------------------
    gate_ids = {row["id"] for row in _read_jsonl(gate_dataset) if row.get("id")}
    excluded = gate_ids | {
        row["id"]
        for row in merged.values()
        if row.get("id") and row.get("metadata", {}).get("recitation_risk") is True
    }
    prior_rows = [
        row for row in merged.values() if row.get("id") not in {r.get("id") for r in rows}
    ]
    replayed = _replay_sample(prior_rows, len(rows), replay_ratio, excluded, random.Random(seed))
    mix = rows + replayed

    # --- 3. train ---------------------------------------------------------
    round_dir = output_dir / f"round-{round_number}"
    round_dataset = round_dir / "mix"
    _write_jsonl(round_dataset / "train.jsonl", mix)
    trainer = train_fn
    if trainer is None:
        from brainforge.cli.train_cmd import train_student

        trainer = train_student
    trainer(config, round_dataset, round_dir / "adapter")

    # --- 4. gate -----------------------------------------------------------
    gate = gate_fn
    if gate is None:
        from brainforge.training.task_eval import evaluate_model_on_records

        gate = evaluate_model_on_records
    gate_result = gate(round_dir / "adapter", gate_dataset)
    accepted = gate_ok(gate_result, previous_gate)

    # --- 5. journal ---------------------------------------------------------
    journal = {
        "round": round_number,
        "accepted": accepted,
        "ingested": len(rows),
        "fresh_replacements": fresh,
        "replayed": len(replayed),
        "prior_corpus": prior_count,
        "mix_size": len(mix),
        "replay_ratio": replay_ratio,
        "excluded_ids": len(excluded),
        "gate": gate_result,
        "checkpoint": str(round_dir / "adapter"),
    }
    (round_dir / "round.json").write_text(json.dumps(journal, indent=2), encoding="utf-8")
    (output_dir / f"round-{round_number}.json").write_text(
        json.dumps(journal, indent=2), encoding="utf-8"
    )
    if accepted:
        (output_dir / "current.json").write_text(
            json.dumps(
                {
                    "round": round_number,
                    "gate": gate_result,
                    "checkpoint": str(round_dir / "adapter"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return journal
