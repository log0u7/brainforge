"""Unit tests for the consolidation cycle (ADR-0020, MR3).

The cycle = ingest trajectories, stratified replay mix, one training pass,
a cheap per-round gate on the frozen post-cutoff holdout. A failed gate
keeps the previous round's checkpoint; nothing is overwritten in place.
Every round journals provenance, records seen and the gate outcome.

All heavy dependencies are stubbed: ``train_student`` and the gate callable
are injected fakes, so the suite runs without the training extra.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from brainforge.errors import BrainforgeError
from brainforge.training.consolidation import consolidate, gate_ok


def _record(rec_id: str, source: str = "repo-a", risk: bool = False) -> dict[str, Any]:
    return {
        "id": rec_id,
        "messages": [
            {"role": "user", "content": f"challenge {rec_id}"},
            {"role": "assistant", "content": f"solution {rec_id}"},
        ],
        "metadata": {"source": {"type": source}, "recitation_risk": risk},
    }


def _trajectory(rec_id: str, flag: str | None) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {"role": "user", "content": f"challenge {rec_id}"},
    ]
    if flag is not None:
        messages.append({"role": "assistant", "content": json.dumps({"flag": flag})})
    return {"id": rec_id, "messages": messages}


class _FakeTrainer:
    """Replaces brainforge.training.qlora.train_student."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config: Any, dataset_dir: Path, output_dir: Path, **kwargs: Any) -> dict:
        rows = (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        self.calls.append(
            {"dataset_dir": dataset_dir, "n_rows": len(rows), "output_dir": output_dir}
        )
        return {"output_dir": str(output_dir), "backend": "stub"}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _gate_ok_case() -> dict[str, Any]:
    return {"id": "ctf-lab-1", "metadata": {"recitation_risk": False}}


def test_gate_ok_regresses_on_worse_metric() -> None:
    # round 1 accepted (no previous), round 2 accepted when not worse
    assert gate_ok({"flag_accuracy": 0.5}, {}) is True
    assert gate_ok({"flag_accuracy": 0.5}, {"flag_accuracy": 0.5}) is True
    assert gate_ok({"flag_accuracy": 0.7}, {"flag_accuracy": 0.5}) is True
    assert gate_ok({"flag_accuracy": 0.3}, {"flag_accuracy": 0.5}) is False


def test_gate_ok_ignores_missing_metric_keys() -> None:
    # the holdout carries only flag_accuracy for CTF; a metric absent from
    # the previous round is not a regression
    assert gate_ok({"accuracy": 0.4}, {"flag_accuracy": 0.9}) is True


def test_consolidate_first_round_trains_and_journals(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(trajectories, [_trajectory("ctf-lab-1", "FLAG{abc}")])
    dataset_dir = tmp_path / "grpo"
    gate_dataset = tmp_path / "gate.jsonl"
    _write_jsonl(gate_dataset, [_gate_ok_case()])
    trainer = _FakeTrainer()

    def fake_gate(model: Path, dataset: Path) -> dict[str, Any]:
        return {"accuracy": 0.5, "flag_accuracy": 0.5, "n_records": 1}

    result = consolidate(
        config=object(),
        trajectories=trajectories,
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "consolidation",
        replay_ratio=0.3,
        gate_dataset=gate_dataset,
        train_fn=trainer,
        gate_fn=fake_gate,
    )

    assert result["accepted"] is True
    assert result["round"] == 1
    assert result["gate"] == {"accuracy": 0.5, "flag_accuracy": 0.5, "n_records": 1}
    # train.jsonl now holds the ingested record
    rows = (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    # sidecar carries the execution-read flag
    flags = (dataset_dir / "grpo_flags.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(flags[0]) == {"id": "ctf-lab-1", "expected_flag": "FLAG{abc}"}
    # the trainer saw the mixed dataset
    assert trainer.calls[0]["n_rows"] == 1
    # journal written
    journal_path = tmp_path / "consolidation" / "round-1.json"
    assert journal_path.is_file()
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["accepted"] is True
    assert journal["ingested"] == 1
    assert journal["replayed"] == 0
    assert journal["gate"] == result["gate"]


def test_consolidate_replay_mixes_prior_rounds_stratified(tmp_path: Path) -> None:
    # prior corpus: 10 records from two sources; new round brings 2 records;
    # replay_ratio 0.5 -> the mix must hold 2 new + 2 replayed (50% old)
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(trajectories, [_trajectory("new-1", "FLAG{n1}"), _trajectory("new-2", None)])
    dataset_dir = tmp_path / "grpo"
    prior = [_record(f"old-{i}", source="repo-a" if i < 5 else "repo-b") for i in range(10)]
    _write_jsonl(dataset_dir / "train.jsonl", prior)
    gate_dataset = tmp_path / "gate.jsonl"
    _write_jsonl(gate_dataset, [_gate_ok_case()])
    trainer = _FakeTrainer()

    def fake_gate(model: Path, dataset: Path) -> dict[str, Any]:
        return {"flag_accuracy": 0.5}

    result = consolidate(
        config=object(),
        trajectories=trajectories,
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "consolidation",
        replay_ratio=0.5,
        gate_dataset=gate_dataset,
        train_fn=trainer,
        gate_fn=fake_gate,
    )

    assert result["ingested"] == 2
    assert result["replayed"] == 2
    rows = [
        json.loads(line)
        for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids = {r["id"] for r in rows}
    # cumulative corpus: prior 10 + 2 new
    assert len(ids) == 12
    # the training mix held 4 rows: 2 new + 2 stratified replay (one per source)
    mix_ids = trainer.calls[0]["n_rows"]
    assert mix_ids == 4


def test_consolidate_replay_excludes_contaminated_and_holdout_ids(tmp_path: Path) -> None:
    # one prior record carries recitation_risk; one shares the gate holdout id;
    # both must be excluded from the replay mix
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(trajectories, [_trajectory("new-1", "FLAG{n1}")])
    dataset_dir = tmp_path / "grpo"
    prior = [
        _record("clean-1", source="repo-a"),
        _record("risky-1", source="repo-a", risk=True),
        _record("ctf-lab-1", source="repo-a"),
    ]
    _write_jsonl(dataset_dir / "train.jsonl", prior)
    gate_dataset = tmp_path / "gate.jsonl"
    _write_jsonl(gate_dataset, [_gate_ok_case()])
    trainer = _FakeTrainer()

    def fake_gate(model: Path, dataset: Path) -> dict[str, Any]:
        return {"flag_accuracy": 0.5}

    consolidate(
        config=object(),
        trajectories=trajectories,
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "consolidation",
        replay_ratio=0.5,
        gate_dataset=gate_dataset,
        train_fn=trainer,
        gate_fn=fake_gate,
    )

    # the mix held 1 new + 1 replayed clean record; risky and holdout-id excluded
    assert trainer.calls[0]["n_rows"] == 2


def test_consolidate_failed_gate_keeps_previous_checkpoint(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(trajectories, [_trajectory("new-1", "FLAG{n1}")])
    dataset_dir = tmp_path / "grpo"
    gate_dataset = tmp_path / "gate.jsonl"
    _write_jsonl(gate_dataset, [_gate_ok_case()])
    trainer = _FakeTrainer()

    gates = [
        {"flag_accuracy": 0.8, "n_records": 1},
        {"flag_accuracy": 0.4, "n_records": 1},
    ]

    def fake_gate(model: Path, dataset: Path) -> dict[str, Any]:
        return gates.pop(0)

    first = consolidate(
        config=object(),
        trajectories=_write_and_return(trajectories),
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "consolidation",
        replay_ratio=0.3,
        gate_dataset=gate_dataset,
        train_fn=trainer,
        gate_fn=fake_gate,
    )
    assert first["accepted"] is True
    # accepted round promotes its checkpoint
    assert (tmp_path / "consolidation" / "current.json").is_file()
    current = json.loads((tmp_path / "consolidation" / "current.json").read_text(encoding="utf-8"))
    assert current["round"] == 1

    second = consolidate(
        config=object(),
        trajectories=trajectories,
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "consolidation",
        replay_ratio=0.3,
        gate_dataset=gate_dataset,
        train_fn=trainer,
        gate_fn=fake_gate,
    )
    assert second["accepted"] is False
    # the current pointer still references round 1 - nothing overwritten
    current = json.loads((tmp_path / "consolidation" / "current.json").read_text(encoding="utf-8"))
    assert current["round"] == 1
    # round 2 journal records the failure and keeps its own artifacts
    journal = json.loads((tmp_path / "consolidation" / "round-2.json").read_text(encoding="utf-8"))
    assert journal["accepted"] is False


def _write_and_return(path: Path) -> Path:
    if not path.is_file():
        _write_jsonl(path, [_trajectory("new-1", "FLAG{n1}")])
    return path


def test_consolidate_requires_trajectories(tmp_path: Path) -> None:
    with pytest.raises(BrainforgeError):
        consolidate(
            config=object(),
            trajectories=tmp_path / "missing.jsonl",
            dataset_dir=tmp_path / "grpo",
            output_dir=tmp_path / "consolidation",
            replay_ratio=0.3,
            gate_dataset=tmp_path / "gate.jsonl",
            train_fn=_FakeTrainer(),
            gate_fn=lambda model, dataset: {"flag_accuracy": 0.5},
        )
