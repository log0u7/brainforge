"""Unit tests for the three-arm benchmark orchestrator (MR4).

Arms: no-KD (plain QLoRA SFT), sidecar (train run --kd, ADR-0015),
GKD (strategy gkd, ADR-0020), plus the opportunistic LoRA rank 64 arm.
The orchestrator trains each arm into its own directory, evaluates on the
same frozen holdout, and renders a markdown comparison table.

The trainer and the gate are injected callables, so the suite runs
without the training extra.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError
from brainforge.training.benchmark import (
    ARMS,
    render_markdown_table,
    run_arm,
    run_three_arm_benchmark,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _record(rec_id: str) -> dict[str, Any]:
    return {
        "id": rec_id,
        "messages": [
            {"role": "user", "content": f"challenge {rec_id}"},
            {"role": "assistant", "content": f"answer {rec_id}"},
        ],
    }


class _FakeTrainer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self, config: TrainingConfig, dataset_dir: Path, output_dir: Path, **kw: Any
    ) -> dict:
        self.calls.append(output_dir.name)
        return {"output_dir": str(output_dir), "backend": "stub"}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prepared"
    _write_jsonl(d / "train.jsonl", [_record("a"), _record("b")])
    _write_jsonl(d / "validation.jsonl", [_record("a")])
    return d


def test_arms_registry_covers_three_plus_lora() -> None:
    assert set(ARMS) == {"no-kd", "sidecar", "gkd", "lora64"}


def test_run_arm_trains_and_evaluates(dataset_dir: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    trainer = _FakeTrainer()

    def fake_gate(model: Path, dataset: Path, **kw: Any) -> dict[str, Any]:
        return {"accuracy": 0.5, "flag_accuracy": 0.5, "n_records": 1}

    result = run_arm(
        config=TrainingConfig(),
        arm="no-kd",
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "bench",
        gate_dataset=gate,
        train_fn=trainer,
        gate_fn=fake_gate,
    )
    assert result["arm"] == "no-kd"
    assert result["metrics"]["flag_accuracy"] == 0.5
    assert trainer.calls == ["no-kd"]


def test_run_arm_sidecar_requires_kd_sidecar(dataset_dir: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    with pytest.raises(BrainforgeError, match=r"kd\.jsonl"):
        run_arm(
            config=TrainingConfig(),
            arm="sidecar",
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "bench",
            gate_dataset=gate,
            train_fn=_FakeTrainer(),
            gate_fn=lambda model, dataset, **kw: {},
        )


def test_run_arm_lora64_overrides_rank(dataset_dir: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    seen: list[TrainingConfig] = []

    def fake_train(config: TrainingConfig, dataset_dir: Path, output_dir: Path, **kw: Any) -> dict:
        seen.append(config)
        return {"output_dir": str(output_dir)}

    run_arm(
        config=TrainingConfig(),
        arm="lora64",
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "bench",
        gate_dataset=gate,
        train_fn=fake_train,
        gate_fn=lambda model, dataset, **kw: {"accuracy": 0.5},
    )
    assert seen[0].lora_rank == 64


def test_run_arm_gkd_requires_teacher(dataset_dir: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    with pytest.raises(BrainforgeError, match="gkd_teacher"):
        run_arm(
            config=TrainingConfig(),
            arm="gkd",
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "bench",
            gate_dataset=gate,
            train_fn=_FakeTrainer(),
            gate_fn=lambda model, dataset, **kw: {},
        )


def test_run_arm_passes_max_new_tokens_to_gate(dataset_dir: Path, tmp_path: Path) -> None:
    """Regression: the bench gate got max_new_tokens=512 (task_eval default)
    while the ground-truth completions need up to ~1550 tokens - answers were
    cut and scored unparseable, invalidating every benchmark number."""
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    seen: list[dict[str, Any]] = []

    def fake_gate(model: Path, dataset: Path, **kw: Any) -> dict[str, Any]:
        seen.append(kw)
        return {"accuracy": 0.5, "flag_accuracy": 0.5, "n_records": 1}

    run_arm(
        config=TrainingConfig(),
        arm="no-kd",
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "bench",
        gate_dataset=gate,
        train_fn=_FakeTrainer(),
        gate_fn=fake_gate,
    )
    assert seen[0].get("max_new_tokens") == 2048


def test_run_arm_accepts_explicit_max_new_tokens(dataset_dir: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    seen: list[dict[str, Any]] = []

    def fake_gate(model: Path, dataset: Path, **kw: Any) -> dict[str, Any]:
        seen.append(kw)
        return {"accuracy": 0.5, "flag_accuracy": 0.5, "n_records": 1}

    run_arm(
        config=TrainingConfig(),
        arm="no-kd",
        dataset_dir=dataset_dir,
        output_dir=tmp_path / "bench",
        gate_dataset=gate,
        train_fn=_FakeTrainer(),
        gate_fn=fake_gate,
        max_new_tokens=4096,
    )
    assert seen[0].get("max_new_tokens") == 4096


def test_run_three_arm_benchmark_runs_all_arms(dataset_dir: Path, tmp_path: Path) -> None:
    gate = tmp_path / "gate.jsonl"
    _write_jsonl(gate, [_record("g1")])
    # the sidecar arm needs the sidecar present for a clean run
    _write_jsonl(dataset_dir / "kd.jsonl", [{"id": "a", "positions": [[0, 0.9]]}])
    trainer = _FakeTrainer()

    def fake_gate(model: Path, dataset: Path, **kw: Any) -> dict[str, Any]:
        return {"accuracy": 0.5, "flag_accuracy": 0.5, "n_records": 1}

    results = run_three_arm_benchmark(
        config=TrainingConfig(gkd_teacher="Qwen/Qwen3-4B"),
        dataset_dir=dataset_dir,
        gate_dataset=gate,
        output_dir=tmp_path / "bench",
        train_fn=trainer,
        gate_fn=fake_gate,
    )
    assert set(results) == {"no-kd", "sidecar", "gkd", "lora64"}
    assert trainer.calls == ["no-kd", "sidecar", "gkd", "lora64"]


def test_render_markdown_table_has_all_arms_and_metrics() -> None:
    results = {
        "no-kd": {
            "arm": "no-kd",
            "metrics": {"accuracy": 0.5, "flag_accuracy": 0.4, "n_records": 26},
        },
        "sidecar": {
            "arm": "sidecar",
            "metrics": {"accuracy": 0.4, "flag_accuracy": 0.3, "n_records": 26},
        },
        "gkd": {"arm": "gkd", "metrics": {"accuracy": 0.6, "flag_accuracy": 0.5, "n_records": 26}},
        "lora64": {
            "arm": "lora64",
            "metrics": {"accuracy": 0.55, "flag_accuracy": 0.45, "n_records": 26},
        },
    }
    table = render_markdown_table(results)
    assert "| arm " in table
    for arm in ("no-kd", "sidecar", "gkd", "lora64"):
        assert arm in table
    assert "0.5000" in table  # accuracy of no-kd
    assert "26" in table  # n_records honesty rule
