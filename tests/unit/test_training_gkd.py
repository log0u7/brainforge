"""GKD training strategy: on-policy distillation against a local teacher.

Covers the contract of ADR-0020:
- the ``gkd`` strategy dispatches from ``train_student``;
- the teacher must be named and must share the student tokenizer;
- the GKDConfig knobs map from the TrainingConfig fields;
- the trainer call carries the dataset rows, the LoRA and quantization plans;
- heavy imports stay lazy (no module-level trl/torch imports).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError

# ---------------------------------------------------------------------------
# Stub of trl.experimental.gkd: records every call so the tests assert the
# trainer wiring without importing torch/trl (the training extra may be absent).
# ---------------------------------------------------------------------------


class _FakeGKDConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeGKDTrainer:
    """Captures the constructor arguments and fakes the training loop."""

    last_init: dict | None = None

    def __init__(self, **kwargs):
        _FakeGKDTrainer.last_init = kwargs

    def train(self):
        return SimpleNamespace(metrics={"train_loss": 0.5})

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def fake_gkd_module(monkeypatch):
    """Hermetic trl/peft/datasets stubs: the trainer wiring runs without the
    training extra (CI installs plain uv sync, trl only ships with it)."""
    module = SimpleNamespace(GKDConfig=_FakeGKDConfig, GKDTrainer=_FakeGKDTrainer)
    experimental = SimpleNamespace(gkd=module)
    trl_stub = SimpleNamespace(experimental=experimental)
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    monkeypatch.setitem(sys.modules, "trl.experimental", experimental)
    monkeypatch.setitem(sys.modules, "trl.experimental.gkd", module)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    from brainforge.training import hardware

    plan = SimpleNamespace(quantization="none")
    monkeypatch.setattr(hardware, "plan_for_config", lambda config, kd=False: plan)
    _FakeGKDTrainer.last_init = None
    return module


def _write_dataset(dataset_dir: Path, n_rows: int = 2) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"rec-{i}",
            "messages": [
                {"role": "user", "content": f"prompt {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ],
        }
        for i in range(n_rows)
    ]
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return dataset_dir


def _config(**overrides: Any) -> TrainingConfig:
    fields: dict[str, Any] = {
        "base_model": "Qwen/Qwen3-1.7B",
        "gkd_teacher": "Qwen/Qwen3-4B",
        "gkd_lmbda": 1.0,
        "gkd_beta": 1.0,
        "gkd_seq_kd": False,
        "quantization": "none",
        "device": "cpu",
        "strategy": "gkd",
    }
    fields.update(overrides)
    return TrainingConfig(**fields)


def test_train_gkd_builds_trainer_with_config_knobs(fake_gkd_module, tmp_path):
    from brainforge.training.gkd import train_gkd

    dataset_dir = _write_dataset(tmp_path / "datasets" / "prepared")
    train_gkd(_config(), dataset_dir, tmp_path / "out")

    init = _FakeGKDTrainer.last_init
    assert init is not None, "the trainer was never constructed"
    assert init["model"] == "Qwen/Qwen3-1.7B"
    assert init["teacher_model"] == "Qwen/Qwen3-4B"
    config = init["args"]
    assert config.kwargs["lmbda"] == 1.0
    assert config.kwargs["beta"] == 1.0
    assert config.kwargs["seq_kd"] is False
    assert config.kwargs["teacher_model_name_or_path"] == "Qwen/Qwen3-4B"


def test_train_gkd_requires_teacher(fake_gkd_module, tmp_path):
    from brainforge.training.gkd import train_gkd

    dataset_dir = _write_dataset(tmp_path / "datasets" / "prepared")
    with pytest.raises(BrainforgeError, match="gkd_teacher"):
        train_gkd(_config(gkd_teacher=""), dataset_dir, tmp_path / "out")


def test_train_gkd_rejects_empty_dataset(fake_gkd_module, tmp_path):
    from brainforge.training.gkd import train_gkd

    dataset_dir = tmp_path / "datasets" / "empty"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(BrainforgeError, match="empty dataset split"):
        train_gkd(_config(), dataset_dir, tmp_path / "out")


def test_train_gkd_writes_summary(fake_gkd_module, tmp_path):
    from brainforge.training.gkd import train_gkd

    dataset_dir = _write_dataset(tmp_path / "datasets" / "prepared")
    out = train_gkd(_config(), dataset_dir, tmp_path / "out")
    summary = json.loads((tmp_path / "out" / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "trl-gkd"
    assert summary["gkd_teacher"] == "Qwen/Qwen3-4B"
    assert summary["n_prompts"] == 2
    assert out["output_dir"] == str(tmp_path / "out")


def test_train_student_dispatches_to_gkd(auto_fit_stub, monkeypatch, tmp_path):
    from brainforge.cli.train_cmd import train_student
    from brainforge.training import gkd as gkd_module

    dataset_dir = _write_dataset(tmp_path / "datasets" / "prepared", n_rows=1)
    calls: dict[str, Any] = {}

    def _fake_train_gkd(training, dataset_dir, output_dir, resume=False):
        calls["strategy"] = training.strategy
        return {"output_dir": str(output_dir)}

    monkeypatch.setattr(gkd_module, "train_gkd", _fake_train_gkd)
    summary = train_student(_config(strategy="gkd"), dataset_dir, tmp_path / "out")
    assert calls["strategy"] == "gkd"
    assert summary["output_dir"] == str(tmp_path / "out")
