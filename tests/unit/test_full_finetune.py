"""Full fine-tuning strategy tests.

Fake torch/trl/transformers/peft/datasets modules are injected so the real
train_qlora code path runs without the training extra or a GPU.
"""

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from brainforge.cli import app
from brainforge.config.models import TrainingConfig
from brainforge.training.qlora import train_qlora

runner = CliRunner()


class _FakeState:
    def __init__(self) -> None:
        self.log_history = [{"loss": 0.5}]


class _FakeSFTTrainer:
    instances: ClassVar[list["_FakeSFTTrainer"]] = []

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.state = _FakeState()
        self.resume_arg = None
        type(self).instances.append(self)

    def train(self, resume_from_checkpoint=None) -> None:
        self.resume_arg = resume_from_checkpoint

    def save_model(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeSFTConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeDataset:
    @classmethod
    def from_list(cls, rows):
        return {"rows": rows}


@pytest.fixture
def fake_training_modules(monkeypatch):
    _FakeSFTTrainer.instances = []
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True
    )
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float16 = "float16"
    torch_mod.float32 = "float32"
    trl_mod = types.ModuleType("trl")
    trl_mod.SFTConfig = _FakeSFTConfig
    trl_mod.SFTTrainer = _FakeSFTTrainer
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = _FakeDataset
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.BitsAndBytesConfig = lambda **kwargs: {"quant": kwargs}
    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = lambda **kwargs: {"lora": kwargs}
    for name, module in (
        ("torch", torch_mod),
        ("trl", trl_mod),
        ("datasets", datasets_mod),
        ("transformers", transformers_mod),
        ("peft", peft_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return _FakeSFTTrainer


def _dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "prepared"
    dataset_dir.mkdir()
    record = {"id": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {}}
    (dataset_dir / "train.jsonl").write_text(json.dumps(record) + "\n")
    return dataset_dir


def test_strategy_defaults_to_qlora():
    assert TrainingConfig().strategy == "qlora"


def test_strategy_rejects_unknown_value():
    with pytest.raises(ValueError):
        TrainingConfig(strategy="lora")


def test_full_strategy_skips_quantization_and_peft(fake_training_modules, tmp_path):
    config = TrainingConfig(strategy="full")
    summary = train_qlora(config, _dataset_dir(tmp_path), tmp_path / "run")
    trainer = fake_training_modules.instances[-1]
    assert "quantization_config" not in trainer.init_kwargs
    assert "peft_config" not in trainer.init_kwargs
    assert summary["strategy"] == "full"
    stored = json.loads((tmp_path / "run" / "train_summary.json").read_text())
    assert stored["strategy"] == "full"


def test_qlora_strategy_keeps_quantization_and_peft(fake_training_modules, tmp_path):
    config = TrainingConfig()
    summary = train_qlora(config, _dataset_dir(tmp_path), tmp_path / "run")
    trainer = fake_training_modules.instances[-1]
    assert trainer.init_kwargs["quantization_config"] is not None
    assert trainer.init_kwargs["peft_config"] is not None
    assert summary["strategy"] == "qlora"
    assert summary["device_kind"] == "cuda"
    assert summary["dtype"] == "bfloat16"
    assert summary["quantization"] == "4bit"
    assert summary["effective_strategy"] == "qlora"
    stored = json.loads((tmp_path / "run" / "train_summary.json").read_text())
    assert stored["strategy"] == "qlora"
    assert stored["device"] == "cuda:0"


def test_cli_strategy_flag_overrides_config(fake_training_modules, project_env, auto_fit_stub):
    dataset_dir = _dataset_dir(project_env)
    result = runner.invoke(
        app, ["train", "run", "--dataset-dir", str(dataset_dir), "--strategy", "full"]
    )
    assert result.exit_code == 0, result.output
    trainer = fake_training_modules.instances[-1]
    assert "peft_config" not in trainer.init_kwargs
    assert "strategy: full" in result.output


def test_cli_strategy_rejects_invalid_value(project_env):
    result = runner.invoke(app, ["train", "run", "--strategy", "lora"])
    assert result.exit_code != 0
