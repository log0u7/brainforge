"""Unsloth backend tests.

Fake unsloth/torch/trl/datasets modules are injected so the real
train_unsloth code path runs without the unsloth extra or a GPU.
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
from brainforge.errors import BrainforgeError
from brainforge.training.unsloth_backend import train_unsloth

runner = CliRunner()


class _FakeState:
    def __init__(self) -> None:
        self.log_history = [{"loss": 0.5}]


class _FakeSFTTrainer:
    instances: ClassVar[list["_FakeSFTTrainer"]] = []

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.state = _FakeState()
        type(self).instances.append(self)

    def train(self, resume_from_checkpoint=None) -> None:
        self.resume_arg = resume_from_checkpoint

    def save_model(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeModel:
    def save_pretrained(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeTokenizer:
    def save_pretrained(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeFastLanguageModel:
    calls: ClassVar[dict] = {}

    @staticmethod
    def from_pretrained(**kwargs):
        _FakeFastLanguageModel.calls["from_pretrained"] = kwargs
        return _FakeModel(), _FakeTokenizer()

    @staticmethod
    def get_peft_model(model, **kwargs):
        _FakeFastLanguageModel.calls["get_peft_model"] = kwargs
        return _FakeModel()


def _fake_modules(with_unsloth: bool = True) -> dict:
    modules = {
        "torch": types.ModuleType("torch"),
        "trl": types.ModuleType("trl"),
        "datasets": types.ModuleType("datasets"),
        "transformers": types.ModuleType("transformers"),
        "peft": types.ModuleType("peft"),
    }
    modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: True)
    modules["torch"].bfloat16 = "bfloat16"
    modules["trl"].SFTConfig = lambda **kwargs: types.SimpleNamespace(kwargs=kwargs)
    modules["trl"].SFTTrainer = _FakeSFTTrainer
    modules["datasets"].Dataset = type(
        "Dataset", (), {"from_list": classmethod(lambda cls, rows: {"rows": rows})}
    )
    modules["transformers"].BitsAndBytesConfig = lambda **kwargs: {"quant": kwargs}
    modules["peft"].LoraConfig = lambda **kwargs: {"lora": kwargs}
    if with_unsloth:
        unsloth_mod = types.ModuleType("unsloth")
        unsloth_mod.FastLanguageModel = _FakeFastLanguageModel
        modules["unsloth"] = unsloth_mod
    return modules


@pytest.fixture
def fake_training_modules(monkeypatch):
    _FakeSFTTrainer.instances = []
    _FakeFastLanguageModel.calls = {}
    for name, module in _fake_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    return _FakeSFTTrainer


def _dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "prepared"
    dataset_dir.mkdir()
    record = {"id": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {}}
    (dataset_dir / "train.jsonl").write_text(json.dumps(record) + "\n")
    return dataset_dir


def test_backend_defaults_to_trl():
    assert TrainingConfig().backend == "trl"


def test_backend_rejects_unknown_value():
    with pytest.raises(ValueError):
        TrainingConfig(backend="vllm")


def test_unsloth_backend_fails_cleanly_without_extra(monkeypatch, tmp_path):
    for name, module in _fake_modules(with_unsloth=False).items():
        monkeypatch.setitem(sys.modules, name, module)
    config = TrainingConfig(backend="unsloth")
    with pytest.raises(BrainforgeError, match="uv sync --extra unsloth"):
        train_unsloth(config, _dataset_dir(tmp_path), tmp_path / "run")


def test_unsloth_qlora_flow(fake_training_modules, tmp_path):
    config = TrainingConfig(backend="unsloth")
    summary = train_unsloth(config, _dataset_dir(tmp_path), tmp_path / "run")
    assert _FakeFastLanguageModel.calls["from_pretrained"]["load_in_4bit"] is True
    peft_call = _FakeFastLanguageModel.calls["get_peft_model"]
    assert peft_call["r"] == config.lora_rank
    assert peft_call["lora_alpha"] == config.lora_alpha
    trainer = fake_training_modules.instances[-1]
    assert isinstance(trainer.init_kwargs["model"], _FakeModel)
    assert isinstance(trainer.init_kwargs["tokenizer"], _FakeTokenizer)
    assert summary["backend"] == "unsloth"
    stored = json.loads((tmp_path / "run" / "train_summary.json").read_text())
    assert stored["backend"] == "unsloth"


def test_unsloth_full_strategy_skips_peft(fake_training_modules, tmp_path):
    config = TrainingConfig(backend="unsloth", strategy="full")
    summary = train_unsloth(config, _dataset_dir(tmp_path), tmp_path / "run")
    assert _FakeFastLanguageModel.calls["from_pretrained"]["load_in_4bit"] is False
    assert "get_peft_model" not in _FakeFastLanguageModel.calls
    assert summary["strategy"] == "full"


def test_unsloth_rejects_8bit_quantization(fake_training_modules, tmp_path):
    config = TrainingConfig(backend="unsloth", quantization="8bit")
    with pytest.raises(BrainforgeError, match="4bit or none"):
        train_unsloth(config, _dataset_dir(tmp_path), tmp_path / "run")


def test_cli_backend_flag_dispatches_unsloth(fake_training_modules, project_env):
    dataset_dir = _dataset_dir(project_env)
    result = runner.invoke(
        app, ["train", "run", "--dataset-dir", str(dataset_dir), "--backend", "unsloth"]
    )
    assert result.exit_code == 0, result.output
    assert "from_pretrained" in _FakeFastLanguageModel.calls
    assert "backend: unsloth" in result.output


def test_cli_default_backend_uses_qlora(fake_training_modules, project_env):
    dataset_dir = _dataset_dir(project_env)
    result = runner.invoke(app, ["train", "run", "--dataset-dir", str(dataset_dir)])
    assert result.exit_code == 0, result.output
    assert "from_pretrained" not in _FakeFastLanguageModel.calls
    assert "backend: trl" in result.output
