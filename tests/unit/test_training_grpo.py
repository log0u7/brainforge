"""GRPO SOTA recipe tests.

Fake torch/trl/transformers/peft/datasets modules are injected so the real
train_grpo code path runs without the training extra or a GPU.
"""

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from brainforge.config.models import TrainingConfig
from brainforge.training.grpo import train_grpo


class _FakeState:
    def __init__(self) -> None:
        self.log_history = [{"loss": 0.5}]


class _FakeGRPOTrainer:
    instances: ClassVar[list["_FakeGRPOTrainer"]] = []

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.state = _FakeState()
        type(self).instances.append(self)

    def train(self) -> None:
        pass

    def save_model(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeGRPOConfig:
    instances: ClassVar[list["_FakeGRPOConfig"]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)


class _FakeDataset:
    @classmethod
    def from_list(cls, rows):
        return {"rows": rows}


@pytest.fixture
def fake_training_modules(monkeypatch):
    _FakeGRPOTrainer.instances = []
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True
    )
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float16 = "float16"
    torch_mod.float32 = "float32"
    trl_mod = types.ModuleType("trl")
    trl_mod.GRPOConfig = _FakeGRPOConfig
    trl_mod.GRPOTrainer = _FakeGRPOTrainer
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
    return _FakeGRPOTrainer, _FakeGRPOConfig


def _dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "prepared"
    dataset_dir.mkdir()
    record = {
        "id": "x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
    }
    (dataset_dir / "train.jsonl").write_text(json.dumps(record) + "\n")
    return dataset_dir


def test_sota_recipe_defaults():
    config = TrainingConfig()
    assert config.grpo_loss_type == "dapo"
    assert config.grpo_epsilon_high == 0.28
    assert config.grpo_scale_rewards is False
    assert config.grpo_beta == 0.0


def test_sota_recipe_rejects_unknown_loss_type():
    with pytest.raises(ValidationError):
        TrainingConfig(grpo_loss_type="made_up")


def test_train_grpo_passes_sota_recipe(fake_training_modules, tmp_path):
    _, fake_config = fake_training_modules
    config = TrainingConfig(strategy="grpo")
    train_grpo(config, _dataset_dir(tmp_path), tmp_path / "run", reward="compliance")
    args = fake_config.instances[-1].kwargs
    # DAPO clip-higher: asymmetric clipping (epsilon 0.2 / epsilon_high 0.28)
    assert args["loss_type"] == "dapo"
    assert args["epsilon_high"] == 0.28
    # Dr. GRPO: unbiased baseline, no reward scaling
    assert args["scale_rewards"] == "none"
    # No KL penalty: pure policy-gradient objective
    assert args["beta"] == 0.0


def test_train_grpo_config_overrides_reach_trl(fake_training_modules, tmp_path):
    _, fake_config = fake_training_modules
    config = TrainingConfig(
        strategy="grpo",
        grpo_loss_type="grpo",
        grpo_epsilon_high=0.3,
        grpo_scale_rewards=True,
        grpo_beta=0.01,
    )
    train_grpo(config, _dataset_dir(tmp_path), tmp_path / "run", reward="compliance")
    args = fake_config.instances[-1].kwargs
    assert args["loss_type"] == "grpo"
    assert args["epsilon_high"] == 0.3
    assert args["scale_rewards"] == "group"
    assert args["beta"] == 0.01
def test_training_config_domain_field():
    """TrainingConfig carries the domain: the pack hook resolves the
    GRPO reward (ctf_flag) and the sidecar columns from it."""
    config = TrainingConfig()
    assert config.domain == ""
    assert TrainingConfig(domain="ctf").domain == "ctf"


def test_train_student_dispatches_grpo_with_domain(tmp_path, monkeypatch):
    """train_student passes the config's domain to train_grpo: without
    it the pack never loads, the reward never sees expected_flag (mr22:
    the ctf_flag reward got no sidecar -> TypeError before training)."""
    import brainforge.training.grpo as grpo_mod
    from brainforge.cli.train_cmd import train_student

    captured: dict = {}

    def fake_grpo(config, dataset_dir, output_dir, resume=False, domain=None):
        captured["domain"] = domain
        return {"output_dir": str(output_dir)}

    monkeypatch.setattr(grpo_mod, "train_grpo", fake_grpo)
    # 1.7B fits the detected cuda budget (the default 8B does not fit
    # 5.1GB and auto_fit_config refuses before dispatching)
    config = TrainingConfig(strategy="grpo", domain="ctf", base_model="Qwen/Qwen3-1.7B")
    train_student(config, tmp_path, tmp_path / "out")
    assert captured["domain"] == "ctf"


def test_train_student_dispatches_grpo_without_domain(tmp_path, monkeypatch):
    """No domain in the config: train_grpo gets None, the generic
    reward path still runs (explicit --grpo-reward)."""
    import brainforge.training.grpo as grpo_mod
    from brainforge.cli.train_cmd import train_student

    captured: dict = {}

    def fake_grpo(config, dataset_dir, output_dir, resume=False, domain=None):
        captured["domain"] = domain
        return {"output_dir": str(output_dir)}

    monkeypatch.setattr(grpo_mod, "train_grpo", fake_grpo)
    config = TrainingConfig(strategy="grpo", base_model="Qwen/Qwen3-1.7B")
    train_student(config, tmp_path, tmp_path / "out")
    assert captured["domain"] is None
