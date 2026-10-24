"""CLI tests for `brainforge train check`."""

import json
import sys
import types
from pathlib import Path

from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()


def _fake_env(monkeypatch) -> None:
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = type(
        "AutoModelForCausalLM",
        (),
        {"_model_mapping": {"llama": "LlamaForCausalLM"}},
    )
    for name, module in (
        ("torch", torch),
        ("transformers", transformers),
        ("peft", types.ModuleType("peft")),
        ("trl", types.ModuleType("trl")),
        ("datasets", types.ModuleType("datasets")),
        ("bitsandbytes", types.ModuleType("bitsandbytes")),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 4096,
                "hidden_size": 64,
                "num_hidden_layers": 2,
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"})
    )
    (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


def test_train_check_text_output(project_env, monkeypatch):
    _fake_env(monkeypatch)
    result = runner.invoke(app, ["train", "check", "--model", str(_model_dir(project_env))])
    assert result.exit_code == 0, result.output
    assert "model:" in result.output
    assert "hardware:" in result.output
    assert "plan:" in result.output
    assert "pass" in result.output


def test_train_check_json_output(project_env, monkeypatch):
    _fake_env(monkeypatch)
    result = runner.invoke(
        app,
        ["train", "check", "--model", str(_model_dir(project_env)), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["model"]["architecture"] == "LlamaForCausalLM"
    assert payload["hardware"]["kind"] == "cpu"


def test_train_check_strict_fails_on_warning(project_env, monkeypatch):
    _fake_env(monkeypatch)
    model = str(_model_dir(project_env))
    assert runner.invoke(app, ["train", "check", "--model", model]).exit_code == 0
    result = runner.invoke(app, ["train", "check", "--model", model, "--strict"])
    assert result.exit_code == 1


def test_train_check_missing_torch_fails_cleanly(project_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    result = runner.invoke(app, ["train", "check", "--model", str(_model_dir(project_env))])
    assert result.exit_code == 1
    assert "torch" in result.output
    assert "Traceback" not in result.output


def test_train_check_rejects_4bit_on_cpu(project_env, monkeypatch):
    _fake_env(monkeypatch)
    result = runner.invoke(
        app,
        [
            "train",
            "check",
            "--model",
            str(_model_dir(project_env)),
            "--device",
            "cpu",
            "--quantization",
            "4bit",
        ],
    )
    assert result.exit_code == 1
    assert "quantization none" in result.output


def test_train_check_missing_model_config_fails_cleanly(project_env, monkeypatch):
    _fake_env(monkeypatch)
    empty = project_env / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["train", "check", "--model", str(empty)])
    assert result.exit_code == 1
    assert "could not read" in result.output
    assert "Traceback" not in result.output
