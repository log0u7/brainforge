"""Preflight metadata, compatibility checks and memory-estimate tests."""

import json
import sys
import types
from pathlib import Path

import pytest

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError
from brainforge.training import preflight
from brainforge.training.hardware import DeviceInfo, resolve_plan
from brainforge.training.preflight import (
    ModelInfo,
    estimate_memory,
    inspect_model,
    report_to_dict,
    run_preflight,
)

GIB = 1024**3


def _fake_env(monkeypatch, *, mapping: dict | None = None) -> None:
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    transformers = types.ModuleType("transformers")
    if mapping is not None:
        transformers.AutoModelForCausalLM = type(
            "AutoModelForCausalLM", (), {"_model_mapping": mapping}
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


def _model_dir(tmp_path: Path, **overrides) -> Path:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "max_position_embeddings": 4096,
        "hidden_size": 64,
        "num_hidden_layers": 2,
    }
    config.update(overrides)
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"})
    )
    (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


def _training() -> TrainingConfig:
    return TrainingConfig(quantization="none", backend="trl")


def _info(**overrides) -> ModelInfo:
    values = {
        "id": "org/model",
        "source": "local",
        "architecture": "LlamaForCausalLM",
        "model_type": "llama",
        "params": 8_000_000_000,
        "max_position_embeddings": 4096,
        "hidden_size": 4096,
        "num_layers": 32,
        "chat_template": True,
        "license": "apache-2.0",
        "gated": False,
        "private": False,
        "pre_quantized": False,
        "remote_code": False,
        "safetensors": True,
        "pipeline_tag": "text-generation",
    }
    values.update(overrides)
    return ModelInfo(**values)


def _cpu_plan():
    device = DeviceInfo("cpu", "cpu", "CPU", None, None, bf16=False, fp16=False)
    return resolve_plan(_training(), device, quantization="none", backend="trl")


def _check(report, check_id: str):
    return next(check for check in report.checks if check.id == check_id)


def test_inspect_local_model_metadata(tmp_path):
    info = inspect_model(str(_model_dir(tmp_path)))
    assert info.source == "local"
    assert info.architecture == "LlamaForCausalLM"
    assert info.model_type == "llama"
    assert info.max_position_embeddings == 4096
    assert info.hidden_size == 64
    assert info.num_layers == 2
    assert info.chat_template is True
    assert info.safetensors is True
    assert info.pre_quantized is False
    assert info.remote_code is False


def test_inspect_local_model_requires_config(tmp_path):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    with pytest.raises(BrainforgeError, match=r"config\.json"):
        inspect_model(str(model_dir))


def test_local_preflight_passes_with_warnings(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    report = run_preflight(_training(), model=str(_model_dir(tmp_path)))
    assert report.ok is True
    assert _check(report, "hardware.device").status == "pass"
    assert _check(report, "model.chat_template").status == "pass"
    assert _check(report, "model.license").status == "warn"
    assert _check(report, "model.pipeline_tag").status == "warn"


def test_missing_chat_template_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path)
    (model_dir / "tokenizer_config.json").write_text("{}")
    report = run_preflight(_training(), model=str(model_dir))
    assert report.ok is False
    assert _check(report, "model.chat_template").status == "fail"


def test_pre_quantized_model_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path, quantization_config={"load_in_4bit": True})
    report = run_preflight(_training(), model=str(model_dir))
    assert report.ok is False
    assert _check(report, "model.pre_quantized").status == "fail"


def test_remote_code_model_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path, auto_map={"AutoModel": "custom.Model"})
    report = run_preflight(_training(), model=str(model_dir))
    assert report.ok is False
    assert _check(report, "model.remote_code").status == "fail"


def test_missing_weights_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    report = run_preflight(_training(), model=str(model_dir))
    assert report.ok is False
    assert _check(report, "model.weights").status == "fail"


def test_bin_only_weights_warn(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    (model_dir / "pytorch_model.bin").write_bytes(b"")
    report = run_preflight(_training(), model=str(model_dir))
    assert _check(report, "model.weights").status == "warn"


def test_max_length_over_limit_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path, max_position_embeddings=128)
    report = run_preflight(_training(), model=str(model_dir), max_length=1024)
    assert report.ok is False
    assert _check(report, "model.max_length").status == "fail"


def test_unknown_architecture_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = _model_dir(tmp_path, model_type="mystery", architectures=["MysteryForCausalLM"])
    report = run_preflight(_training(), model=str(model_dir))
    assert report.ok is False
    assert _check(report, "model.architecture").status == "fail"


def test_unavailable_mapping_skips_architecture(tmp_path, monkeypatch):
    _fake_env(monkeypatch)
    report = run_preflight(_training(), model=str(_model_dir(tmp_path)))
    assert _check(report, "model.architecture").status == "skip"


def test_dataset_missing_train_fails(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    report = run_preflight(_training(), model=str(_model_dir(tmp_path)), dataset_dir=dataset_dir)
    assert report.ok is False
    assert _check(report, "dataset.train").status == "fail"


def test_dataset_missing_validation_warns(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text('{"messages": []}\n')
    report = run_preflight(_training(), model=str(_model_dir(tmp_path)), dataset_dir=dataset_dir)
    assert _check(report, "dataset.train").status == "pass"
    assert _check(report, "dataset.validation").status == "warn"


def test_estimate_memory_returns_none_without_params():
    assert estimate_memory(_info(params=None), _cpu_plan(), 1024, 1) is None


def test_estimate_memory_fits_and_does_not_fit():
    plan = _cpu_plan()
    fits = estimate_memory(_info(), plan, 1024, 1, available_memory=64 * GIB)
    assert fits is not None
    assert fits.fits is True
    assert fits.total_gb > fits.weights_gb
    too_big = estimate_memory(_info(), plan, 1024, 1, available_memory=1 * GIB)
    assert too_big is not None
    assert too_big.fits is False


def test_strict_turns_warnings_into_failure(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    model_dir = str(_model_dir(tmp_path))
    assert run_preflight(_training(), model=model_dir).ok is True
    assert run_preflight(_training(), model=model_dir, strict=True).ok is False


def test_report_to_dict_is_json_serializable(tmp_path, monkeypatch):
    _fake_env(monkeypatch, mapping={"llama": "LlamaForCausalLM"})
    report = run_preflight(_training(), model=str(_model_dir(tmp_path)))
    payload = report_to_dict(report)
    assert payload["ok"] is True
    assert payload["model"]["architecture"] == "LlamaForCausalLM"
    json.dumps(payload)


def test_inspect_hub_uses_metadata_only(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
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
    tokenizer_path = tmp_path / "tokenizer_config.json"
    tokenizer_path.write_text(json.dumps({"chat_template": "{{ messages }}"}))
    calls: list[str] = []
    api_kwargs: list[dict] = []
    download_kwargs: list[dict] = []

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            api_kwargs.append(kwargs)
            return types.SimpleNamespace(
                siblings=[
                    types.SimpleNamespace(rfilename="config.json"),
                    types.SimpleNamespace(rfilename="model.safetensors"),
                ],
                safetensors=types.SimpleNamespace(total=123),
                card_data=types.SimpleNamespace(license="apache-2.0"),
                gated=False,
                private=False,
                pipeline_tag="text-generation",
            )

    def fake_download(repo_id, filename, **kwargs):
        calls.append(filename)
        download_kwargs.append(kwargs)
        if filename == "config.json":
            return str(config_path)
        if filename == "tokenizer_config.json":
            return str(tokenizer_path)
        raise preflight.EntryNotFoundError("missing")

    monkeypatch.setattr(preflight, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(preflight, "hf_hub_download", fake_download)
    info = inspect_model("org/model")
    assert info.params == 123
    assert info.source == "hub"
    assert info.chat_template is True
    assert calls == ["config.json", "tokenizer_config.json", "chat_template.jinja"]
    assert all("safetensors" not in name for name in calls)
    # huggingface_hub 1.x: token=True *requires* a token; anonymous public
    # access must stay possible (regression: TokenRequired crash).
    assert all(kwargs.get("token") is not True for kwargs in api_kwargs + download_kwargs)


def test_gated_hub_model_mentions_token(monkeypatch):
    class FakeGated(Exception):
        pass

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            raise FakeGated("gated")

    monkeypatch.setattr(preflight, "GatedRepoError", FakeGated)
    monkeypatch.setattr(preflight, "HfApi", lambda: FakeApi())
    with pytest.raises(BrainforgeError, match="HF_TOKEN"):
        inspect_model("org/model")
