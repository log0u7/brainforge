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


def test_dependency_checks_kd_does_not_require_trl(monkeypatch):
    from brainforge.training import preflight

    installed = {"transformers", "datasets", "peft"}
    monkeypatch.setattr(preflight, "_available", lambda name: name in installed)
    device = preflight.DeviceInfo("cpu", "cpu", "CPU", None, None, False, False)
    config = TrainingConfig(quantization="none", strategy="qlora")
    checks: list[preflight.Check] = []
    preflight._dependency_checks(preflight.resolve_plan(config, device, kd=True), checks)
    assert not [check for check in checks if check.status == "fail"]
    trl_checks: list[preflight.Check] = []
    trl_plan = preflight.resolve_plan(config, device, backend="trl")
    preflight._dependency_checks(trl_plan, trl_checks)
    assert "training.trl" in [check.id for check in trl_checks if check.status == "fail"]


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


def test_resolve_model_dir_accepts_hub_id(tmp_path, monkeypatch):
    import pytest

    from brainforge.errors import BrainforgeError
    from brainforge.training.models import _resolve_model_dir

    local = tmp_path / "local-model"
    local.mkdir()
    assert _resolve_model_dir(local) == local

    fake = tmp_path / "snapshot"
    fake.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda name: str(fake))
    assert _resolve_model_dir("Qwen/Qwen3-1.7B") == fake

    with pytest.raises(BrainforgeError):
        _resolve_model_dir(tmp_path / "does-not-exist")


def test_resolve_model_dir_wraps_hub_errors(tmp_path, monkeypatch):
    import pytest

    from brainforge.errors import BrainforgeError
    from brainforge.training.models import _resolve_model_dir

    def boom(name):
        raise ValueError("Invalid model id")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    with pytest.raises(BrainforgeError):
        _resolve_model_dir("org/a/b/c")


def test_auto_fit_no_adjustment_needed():
    """Model fits at requested config: no changes, no warnings."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=1_700_000_000, hidden=2048, layers=28)
    config = _fake_training_config(max_length=4096, batch_size=1, gradient_accumulation=8)
    devices = [_fake_device(memory_total=10 * 1024**3)]
    result = _auto_fit(model, config, devices)
    assert result.max_length == 4096
    assert result.batch_size == 1
    assert not result.adjustments
    assert not result.cpu_offload


def test_auto_fit_reduces_seq_length():
    """8B model at default seq overflows 10GB: auto-fit drops seq, fits."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=8_000_000_000, hidden=4096, layers=36)
    config = _fake_training_config()
    devices = [_fake_device(memory_total=8 * 1024**3)]
    result = _auto_fit(model, config, devices)
    assert result.max_length < 4096
    assert result.max_length >= 1024
    assert any("max_length" in a for a in result.adjustments)
    assert not result.cpu_offload


def test_auto_fit_cpu_offload_warning():
    """8B at 2048 still overflows GPU but fits in system RAM: CPU offload warning."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=8_000_000_000, hidden=4096, layers=36)
    config = _fake_training_config(max_length=4096, batch_size=1, gradient_accumulation=8)
    gpu = _fake_device(memory_total=4 * 1024**3)
    result = _auto_fit(model, config, [gpu], system_ram=32 * 1024**3)
    assert result.cpu_offload
    assert any("offload" in a.lower() or "slow" in a.lower() for a in result.adjustments)


def test_auto_fit_impossible():
    """Model too large for any available device: explicit error."""
    import pytest

    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=80_000_000_000, hidden=8192, layers=80)
    config = _fake_training_config(max_length=4096, batch_size=1, gradient_accumulation=8)
    devices = [_fake_device(memory_total=8 * 1024**3)]
    with pytest.raises(BrainforgeError, match=r"impossible|too large|no fit"):
        _auto_fit(model, config, devices, system_ram=8 * 1024**3)


def test_auto_fit_warmup_warning():
    """warmup_steps >= total steps: LR stays at 0, training is a no-op."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=1_700_000_000, hidden=2048, layers=28)
    config = _fake_training_config(warmup_steps=5)
    devices = [_fake_device(memory_total=10 * 1024**3)]
    result = _auto_fit(model, config, devices, total_steps=3)
    assert any("warmup" in a.lower() or "learning rate" in a.lower() for a in result.adjustments)


def test_auto_fit_agnostic_mps():
    """Works with MPS device info (Apple Silicon): no CUDA-specific code."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=1_700_000_000, hidden=2048, layers=28)
    config = _fake_training_config(max_length=4096, batch_size=1, gradient_accumulation=8)
    devices = [_fake_device(memory_total=16 * 1024**3, kind="mps")]
    result = _auto_fit(model, config, devices)
    assert result.max_length == 4096
    assert not result.adjustments


def test_user_set_seq_respected_with_warning():
    """User explicitly sets max_length != default: auto-fit respects it,
    warns if CPU offload needed, suggests better value."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=8_000_000_000, hidden=4096, layers=36)
    config = _fake_training_config(max_length=4096, batch_size=1, gradient_accumulation=8)
    gpu = _fake_device(memory_total=4 * 1024**3)
    result = _auto_fit(model, config, [gpu], system_ram=32 * 1024**3)
    # user set 4096 (non-default): respected
    assert result.max_length == 4096
    # but warned about CPU offload
    assert result.cpu_offload
    assert any("2048" in a or "seq" in a.lower() for a in result.adjustments)


def test_auto_fit_user_override_preserved():
    """User explicitly sets max_length=2048: auto-fit must not change it,
    even if 4096 would also fit."""
    from brainforge.training.preflight import _auto_fit

    model = _fake_model_info(params=8_000_000_000, hidden=4096, layers=36)
    config = _fake_training_config(max_length=2048, batch_size=1, gradient_accumulation=8)
    devices = [_fake_device(memory_total=10 * 1024**3)]
    result = _auto_fit(model, config, devices)
    assert result.max_length == 2048
    assert not result.adjustments


def _fake_model_info(params: int, hidden: int, layers: int) -> ModelInfo:
    return ModelInfo(
        id="test/model",
        source="local",
        architecture="llama",
        model_type="llama",
        params=params,
        max_position_embeddings=8192,
        hidden_size=hidden,
        num_layers=layers,
        chat_template=True,
        license=None,
        gated=False,
        private=False,
        pre_quantized=False,
        remote_code=False,
        safetensors=True,
        pipeline_tag="text-generation",
        num_attention_heads=hidden // 128,
    )


def _fake_training_config(**overrides) -> TrainingConfig:
    defaults = TrainingConfig()
    base = {
        "max_length": defaults.max_length,
        "batch_size": defaults.batch_size,
        "gradient_accumulation": defaults.gradient_accumulation,
        "warmup_steps": 0,
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _fake_device(memory_total: int, kind: str = "cuda") -> DeviceInfo:
    return DeviceInfo(
        kind=kind,
        device=f"test-{kind}",
        name=f"test-{kind}",
        total_memory=memory_total,
        free_memory=memory_total,
        bf16=True,
        fp16=True,
    )


def test_auto_fit_config_public_helper(tmp_path, monkeypatch):
    """auto_fit_config: fit to hardware, apply result, return warnings list."""
    model = _fake_model_info(params=1_700_000_000, hidden=2048, layers=28)
    device = _fake_device(memory_total=10 * 1024**3)
    monkeypatch.setattr(preflight, "detect_devices", lambda: [device])
    monkeypatch.setattr(preflight, "inspect_model", lambda _m: model)
    config = _fake_training_config()  # defaults: max_length 1024, batch 1
    train = tmp_path / "train.jsonl"
    train.write_text('{"messages": []}\n' * 32)
    new_config, warnings = preflight.auto_fit_config(config, train.parent)
    assert new_config.max_length == 4096  # fits at 10GB, seq upgraded
    assert new_config.batch_size in (1, 2)  # largest batch that fits
    assert any("max_length" in w for w in warnings)


def test_auto_fit_config_warmup_warning(tmp_path, monkeypatch):
    """auto_fit_config: warmup_steps >= computed steps (post-fit) warns."""
    model = _fake_model_info(params=1_700_000_000, hidden=2048, layers=28)
    device = _fake_device(memory_total=10 * 1024**3)
    monkeypatch.setattr(preflight, "detect_devices", lambda: [device])
    monkeypatch.setattr(preflight, "inspect_model", lambda _m: model)
    config = _fake_training_config(warmup_steps=5)
    train = tmp_path / "train.jsonl"
    train.write_text('{"messages": []}\n' * 3)  # steps = ceil(3/(1*16))*3 = 3
    _config, warnings = preflight.auto_fit_config(config, train.parent)
    assert any("warmup" in w.lower() for w in warnings)


def test_train_student_applies_auto_fit(tmp_path, monkeypatch):
    """train_student wires auto_fit_config: captured config carries adjusted seq/batch."""
    from brainforge.cli.train_cmd import train_student

    model = _fake_model_info(params=8_000_000_000, hidden=4096, layers=36)
    device = _fake_device(memory_total=8 * 1024**3)
    calls = {"inspect": 0}

    def fake_inspect(_m):
        calls["inspect"] += 1
        return model

    monkeypatch.setattr(preflight, "detect_devices", lambda: [device])
    monkeypatch.setattr(preflight, "inspect_model", fake_inspect)
    train = tmp_path / "train.jsonl"
    train.write_text('{"messages": []}\n' * 64)
    captured: dict = {}

    def fake_qlora(config, dataset_dir, output_dir, resume=False):
        captured["config"] = config
        return {"output_dir": str(output_dir)}

    import brainforge.training.qlora as qlora_mod

    monkeypatch.setattr(qlora_mod, "train_qlora", fake_qlora)
    train_student(_fake_training_config(), train.parent, tmp_path / "out")
    assert calls["inspect"] == 1  # wiring ran exactly once
    assert captured["config"].max_length < 4096  # 8B on 8GB: seq reduced
    assert captured["config"].max_length >= 1024
