"""Hardware detection and training-plan resolution tests.

Fake torch modules are injected so the resolver runs without the training
extra or a GPU.
"""

import sys
import types

import pytest

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError
from brainforge.training.hardware import (
    DeviceInfo,
    default_dtype,
    detect_devices,
    plan_for_config,
    resolve_device,
    resolve_plan,
    torch_dtype,
    validate_plan,
)

GIB = 1024**3


def _device(kind: str, *, bf16: bool = False) -> DeviceInfo:
    return DeviceInfo(
        kind=kind,
        device="cpu" if kind == "cpu" else f"{kind}:0",
        name=f"fake {kind}",
        total_memory=None,
        free_memory=None,
        bf16=bf16,
        fp16=kind != "cpu",
    )


def _fake_torch(
    monkeypatch,
    *,
    cuda: bool = False,
    hip: str | None = None,
    xpu: bool = False,
    mps: bool = False,
    bf16: bool = True,
) -> None:
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=hip)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        is_bf16_supported=lambda: bf16,
        get_device_name=lambda index: "Fake GPU",
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=24 * GIB),
        mem_get_info=lambda index: (16 * GIB, 24 * GIB),
    )
    torch.xpu = types.SimpleNamespace(
        is_available=lambda: xpu,
        is_bf16_supported=lambda: bf16,
        get_device_name=lambda index: "Fake XPU",
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=16 * GIB),
        mem_get_info=lambda index: (12 * GIB, 16 * GIB),
    )
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps),
    )
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", torch)


def _fake_runtime(monkeypatch, name: str) -> None:
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_missing_torch_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(BrainforgeError, match="training extra"):
        detect_devices()


def test_cpu_is_always_available(monkeypatch):
    _fake_torch(monkeypatch)
    devices = detect_devices()
    assert [device.kind for device in devices] == ["cpu"]
    assert devices[0].device == "cpu"
    assert devices[0].bf16 is False
    assert default_dtype(devices[0]) == "float32"


def test_detect_cuda_and_memory(monkeypatch):
    _fake_torch(monkeypatch, cuda=True)
    devices = detect_devices()
    assert [device.kind for device in devices] == ["cuda", "cpu"]
    cuda = devices[0]
    assert cuda.device == "cuda:0"
    assert cuda.total_memory == 24 * GIB
    assert cuda.free_memory == 16 * GIB
    assert cuda.bf16 is True
    assert default_dtype(cuda) == "bfloat16"


def test_detect_rocm_from_hip_version(monkeypatch):
    _fake_torch(monkeypatch, cuda=True, hip="6.3")
    devices = detect_devices()
    assert devices[0].kind == "rocm"
    assert resolve_device("auto", devices).kind == "rocm"
    assert resolve_device("cuda", devices).kind == "rocm"


def test_detect_xpu(monkeypatch):
    _fake_torch(monkeypatch, xpu=True)
    devices = detect_devices()
    assert [device.kind for device in devices] == ["xpu", "cpu"]
    assert default_dtype(devices[0]) == "bfloat16"


def test_detect_xpu_without_bf16_falls_back_to_fp16(monkeypatch):
    _fake_torch(monkeypatch, xpu=True, bf16=False)
    assert default_dtype(detect_devices()[0]) == "float16"


def test_detect_mps(monkeypatch):
    _fake_torch(monkeypatch, mps=True)
    devices = detect_devices()
    assert [device.kind for device in devices] == ["mps", "cpu"]
    assert default_dtype(devices[0]) == "float16"


def test_auto_prefers_cuda_over_xpu_and_mps(monkeypatch):
    _fake_torch(monkeypatch, cuda=True, xpu=True, mps=True)
    assert resolve_device("auto", detect_devices()).kind == "cuda"


def test_auto_prefers_xpu_over_mps(monkeypatch):
    _fake_torch(monkeypatch, xpu=True, mps=True)
    assert resolve_device("auto", detect_devices()).kind == "xpu"


def test_auto_tpu_only_fails_fast(monkeypatch):
    _fake_torch(monkeypatch)
    _fake_runtime(monkeypatch, "torch_xla")
    devices = detect_devices()
    assert [device.kind for device in devices] == ["cpu", "tpu"]
    with pytest.raises(BrainforgeError, match="TPU"):
        resolve_device("auto", devices)


def test_auto_npu_only_fails_fast(monkeypatch):
    _fake_torch(monkeypatch)
    _fake_runtime(monkeypatch, "torch_npu")
    with pytest.raises(BrainforgeError, match="NPU"):
        resolve_device("auto", detect_devices())


def test_explicit_cpu_works_on_tpu_only(monkeypatch):
    _fake_torch(monkeypatch)
    _fake_runtime(monkeypatch, "torch_xla")
    assert resolve_device("cpu", detect_devices()).kind == "cpu"


def test_explicit_missing_device_lists_available(monkeypatch):
    _fake_torch(monkeypatch)
    with pytest.raises(BrainforgeError, match="available"):
        resolve_device("cuda", detect_devices())


def test_auto_quantization_uses_4bit_on_cuda(monkeypatch):
    _fake_torch(monkeypatch, cuda=True)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(TrainingConfig(), device, quantization="auto", backend="auto")
    assert plan.quantization == "4bit"
    assert plan.backend == "trl"
    assert plan.effective_strategy == "qlora"


def test_auto_quantization_uses_none_on_cpu(monkeypatch):
    _fake_torch(monkeypatch)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(TrainingConfig(), device, quantization="auto", backend="auto")
    assert plan.quantization == "none"
    assert plan.effective_strategy == "lora"
    assert plan.peft is True
    assert plan.dtype == "float32"


def test_auto_quantization_uses_none_on_mps(monkeypatch):
    _fake_torch(monkeypatch, mps=True)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(TrainingConfig(), device, quantization="auto", backend="auto")
    assert plan.quantization == "none"
    assert plan.effective_strategy == "lora"
    assert plan.dtype == "float16"


def test_full_strategy_forces_none_and_no_peft(monkeypatch):
    _fake_torch(monkeypatch)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(
        TrainingConfig(strategy="full", quantization="4bit"),
        device,
        strategy="full",
        quantization="4bit",
        backend="auto",
    )
    assert plan.quantization == "none"
    assert plan.effective_strategy == "full"
    assert plan.peft is False


def test_kd_forces_kd_backend(monkeypatch):
    _fake_torch(monkeypatch, cuda=True)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(
        TrainingConfig(backend="unsloth"), device, quantization="auto", backend="unsloth", kd=True
    )
    assert plan.backend == "kd"


def test_validate_rejects_4bit_on_cpu(monkeypatch):
    _fake_torch(monkeypatch)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(TrainingConfig(), device, quantization="4bit", backend="auto")
    errors = validate_plan(plan, device, TrainingConfig())
    assert any("quantization none" in error for error in errors)


def test_validate_rejects_8bit_on_mps(monkeypatch):
    _fake_torch(monkeypatch, mps=True)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(TrainingConfig(), device, quantization="8bit", backend="auto")
    assert any(
        "quantization none" in error for error in validate_plan(plan, device, TrainingConfig())
    )


def test_validate_rejects_unsloth_on_mps(monkeypatch):
    _fake_torch(monkeypatch, mps=True)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(
        TrainingConfig(backend="unsloth"), device, quantization="none", backend="unsloth"
    )
    assert any(
        "Unsloth requires CUDA" in error for error in validate_plan(plan, device, TrainingConfig())
    )


def test_validate_rejects_8bit_unsloth(monkeypatch):
    _fake_torch(monkeypatch, cuda=True)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(
        TrainingConfig(backend="unsloth"), device, quantization="8bit", backend="unsloth"
    )
    assert any("4bit or none" in error for error in validate_plan(plan, device, TrainingConfig()))


def test_validate_rejects_flash_attention_on_cpu(monkeypatch):
    _fake_torch(monkeypatch)
    device = resolve_device("auto", detect_devices())
    config = TrainingConfig(attn_implementation="flash_attention_2")
    plan = resolve_plan(config, device, quantization="none", backend="auto")
    assert any("flash_attention_2" in error for error in validate_plan(plan, device, config))


def test_plan_for_config_raises_on_invalid_combination(monkeypatch):
    _fake_torch(monkeypatch)
    config = TrainingConfig(quantization="4bit")
    with pytest.raises(BrainforgeError, match="quantization none"):
        plan_for_config(config)


def test_torch_dtype_resolves_lazy_attribute(monkeypatch):
    _fake_torch(monkeypatch)
    device = resolve_device("auto", detect_devices())
    plan = resolve_plan(TrainingConfig(), device, quantization="none", backend="auto")
    assert torch_dtype(plan) == "float32"
