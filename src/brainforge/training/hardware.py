"""Device detection and the single training-plan resolver.

Heavy imports stay lazy so the base package and unit suite run without the
``training`` extra.
"""

import importlib.util
import platform
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, cast

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError

DeviceKind = Literal["cuda", "rocm", "xpu", "mps", "cpu", "tpu", "npu"]
Dtype = Literal["bfloat16", "float16", "float32"]
Quantization = Literal["4bit", "8bit", "none"]
Backend = Literal["trl", "unsloth", "kd"]
Strategy = Literal["qlora", "full"]
EffectiveStrategy = Literal["qlora", "lora", "full"]


@dataclass(frozen=True)
class DeviceInfo:
    kind: DeviceKind
    device: str
    name: str
    total_memory: int | None
    free_memory: int | None
    bf16: bool
    fp16: bool


@dataclass(frozen=True)
class TrainingPlan:
    backend: Backend
    device: str
    device_kind: DeviceKind
    dtype: Dtype
    quantization: Quantization
    peft: bool
    attn_implementation: str
    strategy: Strategy
    effective_strategy: EffectiveStrategy


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BrainforgeError(
            "torch is not installed; install the training extra first: uv sync --extra training"
        ) from exc
    return torch


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cuda_device(torch: Any) -> DeviceInfo | None:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return None
    hip = getattr(getattr(torch, "version", None), "hip", None)
    kind: DeviceKind = "rocm" if hip else "cuda"
    total = free = None
    name = "CUDA device"
    with suppress(AttributeError, IndexError, RuntimeError, TypeError):
        total = _safe_int(cuda.get_device_properties(0).total_memory)
    with suppress(AttributeError, IndexError, RuntimeError, TypeError):
        free = _safe_int(cuda.mem_get_info(0)[0])
    with suppress(AttributeError, IndexError, RuntimeError, TypeError):
        name = str(cuda.get_device_name(0))
    try:
        bf16 = bool(cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16 = False
    return DeviceInfo(
        kind=kind,
        device="cuda:0",
        name=name,
        total_memory=total,
        free_memory=free,
        bf16=bf16,
        fp16=True,
    )


def _xpu_device(torch: Any) -> DeviceInfo | None:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not getattr(xpu, "is_available", lambda: False)():
        return None
    total = free = None
    name = "Intel XPU"
    with suppress(AttributeError, IndexError, RuntimeError, TypeError):
        total = _safe_int(xpu.get_device_properties(0).total_memory)
    with suppress(AttributeError, IndexError, RuntimeError, TypeError):
        free = _safe_int(xpu.mem_get_info(0)[0])
    with suppress(AttributeError, IndexError, RuntimeError, TypeError):
        name = str(xpu.get_device_name(0))
    try:
        bf16 = bool(xpu.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16 = False
    return DeviceInfo(
        kind="xpu",
        device="xpu:0",
        name=name,
        total_memory=total,
        free_memory=free,
        bf16=bf16,
        fp16=True,
    )


def _mps_device(torch: Any) -> DeviceInfo | None:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not getattr(mps, "is_available", lambda: False)():
        return None
    return DeviceInfo(
        kind="mps",
        device="mps",
        name="Apple MPS",
        total_memory=None,
        free_memory=None,
        bf16=False,
        fp16=True,
    )


def _cpu_device() -> DeviceInfo:
    return DeviceInfo(
        kind="cpu",
        device="cpu",
        name=platform.processor() or platform.machine() or "CPU",
        total_memory=None,
        free_memory=None,
        bf16=False,
        fp16=False,
    )


def detect_devices() -> list[DeviceInfo]:
    """Detect supported accelerators plus CPU, and unsupported TPU/NPU runtimes."""
    torch = _import_torch()
    devices = [
        device
        for device in (_cuda_device(torch), _xpu_device(torch), _mps_device(torch))
        if device is not None
    ]
    devices.append(_cpu_device())
    if _module_available("torch_xla"):
        devices.append(DeviceInfo("tpu", "tpu", "Google TPU", None, None, bf16=False, fp16=False))
    if _module_available("torch_npu"):
        devices.append(DeviceInfo("npu", "npu", "Ascend NPU", None, None, bf16=False, fp16=False))
    return devices


def resolve_device(requested: str, devices: list[DeviceInfo]) -> DeviceInfo:
    """Resolve an explicit device or the auto order CUDA/ROCm, XPU, MPS, CPU."""
    if requested == "auto":
        for kind in ("cuda", "rocm", "xpu", "mps"):
            for device in devices:
                if device.kind == kind:
                    return device
        unsupported = [device for device in devices if device.kind in ("tpu", "npu")]
        if unsupported:
            names = " and ".join(device.kind.upper() for device in unsupported)
            raise BrainforgeError(
                f"{names} runtime detected but not supported in this release;"
                " use --device cpu to train on CPU"
            )
        return next(device for device in devices if device.kind == "cpu")
    wanted = ("cuda", "rocm") if requested == "cuda" else (requested,)
    for device in devices:
        if device.kind in wanted:
            return device
    available = ", ".join(device.kind for device in devices)
    raise BrainforgeError(
        f"requested device '{requested}' is not available (available: {available})"
    )


def default_dtype(device: DeviceInfo) -> Dtype:
    if device.kind in ("cuda", "rocm", "xpu"):
        return "bfloat16" if device.bf16 else "float16"
    if device.kind == "mps":
        return "float16"
    return "float32"


def resolve_quantization(requested: str, device_kind: DeviceKind) -> Quantization:
    """Resolve ``auto`` quantization for a device kind; explicit values pass through."""
    if requested != "auto":
        return cast(Quantization, requested)
    return "4bit" if device_kind in ("cuda", "rocm") else "none"


def resolve_plan(
    config: TrainingConfig,
    device: DeviceInfo,
    *,
    strategy: str | None = None,
    quantization: str | None = None,
    backend: str | None = None,
    kd: bool = False,
) -> TrainingPlan:
    """Select backend, dtype, quantization and strategy without validating."""
    chosen_strategy = cast(Strategy, strategy or config.strategy)
    chosen_quantization = resolve_quantization(quantization or config.quantization, device.kind)
    if chosen_strategy == "full":
        chosen_quantization = "none"
    chosen_backend = "kd" if kd else cast(Backend, backend or config.backend)
    if chosen_backend == "auto":
        chosen_backend = "trl"
    peft = chosen_strategy != "full"
    if not peft:
        effective: EffectiveStrategy = "full"
    elif chosen_quantization == "none":
        effective = "lora"
    else:
        effective = "qlora"
    return TrainingPlan(
        backend=chosen_backend,
        device=device.device,
        device_kind=device.kind,
        dtype=default_dtype(device),
        quantization=chosen_quantization,
        peft=peft,
        attn_implementation=config.attn_implementation,
        strategy=chosen_strategy,
        effective_strategy=effective,
    )


def validate_plan(plan: TrainingPlan, device: DeviceInfo, config: TrainingConfig) -> list[str]:
    """Return every hard error for the selected plan and device."""
    errors: list[str] = []
    if plan.quantization != "none" and device.kind not in ("cuda", "rocm"):
        errors.append(
            f"quantization {plan.quantization} is not supported on {device.kind};"
            " use quantization none in this release"
        )
    if plan.backend == "unsloth":
        if device.kind not in ("cuda", "rocm"):
            errors.append("Unsloth requires CUDA or ROCm; use the trl backend on this device")
        if plan.quantization == "8bit":
            errors.append(
                "unsloth backend supports 4bit or none quantization;"
                " use the trl backend for 8bit training"
            )
    if plan.attn_implementation == "flash_attention_2" and device.kind not in (
        "cuda",
        "rocm",
    ):
        errors.append("flash_attention_2 requires CUDA or ROCm; use sdpa or eager")
    return errors


def plan_for_config(config: TrainingConfig, *, kd: bool = False) -> TrainingPlan:
    """Detect, resolve and validate the plan for direct backend calls."""
    devices = detect_devices()
    device = resolve_device(str(getattr(config, "device", "auto")), devices)
    plan = resolve_plan(config, device, kd=kd)
    errors = validate_plan(plan, device, config)
    if errors:
        raise BrainforgeError("\n".join(errors))
    return plan


def plan_for_inference(
    quantization: str = "auto", attn_implementation: str = "sdpa"
) -> TrainingPlan:
    """Resolve a plan for inference paths (evaluation, chat, export)."""
    return plan_for_config(
        TrainingConfig(
            quantization=cast(Quantization, quantization),
            strategy="qlora",
            attn_implementation=attn_implementation,
        )
    )


def torch_dtype(plan: TrainingPlan) -> Any:
    """Resolve the lazy torch dtype attribute for a plan."""
    torch = _import_torch()
    return getattr(torch, plan.dtype)
