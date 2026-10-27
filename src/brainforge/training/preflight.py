"""Preflight model metadata, compatibility checks and memory estimates.

The command reads Hub metadata and small config files only. It never downloads
model weights. Heavy training imports stay lazy.
"""

import importlib.util
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError
from brainforge.training.hardware import (
    DeviceInfo,
    TrainingPlan,
    detect_devices,
    resolve_device,
    resolve_plan,
    validate_plan,
)

_LICENSE_RE = re.compile(r"(?m)^license:\s*['\"]?([^'\"\n]+)")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    source: Literal["hub", "local"]
    architecture: str | None
    model_type: str | None
    params: int | None
    max_position_embeddings: int | None
    hidden_size: int | None
    num_layers: int | None
    chat_template: bool
    license: str | None
    gated: bool
    private: bool
    pre_quantized: bool
    remote_code: bool
    safetensors: bool
    pipeline_tag: str | None
    weights: bool = False
    num_attention_heads: int | None = None
    rope_scaling: bool = False


@dataclass(frozen=True)
class MemoryEstimate:
    weights_gb: float
    adapters_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    available_gb: float | None
    fits: bool | None


@dataclass(frozen=True)
class Check:
    id: str
    status: Literal["pass", "warn", "fail", "skip"]
    message: str


@dataclass
class CheckReport:
    ok: bool
    strict: bool
    model: ModelInfo | None
    hardware: DeviceInfo | None
    plan: TrainingPlan | None
    estimate: MemoryEstimate | None
    checks: list[Check]


def _available(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _config_value(config: dict, key: str) -> Any:
    if key in config:
        return config[key]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get(key)
    return None


def _read_json(path: Path, *, required: bool = True) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if not required:
            return {}
        raise BrainforgeError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        if not required:
            return {}
        raise BrainforgeError(f"{path} must contain a JSON object")
    return data


def _safetensors_params(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if header_size <= 0 or header_size > 100_000_000:
                return None
            header = json.loads(handle.read(header_size))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    total = 0
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(value, dict):
            continue
        shape = value.get("shape")
        if not isinstance(shape, list):
            continue
        count = 1
        for dimension in shape:
            count *= int(dimension)
        total += count
    return total or None


def _local_license(path: Path) -> str | None:
    readme = path / "README.md"
    if not readme.is_file():
        return None
    match = _LICENSE_RE.search(readme.read_text(encoding="utf-8", errors="replace"))
    return match.group(1).strip() if match else None


def _inspect_local(path: Path, model_id: str) -> ModelInfo:
    config = _read_json(path / "config.json")
    tokenizer_config = _read_json(path / "tokenizer_config.json", required=False)
    files = [item for item in path.iterdir() if item.is_file()]
    safetensors_files = [item for item in files if item.suffix == ".safetensors"]
    weights = bool(safetensors_files) or any(item.suffix == ".bin" for item in files)
    params = None
    if safetensors_files:
        counts = [_safetensors_params(item) for item in safetensors_files]
        if all(count is not None for count in counts):
            params = sum(count for count in counts if count is not None)
    architectures = config.get("architectures")
    architecture = (
        str(architectures[0]) if isinstance(architectures, list) and architectures else None
    )
    return ModelInfo(
        id=model_id,
        source="local",
        architecture=architecture,
        model_type=_string(config.get("model_type")),
        params=params,
        max_position_embeddings=_as_int(_config_value(config, "max_position_embeddings")),
        hidden_size=_as_int(_config_value(config, "hidden_size")),
        num_layers=_as_int(_config_value(config, "num_hidden_layers")),
        chat_template=tokenizer_config.get("chat_template") is not None
        or (path / "chat_template.jinja").is_file(),
        license=_local_license(path),
        gated=False,
        private=False,
        pre_quantized=config.get("quantization_config") is not None,
        remote_code=config.get("auto_map") is not None,
        safetensors=bool(safetensors_files),
        pipeline_tag=None,
        weights=weights,
        num_attention_heads=_as_int(_config_value(config, "num_attention_heads")),
        rope_scaling=_config_value(config, "rope_scaling") is not None,
    )


def _string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _hub_download(repo_id: str, filename: str) -> Path | None:
    try:
        return Path(hf_hub_download(repo_id, filename))
    except EntryNotFoundError:
        return None


def _hub_json(repo_id: str, filename: str, *, required: bool) -> dict:
    path = _hub_download(repo_id, filename)
    if path is None:
        if required:
            raise BrainforgeError(f"model '{repo_id}' has no {filename}")
        return {}
    return _read_json(path)


def _inspect_hub(repo_id: str) -> ModelInfo:
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except GatedRepoError as exc:
        raise BrainforgeError(
            f"model '{repo_id}' is gated; accept the license and set HF_TOKEN"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise BrainforgeError(
            f"model '{repo_id}' was not found on the Hub (check the id or run"
            " 'huggingface-cli login')"
        ) from exc
    except HfHubHTTPError as exc:
        raise BrainforgeError(f"could not read metadata for '{repo_id}': {exc}") from exc

    files = [str(getattr(sibling, "rfilename", "")) for sibling in info.siblings or []]
    config = _hub_json(repo_id, "config.json", required=True)
    tokenizer_config = _hub_json(repo_id, "tokenizer_config.json", required=False)
    chat_template_file = _hub_download(repo_id, "chat_template.jinja") is not None
    safetensors = any(name.endswith(".safetensors") for name in files)
    weights = safetensors or any(name.endswith(".bin") for name in files)
    architectures = config.get("architectures")
    architecture = (
        str(architectures[0]) if isinstance(architectures, list) and architectures else None
    )
    card_data = info.card_data
    return ModelInfo(
        id=repo_id,
        source="hub",
        architecture=architecture,
        model_type=_string(config.get("model_type")),
        params=_as_int(info.safetensors.total) if info.safetensors is not None else None,
        max_position_embeddings=_as_int(_config_value(config, "max_position_embeddings")),
        hidden_size=_as_int(_config_value(config, "hidden_size")),
        num_layers=_as_int(_config_value(config, "num_hidden_layers")),
        chat_template=tokenizer_config.get("chat_template") is not None or chat_template_file,
        license=_string(getattr(card_data, "license", None)) if card_data is not None else None,
        gated=bool(getattr(info, "gated", False)),
        private=bool(getattr(info, "private", False)),
        pre_quantized=config.get("quantization_config") is not None,
        remote_code=config.get("auto_map") is not None,
        safetensors=safetensors,
        pipeline_tag=_string(getattr(info, "pipeline_tag", None)),
        weights=weights,
        num_attention_heads=_as_int(_config_value(config, "num_attention_heads")),
        rope_scaling=_config_value(config, "rope_scaling") is not None,
    )


def inspect_model(model: str) -> ModelInfo:
    """Read local model metadata or Hub metadata without downloading weights."""
    path = Path(model).expanduser()
    if path.is_dir():
        return _inspect_local(path, str(model))
    if path.exists():
        raise BrainforgeError(f"model path is not a directory: {path}")
    return _inspect_hub(str(model))


def estimate_memory(
    model: ModelInfo,
    plan: TrainingPlan,
    max_length: int,
    batch_size: int,
    lora_rank: int = 16,
    available_memory: int | None = None,
) -> MemoryEstimate | None:
    """Estimate training memory from parameter counts; None when unknown."""
    if model.params is None:
        return None
    params = float(model.params)
    dtype_bytes = 4 if plan.dtype == "float32" else 2
    bytes_per_param = {
        "4bit": 0.55,
        "8bit": 1.05,
        "none": float(dtype_bytes),
    }[plan.quantization]
    weights_gb = params * bytes_per_param / 1e9

    adapters_gb = 0.0
    optimizer_gb = 0.0
    if plan.peft:
        if model.hidden_size and model.num_layers:
            adapters_gb = 2 * lora_rank * model.hidden_size * model.num_layers * 7 * 4 * 1.2 / 1e9
        optimizer_gb = adapters_gb * 12
    else:
        optimizer_gb = params * (8 + dtype_bytes + 4) / 1e9

    activations_gb = 0.0
    if model.hidden_size and model.num_layers:
        heads = model.num_attention_heads or max(1, model.hidden_size // 128)
        factor = 0.1 if plan.attn_implementation in ("sdpa", "flash_attention_2") else 1.0
        activations_gb = (
            batch_size * max_length * model.hidden_size * model.num_layers * dtype_bytes / 1e9
        )
        activations_gb += batch_size * heads * max_length * max_length * dtype_bytes * factor / 1e9

    overhead_gb = 0.1 * (weights_gb + adapters_gb + optimizer_gb + activations_gb) + 0.5
    total_gb = weights_gb + adapters_gb + optimizer_gb + activations_gb + overhead_gb
    available_gb = available_memory / 1e9 if available_memory is not None else None
    fits = total_gb <= available_gb if available_gb is not None else None
    return MemoryEstimate(
        weights_gb=weights_gb,
        adapters_gb=adapters_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        overhead_gb=overhead_gb,
        total_gb=total_gb,
        available_gb=available_gb,
        fits=fits,
    )


@dataclass(frozen=True)
class AutoFitResult:
    max_length: int
    batch_size: int
    gradient_accumulation: int
    adjustments: list[str]


def _user_set(config: TrainingConfig, field: str, default: int) -> bool:
    return bool(getattr(config, field) != default)


def _warmup_warning(config: TrainingConfig, total_steps: int | None) -> str | None:
    """Warn when warmup_steps would eat the whole run (LR stays at 0)."""
    if total_steps is not None and config.warmup_steps >= total_steps:
        return (
            f"warmup_steps ({config.warmup_steps}) >= total_steps ({total_steps}): "
            "learning rate will stay at 0, training is a no-op. "
            "Set warmup_steps=0 for short runs."
        )
    return None


def _device_budget(device: DeviceInfo, ram_available: int | None) -> int | None:
    """Training memory budget for a device.

    Accelerators get their free VRAM (or 90% of total when free is unknown);
    cpu/mps/tpu/npu train out of system RAM: available memory minus a safety
    margin, since the desktop and everything else keeps running alongside.
    """
    if device.kind in ("cuda", "rocm", "xpu"):
        if device.free_memory is not None:
            return device.free_memory
        if device.total_memory is not None:
            return int(device.total_memory * 0.9)
        return None
    if ram_available is None:
        return None
    margin = max(2 * 1024**3, int(ram_available * 0.15))
    return max(0, ram_available - margin)


def _system_ram() -> int | None:
    """Currently available system RAM, or None when psutil is missing."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def _too_large_error(
    model: ModelInfo, plan: TrainingPlan, device: DeviceInfo, floor_gb: float, budget: int
) -> BrainforgeError:
    if device.kind in ("cuda", "rocm", "xpu"):
        hint = "use a smaller student model, or train on CPU with --device cpu (slow)"
    else:
        hint = "use a smaller student model, or train on a machine with a GPU"
    return BrainforgeError(
        f"model '{model.id}' does not fit {device.kind}: training needs at least "
        f"~{floor_gb:.1f} GB (weights + adapters + optimizer, {plan.dtype}"
        f"{'/' + plan.quantization if plan.quantization != 'none' else ''}) but the "
        f"{device.kind} budget is {budget / 1e9:.1f} GB. {hint}. (estimate, not a guarantee)"
    )


def auto_fit_config(
    training: TrainingConfig,
    dataset_dir: Path | str,
    *,
    kd: bool = False,
) -> tuple[TrainingConfig, list[str]]:
    """Public wiring point: fit seq/batch to hardware, warn on warmup/steps.

    Returns (adjusted config, warning lines). Raises BrainforgeError when the
    model cannot fit the device budget. Metadata or hardware detection
    failures degrade to a skip warning instead of blocking training.
    """
    train_path = Path(dataset_dir) / "train.jsonl"
    rows = 0
    if train_path.exists():
        rows = sum(1 for _ in train_path.open(encoding="utf-8"))
    base_model = getattr(training, "base_model", "")
    try:
        devices = detect_devices()
    except BrainforgeError as exc:
        return training, [f"auto-fit skipped: {exc}"]
    if not devices:
        # merged semantics from PR 42: auto-fit must not block training it
        # cannot reason about
        return training, ["auto-fit skipped: no compute device detected"]
    if not base_model:
        return training, ["auto-fit skipped: no base_model in training config"]
    try:
        model = inspect_model(str(base_model))
    except BrainforgeError as exc:
        return training, [f"auto-fit skipped: {exc}"]
    result = _auto_fit(model, training, devices, kd=kd)
    config = training.model_copy(
        update={"max_length": result.max_length, "batch_size": result.batch_size}
    )
    warnings = list(result.adjustments)
    if rows:
        steps = math.ceil(rows / (config.batch_size * config.gradient_accumulation)) * config.epochs
        warmup = _warmup_warning(config, steps)
        if warmup:
            warnings.append(warmup)
    return config, warnings


def _auto_fit(
    model: ModelInfo,
    config: TrainingConfig,
    devices: list[DeviceInfo],
    *,
    system_ram: int | None = None,
    kd: bool = False,
) -> AutoFitResult:
    """Fit seq/batch to the device budget, or refuse with an explicit error.

    The plan is the one training will actually use (resolve_plan), not an
    idealized one: on CPU that means fp32 weights at 4 bytes/param, which is
    what previously OOM-killed whole desktop sessions. The weights floor
    (weights + adapters + optimizer) is refused even when the user set
    max_length/batch_size explicitly; user-touched values are otherwise
    respected and warned about instead of silently adjusted.
    """
    if not devices:
        raise BrainforgeError("no compute device detected")
    primary = devices[0]
    if system_ram is None:
        system_ram = _system_ram()
    budget = _device_budget(primary, system_ram)

    user_touched = _user_set(config, "max_length", TrainingConfig().max_length) or _user_set(
        config, "batch_size", TrainingConfig().batch_size
    )

    plan = resolve_plan(config, primary, kd=kd)
    adjustments: list[str] = []
    if budget is None:
        adjustments.append(
            f"auto-fit skipped: no memory budget known for {primary.kind}"
            " (install psutil for CPU/MPS fit checks)"
        )
        return AutoFitResult(
            max_length=config.max_length,
            batch_size=config.batch_size,
            gradient_accumulation=config.gradient_accumulation,
            adjustments=adjustments,
        )
    est = estimate_memory(
        model, plan, config.max_length, config.batch_size, config.lora_rank, budget
    )
    if est is None:
        adjustments.append("auto-fit skipped: model parameter count unknown")
        return AutoFitResult(
            max_length=config.max_length,
            batch_size=config.batch_size,
            gradient_accumulation=config.gradient_accumulation,
            adjustments=adjustments,
        )
    floor_gb = est.weights_gb + est.adapters_gb + est.optimizer_gb
    if floor_gb * 1e9 > budget:
        raise _too_large_error(model, plan, primary, floor_gb, budget)

    seq_options = [config.max_length] if user_touched else [4096, 2048, 1024]
    batch_options = [config.batch_size] if user_touched else [2, 1]
    for seq in seq_options:
        for batch in batch_options:
            fit = estimate_memory(model, plan, seq, batch, config.lora_rank, budget)
            if fit is None or fit.fits:
                if seq != config.max_length:
                    adjustments.append(
                        f"max_length adjusted {config.max_length} -> {seq} "
                        f"(budget {budget / 1e9:.1f} GB)"
                    )
                if batch != config.batch_size:
                    adjustments.append(f"batch_size adjusted {config.batch_size} -> {batch}")
                return AutoFitResult(
                    max_length=seq,
                    batch_size=batch,
                    gradient_accumulation=config.gradient_accumulation,
                    adjustments=adjustments,
                )
    if user_touched:
        adjustments.append(
            f"max_length={config.max_length} batch_size={config.batch_size} may not fit "
            f"({est.total_gb:.1f} GB estimated vs {budget / 1e9:.1f} GB budget);"
            " training proceeds as requested"
        )
        return AutoFitResult(
            max_length=config.max_length,
            batch_size=config.batch_size,
            gradient_accumulation=config.gradient_accumulation,
            adjustments=adjustments,
        )
    smallest = estimate_memory(model, plan, 1024, 1, config.lora_rank, budget)
    raise BrainforgeError(
        f"auto-fit impossible: model '{model.id}' too large for {primary.kind}: "
        f"best estimate {(smallest.total_gb if smallest is not None else est.total_gb):.1f} GB "
        f"exceeds the {budget / 1e9:.1f} GB budget even at seq=1024 batch=1"
        " (estimate, not a guarantee)"
    )


def _causal_lm_mapping() -> Any | None:
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

        return MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    except (ImportError, AttributeError):
        pass
    try:
        from transformers import AutoModelForCausalLM

        return getattr(AutoModelForCausalLM, "_model_mapping", None)
    except (ImportError, AttributeError):
        return None


def _plan_check_id(error: str) -> str:
    if "quantization" in error:
        return "plan.quantization"
    if "unsloth" in error.lower():
        return "plan.backend"
    if "flash_attention" in error:
        return "plan.attention"
    return "plan"


def _dependency_checks(plan: TrainingPlan | None, checks: list[Check]) -> None:
    required = ["transformers", "datasets"]
    if plan is not None:
        # kd trains through transformers.Trainer; trl is only needed by the trl and unsloth backends
        if plan.backend in ("trl", "unsloth"):
            required.append("trl")
        if plan.peft:
            required.append("peft")
        if plan.quantization != "none":
            required.append("bitsandbytes")
        if plan.backend == "unsloth":
            required.append("unsloth")
    for name in required:
        if _available(name):
            checks.append(Check(f"training.{name}", "pass", f"{name} is installed"))
        else:
            checks.append(Check(f"training.{name}", "fail", f"{name} is not installed"))


def _architecture_check(model: ModelInfo, checks: list[Check]) -> None:
    mapping = _causal_lm_mapping()
    if mapping is None:
        checks.append(
            Check(
                "model.architecture",
                "skip",
                "transformers causal-LM mapping unavailable; cannot verify architecture",
            )
        )
        return
    supported = bool(
        (model.model_type and model.model_type in mapping)
        or (model.architecture and model.architecture in mapping.values())
    )
    if supported:
        checks.append(
            Check(
                "model.architecture",
                "pass",
                f"{model.model_type or model.architecture} is supported",
            )
        )
    else:
        checks.append(
            Check(
                "model.architecture",
                "fail",
                f"{model.model_type or model.architecture} is not a known causal-LM architecture",
            )
        )


def _model_checks(
    model: ModelInfo,
    plan: TrainingPlan | None,
    hardware: DeviceInfo | None,
    *,
    max_length: int,
    batch_size: int,
    lora_rank: int,
    checks: list[Check],
) -> MemoryEstimate | None:
    _architecture_check(model, checks)
    checks.append(
        Check(
            "model.chat_template",
            "pass" if model.chat_template else "fail",
            "chat template present" if model.chat_template else "no chat template found",
        )
    )
    checks.append(
        Check(
            "model.pre_quantized",
            "fail" if model.pre_quantized else "pass",
            "pre-quantized checkpoint"
            if model.pre_quantized
            else "base checkpoint is not quantized",
        )
    )
    checks.append(
        Check(
            "model.remote_code",
            "fail" if model.remote_code else "pass",
            "remote code required" if model.remote_code else "no remote code required",
        )
    )
    checks.append(
        Check(
            "model.gated",
            "warn" if model.gated else "pass",
            "gated model; ensure HF_TOKEN has access" if model.gated else "not gated",
        )
    )
    checks.append(
        Check(
            "model.license",
            "pass" if model.license else "warn",
            f"license: {model.license}" if model.license else "license not declared",
        )
    )
    if not model.weights:
        checks.append(Check("model.weights", "fail", "no safetensors or .bin weights found"))
    elif not model.safetensors:
        checks.append(Check("model.weights", "warn", ".bin weights only; safetensors preferred"))
    else:
        checks.append(Check("model.weights", "pass", "safetensors weights found"))

    if model.max_position_embeddings is None:
        checks.append(Check("model.max_length", "skip", "max_position_embeddings not declared"))
    elif max_length > model.max_position_embeddings:
        checks.append(
            Check(
                "model.max_length",
                "fail",
                f"max_length {max_length} exceeds model limit {model.max_position_embeddings}",
            )
        )
    elif model.rope_scaling:
        checks.append(Check("model.max_length", "warn", "rope scaling present; verify the limit"))
    else:
        checks.append(Check("model.max_length", "pass", f"max_length {max_length} fits"))

    if model.pipeline_tag == "text-generation":
        checks.append(Check("model.pipeline_tag", "pass", "text-generation model"))
    elif model.pipeline_tag is None:
        checks.append(Check("model.pipeline_tag", "warn", "pipeline_tag not declared"))
    else:
        checks.append(
            Check(
                "model.pipeline_tag",
                "warn",
                f"pipeline_tag is {model.pipeline_tag}, not text-generation",
            )
        )

    if plan is None:
        checks.append(Check("memory.fit", "skip", "no training plan available"))
        return None
    available = None
    if hardware is not None:
        available = hardware.free_memory or hardware.total_memory
    estimate = estimate_memory(
        model, plan, max_length, batch_size, lora_rank, available_memory=available
    )
    if estimate is None:
        checks.append(Check("memory.fit", "skip", "parameter count unavailable"))
        return None
    if estimate.fits is False:
        checks.append(
            Check(
                "memory.fit",
                "fail",
                f"estimated {estimate.total_gb:.1f} GB exceeds {estimate.available_gb:.1f} GB",
            )
        )
    elif estimate.available_gb is not None and estimate.total_gb > 0.85 * estimate.available_gb:
        checks.append(
            Check(
                "memory.fit",
                "warn",
                f"estimated {estimate.total_gb:.1f} GB is above 85% of"
                f" {estimate.available_gb:.1f} GB",
            )
        )
    else:
        checks.append(
            Check(
                "memory.fit",
                "pass",
                f"estimated {estimate.total_gb:.1f} GB (estimate, not a guarantee)",
            )
        )
    return estimate


def _dataset_checks(dataset_dir: Path | None, checks: list[Check]) -> None:
    if dataset_dir is None:
        return
    train = dataset_dir / "train.jsonl"
    if not train.is_file() or train.stat().st_size == 0:
        checks.append(Check("dataset.train", "fail", f"missing or empty {train}"))
    else:
        checks.append(Check("dataset.train", "pass", f"found {train}"))
    validation = dataset_dir / "validation.jsonl"
    if not validation.is_file():
        checks.append(Check("dataset.validation", "warn", f"missing {validation}"))
    else:
        checks.append(Check("dataset.validation", "pass", f"found {validation}"))


def _disk_check(estimate: MemoryEstimate | None, checks: list[Check], target: Path) -> None:
    if estimate is None:
        checks.append(Check("disk.space", "skip", "no memory estimate available"))
        return
    if not target.is_dir():
        # The output dir is created by the training run itself; checking the
        # filesystem of a missing path would crash statvfs-based callers.
        checks.append(Check("disk.space", "skip", f"{target} does not exist yet"))
        return
    free_gb = shutil.disk_usage(target).free / 1e9
    if free_gb < estimate.weights_gb:
        checks.append(
            Check(
                "disk.space",
                "fail",
                f"{free_gb:.1f} GB free is below the {estimate.weights_gb:.1f} GB checkpoint",
            )
        )
    else:
        checks.append(Check("disk.space", "pass", f"{free_gb:.1f} GB free"))


def run_preflight(
    training: TrainingConfig,
    *,
    model: str | Path | None = None,
    strategy: str | None = None,
    quantization: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
    lora_rank: int | None = None,
    kd: bool = False,
    dataset_dir: Path | str | None = None,
    strict: bool = False,
    devices: list[DeviceInfo] | None = None,
) -> CheckReport:
    """Run every preflight check and return a report."""
    checks: list[Check] = []
    hardware: DeviceInfo | None = None
    plan: TrainingPlan | None = None
    detected = devices
    if detected is None:
        try:
            detected = detect_devices()
        except BrainforgeError as exc:
            checks.append(Check("training.torch", "fail", str(exc)))
            detected = []
    if detected:
        requested = device or str(getattr(training, "device", "auto"))
        try:
            hardware = resolve_device(requested, detected)
        except BrainforgeError as exc:
            checks.append(Check("hardware.device", "fail", str(exc)))
        else:
            checks.append(Check("hardware.device", "pass", f"{hardware.kind} ({hardware.name})"))
    else:
        checks.append(Check("hardware.device", "skip", "no device information available"))

    if hardware is not None:
        plan = resolve_plan(
            training,
            hardware,
            strategy=strategy,
            quantization=quantization,
            backend=backend,
            kd=kd,
        )
        errors = validate_plan(plan, hardware, training)
        if errors:
            for error in errors:
                checks.append(Check(_plan_check_id(error), "fail", error))
        else:
            checks.append(
                Check(
                    "plan",
                    "pass",
                    f"{plan.backend}/{plan.effective_strategy}/{plan.quantization}/{plan.dtype}",
                )
            )
        if plan.dtype == "bfloat16" and not hardware.bf16:
            checks.append(Check("hardware.bf16", "warn", "bf16 unsupported; using fp16"))

    _dependency_checks(plan, checks)

    model_id = str(model) if model is not None else training.base_model
    info: ModelInfo | None = None
    try:
        info = inspect_model(model_id)
    except BrainforgeError as exc:
        checks.append(Check("model.metadata", "fail", str(exc)))
    else:
        checks.append(Check("model.metadata", "pass", f"{info.source} model {info.id}"))

    resolved_max_length = max_length or training.max_length
    resolved_batch_size = batch_size or training.batch_size
    resolved_lora_rank = lora_rank or training.lora_rank
    estimate = None
    if info is not None:
        estimate = _model_checks(
            info,
            plan,
            hardware,
            max_length=resolved_max_length,
            batch_size=resolved_batch_size,
            lora_rank=resolved_lora_rank,
            checks=checks,
        )
    _dataset_checks(Path(dataset_dir) if dataset_dir is not None else None, checks)
    _disk_check(estimate, checks, Path(training.output_dir))

    has_fail = any(check.status == "fail" for check in checks)
    has_warn = any(check.status == "warn" for check in checks)
    ok = not has_fail and not (strict and has_warn)
    return CheckReport(
        ok=ok,
        strict=strict,
        model=info,
        hardware=hardware,
        plan=plan,
        estimate=estimate,
        checks=checks,
    )


def report_to_dict(report: CheckReport) -> dict:
    """Return a stable JSON-serializable report."""
    return asdict(report)
