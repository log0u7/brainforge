"""Abliteration: remove the refusal direction from model weights.

Computes a single refusal direction as the mean hidden-state difference
between harmful and benign calibration prompts at a fixed-depth layer, then
projects every attention output and MLP down projection orthogonal to it
(single-direction ablation, Arditi et al. 2024 style).

Experimental: quality is a heuristic, not a guarantee.

Heavy dependencies (torch, transformers) are imported lazily so the package
and the unit test suite run without the ``training`` extra.
"""

import json
from pathlib import Path
from typing import Any

from brainforge.dataset.writer import read_jsonl
from brainforge.errors import BrainforgeError

# ponytail: fixed 2/3-depth target layer + mean-over-tokens direction.
# Upgrade path: sweep the target layer per model, multi-direction or
# Heretic-style iterative optimization.
_TARGET_FRACTION = 2 / 3
_EDIT_PATHS = ("self_attn.o_proj", "mlp.down_proj")
_VALID_LABELS = ("harmful", "benign")


def load_calibration(path: Path | str) -> list[dict]:
    """Load and validate a calibration JSONL of {prompt, label} records."""
    path = Path(path)
    records = []
    for line_number, line in enumerate(read_jsonl(path), start=1):
        prompt = line.get("prompt")
        label = line.get("label")
        if not prompt:
            raise BrainforgeError(f"{path}:{line_number}: missing 'prompt'")
        if label not in _VALID_LABELS:
            raise BrainforgeError(
                f"{path}:{line_number}: 'label' must be one of {list(_VALID_LABELS)}, got '{label}'"
            )
        records.append({"prompt": prompt, "label": label})
    if not records:
        raise BrainforgeError(f"empty calibration file: {path}")
    return records


def select_layers(spec: str, n_layers: int) -> list[int]:
    """Parse a layer selection: ``all`` or a comma-separated index list."""
    if spec == "all":
        return list(range(n_layers))
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise BrainforgeError(f"invalid layer index '{part}'") from exc
        if index < 0 or index >= n_layers:
            raise BrainforgeError(f"layer index {index} out of range (model has {n_layers} layers)")
        indices.append(index)
    if not indices:
        raise BrainforgeError(f"no layers selected from '{spec}'")
    return indices


def _resolve_module(layer: Any, path: str) -> Any | None:
    module = layer
    for part in path.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def abliterate(
    model_path: Path | str,
    calibration_path: Path | str,
    output_dir: Path | str,
    layers: str = "all",
) -> dict:
    """Abliterate the refusal direction and save the modified model.

    Returns ablation stats (also written to ``ablation_stats.json``).
    """
    model_path = str(model_path)
    calibration = load_calibration(calibration_path)
    harmful = [record["prompt"] for record in calibration if record["label"] == "harmful"]
    benign = [record["prompt"] for record in calibration if record["label"] == "benign"]
    if not harmful or not benign:
        raise BrainforgeError("calibration needs at least one harmful and one benign prompt")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise BrainforgeError(
            "torch and transformers are not installed; install the training extra first:"
            " uv sync --extra training"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    decoder_layers = getattr(getattr(model, "model", None), "layers", None)
    if decoder_layers is None:
        raise BrainforgeError(
            f"model at {model_path} exposes no 'model.layers'; abliteration supports"
            " Llama/Qwen-style architectures"
        )
    n_layers = len(decoder_layers)
    target_layer = min(int(n_layers * _TARGET_FRACTION), n_layers - 1)

    def mean_hidden(prompts: list[str]) -> Any:
        captured: list[Any] = []
        sums: Any = None
        count = 0

        def hook(_module: Any, _args: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach().to(torch.float32).mean(dim=1))

        handle = decoder_layers[target_layer].register_forward_hook(hook)
        try:
            for prompt in prompts:
                captured.clear()
                batch = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**batch)
                sums = captured[0] if sums is None else sums + captured[0]
                count += 1
        finally:
            handle.remove()
        return (sums / count).squeeze(0)

    harm_mean = mean_hidden(harmful)
    benign_mean = mean_hidden(benign)
    direction = harm_mean - benign_mean
    before_norm = float(direction.norm())
    if before_norm == 0.0:
        raise BrainforgeError(
            "refusal direction is degenerate (zero norm); check the calibration data"
        )
    r_hat = direction / direction.norm()

    selected = select_layers(layers, n_layers)
    edited: list[int] = []
    skipped: list[str] = []
    for index in selected:
        layer = decoder_layers[index]
        for edit_path in _EDIT_PATHS:
            module = _resolve_module(layer, edit_path)
            if module is None:
                skipped.append(f"layer {index}: {edit_path}")
                continue
            weight = module.weight.data
            projection = weight @ r_hat.to(weight.dtype)
            module.weight.data.copy_(weight - torch.outer(projection, r_hat.to(weight.dtype)))
        edited.append(index)

    after_norm = float((mean_hidden(harmful) - mean_hidden(benign)).norm())
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    stats = {
        "model": model_path,
        "calibration": str(calibration_path),
        "target_layer": target_layer,
        "layers_edited": edited,
        "layers_skipped": skipped,
        "direction_norm_before": before_norm,
        "direction_norm_after": after_norm,
        "suppression_ratio": after_norm / before_norm,
        "output_dir": str(output_dir),
    }
    (output_dir / "ablation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
