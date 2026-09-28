"""Shared student-model loading for chat and task evaluation."""

from pathlib import Path
from typing import Any

from brainforge.errors import BrainforgeError


def load_student_model(
    model_path: Path | str,
    quantization: str | None = None,
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any]:
    """Load a merged model or LoRA adapter directory; returns (model, tokenizer)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise BrainforgeError(f"model directory not found: {model_path}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.chat_template is None:
        raise BrainforgeError(f"tokenizer at {model_path} has no chat template")
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
    }
    if quantization is not None:
        from brainforge.training.qlora import _quantization_config

        quant_config = _quantization_config(quantization)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
    if (model_path / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    return model, tokenizer
