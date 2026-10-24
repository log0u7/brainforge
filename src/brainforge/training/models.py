"""Shared student-model loading for chat and task evaluation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from brainforge.errors import BrainforgeError

if TYPE_CHECKING:
    from brainforge.training.hardware import TrainingPlan


def load_student_model(
    model_path: Path | str,
    plan: "TrainingPlan | None" = None,
) -> tuple[Any, Any]:
    """Load a merged model or LoRA adapter directory; returns (model, tokenizer).

    ``plan`` comes from :func:`brainforge.training.hardware.plan_for_inference`;
    it is resolved automatically when omitted.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise BrainforgeError(f"model directory not found: {model_path}")
    from brainforge.training.hardware import plan_for_inference, torch_dtype
    from brainforge.training.qlora import _quantization_config

    if plan is None:
        plan = plan_for_inference()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.chat_template is None:
        raise BrainforgeError(f"tokenizer at {model_path} has no chat template")
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch_dtype(plan),
        "attn_implementation": plan.attn_implementation,
    }
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)
    if (model_path / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    return model, tokenizer
