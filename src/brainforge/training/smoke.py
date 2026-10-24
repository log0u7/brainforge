import os
import sys
from typing import Any


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed; skipping training smoke test")
        return 0
    from brainforge.errors import BrainforgeError
    from brainforge.training.hardware import default_dtype, detect_devices, resolve_device

    try:
        device = resolve_device("auto", detect_devices())
    except BrainforgeError as exc:
        print(f"skipping training smoke test: {exc}")
        return 0
    if device.kind == "cpu":
        print("no accelerator available; skipping training smoke test")
        return 0
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.environ.get("BRAINFORCE_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    print(f"smoke test: 1 QLoRA step on {model_name} ({device.kind})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, default_dtype(device)),
        load_in_4bit=device.kind in ("cuda", "rocm"),
        device_map="auto",
    )
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    inputs = tokenizer("subprocess.run(user_input, shell=True)", return_tensors="pt").to(
        model.device
    )
    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()
    print(f"smoke test passed: loss={outputs.loss.item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
