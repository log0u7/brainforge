"""CPU-only smoke training: 1 LoRA step on a tiny model, CI-safe without GPU/HF."""

import sys
from typing import Any


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not installed; skipping CPU training smoke test")
        return 0
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        print(f"smoke test (CPU): 1 LoRA step on {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        inputs = tokenizer("subprocess.run(user_input, shell=True)", return_tensors="pt").to(
            model.device
        )
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        print(f"smoke test passed (CPU, fp32): loss={outputs.loss.item():.4f}")
        return 0
    except Exception as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
