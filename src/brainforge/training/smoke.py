import os
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed; skipping training smoke test")
        return 0
    if not torch.cuda.is_available():
        print("no CUDA device available; skipping training smoke test")
        return 0
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.environ.get("BRAINFORCE_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    print(f"smoke test: 1 QLoRA step on {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
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
