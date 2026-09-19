"""Interactive chat with a trained (adapter or merged) student model."""

from pathlib import Path

from brainforge.errors import BrainforgeError

_EXIT_COMMANDS = {"quit", "exit"}


def load_chat_model(model_path):
    """Load a merged model or a LoRA adapter directory for interactive chat."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise BrainforgeError(f"model directory not found: {model_path}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.chat_template is None:
        raise BrainforgeError(f"tokenizer at {model_path} has no chat template")
    if (model_path / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_path), device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), device_map="auto", torch_dtype=torch.bfloat16
        )
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, history: list[str], max_new_tokens: int = 512) -> str:
    """Generate one greedy assistant reply for the flat user-turn history."""
    import torch

    messages = [{"role": "user", "content": turn} for turn in history]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def chat_loop(
    model_path,
    reply_fn=None,
    input_fn=input,
    print_fn=print,
    max_new_tokens: int = 512,
) -> None:
    """Interactive REPL; exits on quit/exit/EOF, ignores blank lines."""
    if reply_fn is None:
        model, tokenizer = load_chat_model(model_path)
        reply_fn = lambda history: generate_reply(  # noqa: E731
            model, tokenizer, history, max_new_tokens
        )
    history: list[str] = []
    print_fn("brainforge chat - type 'quit' or 'exit' to leave")
    while True:
        try:
            user_input = input_fn("you> ")
        except EOFError:
            print_fn("")
            return
        text = user_input.strip()
        if not text:
            continue
        if text.lower() in _EXIT_COMMANDS:
            return
        history.append(text)
        print_fn(f"assistant> {reply_fn(history)}")
