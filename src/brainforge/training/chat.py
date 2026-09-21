"""Interactive chat with a trained (adapter or merged) student model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from brainforge.training.models import load_student_model

_EXIT_COMMANDS = {"quit", "exit"}


def generate_reply(
    model: Any, tokenizer: Any, history: list[str], max_new_tokens: int = 512
) -> str:
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
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def chat_loop(
    model_path: Path | str,
    reply_fn: Callable[[list[str]], str] | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    max_new_tokens: int = 512,
) -> None:
    """Interactive REPL; exits on quit/exit/EOF, ignores blank lines."""
    if reply_fn is None:
        model, tokenizer = load_student_model(model_path)
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
