"""Task-level evaluation of a fine-tuned student model on a held-out split.

Unlike ``qlora.evaluate`` (loss + perplexity), this harness scores the model's
actual task output (verdict + CWE classification) against the judge-verified
ground truth stored in each dataset record's assistant message.
"""

import json
from collections.abc import Callable
from pathlib import Path

from brainforge.errors import BrainforgeError, ProviderError
from brainforge.providers.base import extract_json


def expected_from_record(record: dict) -> dict:
    """Extract {vulnerability_found, cwe} from the record's judge answer."""
    messages = record.get("messages") or []
    if not messages:
        raise BrainforgeError(f"record '{record.get('id')}' has no messages")
    content = messages[-1].get("content", "")
    try:
        data = extract_json(content)
    except Exception as exc:
        raise BrainforgeError(
            f"record '{record.get('id')}' has unparseable assistant JSON: {exc}"
        ) from exc
    return {
        "vulnerability_found": data.get("verdict") == "confirmed",
        "cwe": data.get("cwe"),
    }


def prediction_from_text(text: str) -> dict:
    """Map a raw model answer to {vulnerability_found, cwe}; None = unparseable miss."""
    try:
        data = extract_json(text)
    except ProviderError:
        return {"vulnerability_found": None, "cwe": None}
    return {
        "vulnerability_found": data.get("verdict") == "confirmed",
        "cwe": data.get("cwe"),
    }


def score_task(pairs: list[tuple[dict, dict]]) -> dict:
    """Compute accuracy, FP/FN rates and CWE accuracy from (prediction, expected) pairs."""
    if not pairs:
        raise BrainforgeError("no evaluation pairs")
    n = len(pairs)
    correct = 0
    false_positives = 0
    false_negatives = 0
    cwe_correct = 0
    cwe_compared = 0
    for prediction, expected in pairs:
        predicted = prediction["vulnerability_found"]
        actual = expected["vulnerability_found"]
        if predicted is not None and predicted == actual:
            correct += 1
        elif (predicted is True and actual is False) or (predicted is None and actual is False):
            false_positives += 1
        elif (predicted is False and actual is True) or (predicted is None and actual is True):
            false_negatives += 1
        expected_cwe = expected.get("cwe")
        if actual and expected_cwe:
            cwe_compared += 1
            if prediction.get("cwe") == expected_cwe:
                cwe_correct += 1
    return {
        "n_records": n,
        "accuracy": correct / n,
        "false_positive_rate": false_positives / n,
        "false_negative_rate": false_negatives / n,
        "cwe_accuracy": (cwe_correct / cwe_compared) if cwe_compared else None,
    }


def run_task_eval(
    records: list[dict], generate: Callable[[str | list[str]], str], max_new_tokens: int = 512
) -> dict:
    """Score generated answers against expected ones.

    ``generate`` receives the user message content and returns raw model text;
    pass ``None`` only from the CLI path where the real model loader builds it.
    """
    if not records:
        raise BrainforgeError("empty evaluation dataset")
    domains = {str(record["domain"]) for record in records if record.get("domain") is not None}
    unexpected = domains - {"security"}
    if unexpected:
        raise BrainforgeError(
            f"task evaluation only supports the security domain, found: {sorted(unexpected)}"
        )
    pairs = []
    for record in records:
        user_messages = [m["content"] for m in record.get("messages", []) if m["role"] == "user"]
        text = generate(user_messages)
        pairs.append((prediction_from_text(text), expected_from_record(record)))
    result = score_task(pairs)
    result["domains"] = sorted(domains)
    return result


def evaluate_model_on_records(
    model_path: Path | str,
    dataset_path: Path | str,
    quantization: str = "4bit",
    max_new_tokens: int = 512,
) -> dict:
    """Load the trained model and run task evaluation on a JSONL dataset split."""
    from brainforge.dataset.writer import read_jsonl
    from brainforge.training.qlora import _require_cuda

    _require_cuda()
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    records = read_jsonl(dataset_path)
    generate = _build_generator(model_path, quantization, max_new_tokens)
    result = run_task_eval(records, generate)
    (model_path / "task_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _flatten_message(user_message: str | list[str]) -> str:
    """Normalize a user message (possibly a chat-style list) to a single string."""
    if isinstance(user_message, list):
        return "\n\n".join(user_message)
    return user_message


def _build_generator(
    model_path: Path, quantization: str, max_new_tokens: int
) -> Callable[[str | list[str]], str]:
    import torch

    from brainforge.training.models import load_student_model

    model, tokenizer = load_student_model(model_path, quantization)

    def generate(user_message: str | list[str]) -> str:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": _flatten_message(user_message)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]
        return str(tokenizer.decode(generated, skip_special_tokens=True))

    return generate
