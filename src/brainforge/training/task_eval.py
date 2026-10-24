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
    """Extract {vulnerability_found, abstain, cwe} from the record's judge answer."""
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
        "abstain": data.get("verdict") == "insufficient_information",
        "cwe": data.get("cwe"),
    }


def prediction_from_text(text: str) -> dict:
    """Map a raw model answer to {vulnerability_found, abstain, cwe}; None = unparseable miss."""
    try:
        data = extract_json(text)
    except ProviderError:
        return {"vulnerability_found": None, "abstain": False, "cwe": None}
    return {
        "vulnerability_found": data.get("verdict") == "confirmed",
        "abstain": data.get("verdict") == "insufficient_information",
        "cwe": data.get("cwe"),
    }


def _answer_class(side: dict) -> str:
    """One of confirmed / rejected / abstain / unparseable."""
    if side.get("abstain"):
        return "abstain"
    found = side.get("vulnerability_found")
    if found is True:
        return "confirmed"
    if found is None:
        return "unparseable"
    return "rejected"


def score_task(pairs: list[tuple[dict, dict]]) -> dict:
    """Compute accuracy, FP/FN rates, CWE accuracy and abstention metrics.

    Abstention is a third outcome class (``insufficient_information``): a model
    that abstains where the judge abstained is correct, an abstention on a
    confirmed case is a false negative, and a guess on an abstained case is a
    false positive. ``abstain_recall`` is the graceful-failure metric: of the
    records the judge could not resolve, how many did the model refuse too.
    """
    if not pairs:
        raise BrainforgeError("no evaluation pairs")
    n = len(pairs)
    correct = 0
    false_positives = 0
    false_negatives = 0
    cwe_correct = 0
    cwe_compared = 0
    abstain_expected = 0
    abstain_predicted = 0
    abstain_correct = 0
    for prediction, expected in pairs:
        predicted = _answer_class(prediction)
        actual = _answer_class(expected)
        if actual == "abstain":
            abstain_expected += 1
        if predicted == "abstain":
            abstain_predicted += 1
        if predicted == actual:
            correct += 1
            if actual == "abstain":
                abstain_correct += 1
        elif actual == "confirmed":
            # a real finding the model did not confirm: wrong guess, abstention
            # or unparseable output all count as a miss
            false_negatives += 1
        else:
            # claimed a finding the judge did not confirm, including a guess
            # on a case the judge abstained on
            false_positives += 1
        expected_cwe = expected.get("cwe")
        if expected.get("vulnerability_found") and expected_cwe:
            cwe_compared += 1
            if prediction.get("cwe") == expected_cwe:
                cwe_correct += 1
    return {
        "n_records": n,
        "accuracy": correct / n,
        "false_positive_rate": false_positives / n,
        "false_negative_rate": false_negatives / n,
        "cwe_accuracy": (cwe_correct / cwe_compared) if cwe_compared else None,
        "abstain_rate": abstain_predicted / n,
        "abstain_recall": (abstain_correct / abstain_expected) if abstain_expected else None,
    }


def run_task_eval(
    records: list[dict], generate: Callable[[list], list[str]], max_new_tokens: int = 512
) -> dict:
    """Score generated answers against expected ones.

    ``generate`` receives the list of per-record user messages (one item per
    record, in order) and must return one raw model text per record, in the
    same order. Batched generators amortize ``model.generate`` across records.
    """
    if not records:
        raise BrainforgeError("empty evaluation dataset")
    domains = {str(record["domain"]) for record in records if record.get("domain") is not None}
    unexpected = domains - {"security"}
    if unexpected:
        raise BrainforgeError(
            f"task evaluation only supports the security domain, found: {sorted(unexpected)}"
        )
    prompts = [
        [m["content"] for m in record.get("messages", []) if m["role"] == "user"]
        for record in records
    ]
    texts = generate(prompts)
    pairs = [
        (prediction_from_text(text), expected_from_record(record))
        for record, text in zip(records, texts, strict=True)
    ]
    result = score_task(pairs)
    result["domains"] = sorted(domains)
    return result


def evaluate_model_on_records(
    model_path: Path | str,
    dataset_path: Path | str,
    quantization: str = "auto",
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> dict:
    """Load the trained model and run task evaluation on a JSONL dataset split."""
    from brainforge.dataset.writer import read_jsonl

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    records = read_jsonl(dataset_path)
    generate = _build_generator(model_path, quantization, max_new_tokens, batch_size)
    result = run_task_eval(records, generate)
    (model_path / "task_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _flatten_message(user_message: str | list[str]) -> str:
    """Normalize a user message (possibly a chat-style list) to a single string."""
    if isinstance(user_message, list):
        return "\n\n".join(user_message)
    return user_message


def _build_generator(
    model_path: Path,
    quantization: str,
    max_new_tokens: int,
    batch_size: int = 4,
    attn_implementation: str = "sdpa",
) -> Callable[[list], list[str]]:
    import torch

    from brainforge.training.hardware import plan_for_inference
    from brainforge.training.models import load_student_model

    model, tokenizer = load_student_model(
        model_path, plan_for_inference(quantization, attn_implementation)
    )
    # Left padding keeps every prompt in the batch aligned at the same position.
    tokenizer.padding_side = "left"

    def generate(user_messages: list) -> list[str]:
        prompts = [_flatten_message(item) for item in user_messages]
        replies: list[str] = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt} for prompt in batch],
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(model.device)
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated = output[:, inputs["input_ids"].shape[1] :]
                replies.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return replies

    return generate
