"""Task-level evaluation of a fine-tuned student model on a held-out split.

Unlike ``qlora.evaluate`` (loss + perplexity), this harness scores the model's
actual task output (verdict + CWE classification) against the judge-verified
ground truth stored in each dataset record's assistant message.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brainforge.errors import BrainforgeError, ProviderError
from brainforge.providers.base import extract_json


def expected_from_record(record: dict) -> dict:
    """Extract the judge-verified ground truth from the record's assistant answer.

    Security records yield {vulnerability_found, abstain, cwe}; CTF records
    yield {verdict, abstain, flag} with the judge verdict kept verbatim
    (solved / partial / insufficient_information).
    """
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
    if record.get("domain") == "ctf":
        return {
            "verdict": data.get("verdict"),
            "abstain": data.get("verdict") == "insufficient_information",
            "flag": data.get("flag"),
        }
    return {
        "vulnerability_found": data.get("verdict") == "confirmed",
        "abstain": data.get("verdict") == "insufficient_information",
        "cwe": data.get("cwe"),
    }


def prediction_from_text(text: str, domain: str = "security") -> dict:
    """Map a raw model answer to the domain's prediction shape; None = unparseable miss."""
    try:
        data = extract_json(text)
    except ProviderError:
        if domain == "ctf":
            return {"verdict": None, "abstain": False, "flag": None}
        return {"vulnerability_found": None, "abstain": False, "cwe": None}
    if domain == "ctf":
        return {
            "verdict": data.get("verdict"),
            "abstain": data.get("verdict") == "insufficient_information",
            "flag": data.get("flag"),
        }
    return {
        "vulnerability_found": data.get("verdict") == "confirmed",
        "abstain": data.get("verdict") == "insufficient_information",
        "cwe": data.get("cwe"),
    }


def _answer_class(side: dict, domain: str = "security") -> str:
    """Security: confirmed / rejected / abstain / unparseable.

    CTF keeps the verdict verbatim (solved / partial / abstain /
    unparseable) so a partial solve does not pass as a full one.
    """
    if side.get("abstain"):
        return "abstain"
    if domain == "ctf":
        verdict = side.get("verdict")
        return "unparseable" if verdict is None else str(verdict)
    found = side.get("vulnerability_found")
    if found is True:
        return "confirmed"
    if found is None:
        return "unparseable"
    return "rejected"


def score_task(pairs: list[tuple[dict, dict]], domain: str = "security") -> dict:
    """Compute accuracy, FP/FN rates, CWE/flag accuracy and abstention metrics.

    Abstention is a third outcome class (``insufficient_information``): a model
    that abstains where the judge abstained is correct, an abstention on a
    confirmed case is a false negative, and a guess on an abstained case is a
    false positive. ``abstain_recall`` is the graceful-failure metric: of the
    records the judge could not resolve, how many did the model refuse too.

    For the CTF domain the verdict classes stay verbatim (solved / partial /
    abstain / unparseable) and ``flag_accuracy`` compares predicted flags
    wherever both sides carry one; the judge flag is the gate-checked ground
    truth on verified-tier records.
    """
    if not pairs:
        raise BrainforgeError("no evaluation pairs")
    n = len(pairs)
    correct = 0
    false_positives = 0
    false_negatives = 0
    cwe_correct = 0
    cwe_compared = 0
    flag_correct = 0
    flag_compared = 0
    abstain_expected = 0
    abstain_predicted = 0
    abstain_correct = 0
    for prediction, expected in pairs:
        predicted = _answer_class(prediction, domain)
        actual = _answer_class(expected, domain)
        if actual == "abstain":
            abstain_expected += 1
        if predicted == "abstain":
            abstain_predicted += 1
        if predicted == actual:
            correct += 1
            if actual == "abstain":
                abstain_correct += 1
        elif actual == "confirmed" or (domain == "ctf" and actual == "solved"):
            # a real solve the model did not reproduce: wrong guess, abstention
            # or unparseable output all count as a miss
            false_negatives += 1
        else:
            # claimed more than the judge verified, including a guess on a
            # case the judge abstained on
            false_positives += 1
        expected_cwe = expected.get("cwe")
        if domain != "ctf" and expected.get("vulnerability_found") and expected_cwe:
            cwe_compared += 1
            if prediction.get("cwe") == expected_cwe:
                cwe_correct += 1
        if domain == "ctf":
            expected_flag = expected.get("flag")
            predicted_flag = prediction.get("flag")
            if expected_flag is not None and predicted_flag is not None:
                flag_compared += 1
                if predicted_flag == expected_flag:
                    flag_correct += 1
    metrics = {
        "n_records": n,
        "accuracy": correct / n,
        "false_positive_rate": false_positives / n,
        "false_negative_rate": false_negatives / n,
        "cwe_accuracy": (cwe_correct / cwe_compared) if cwe_compared else None,
        "flag_accuracy": (flag_correct / flag_compared) if flag_compared else None,
        "abstain_rate": abstain_predicted / n,
        "abstain_recall": (abstain_correct / abstain_expected) if abstain_expected else None,
    }
    return metrics


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
    unexpected = domains - {"security", "ctf"}
    if unexpected:
        raise BrainforgeError(
            "task evaluation only supports the security and ctf domains, "
            f"found: {sorted(unexpected)}"
        )
    if len(domains) > 1:
        raise BrainforgeError(
            f"mixed domains not supported: run one domain per evaluation, found: {sorted(domains)}"
        )
    if "ctf" in domains:
        domainless = [record for record in records if not record.get("domain")]
        if domainless:
            raise BrainforgeError(
                f"{len(domainless)} record(s) without a domain in a ctf evaluation:"
                " every record must declare its domain or the CTF scoring would"
                " silently mis-score it"
            )
    domain = domains.pop() if domains else "security"
    prompts = [
        [m["content"] for m in record.get("messages", []) if m["role"] == "user"]
        for record in records
    ]
    texts = generate(prompts)
    pairs = [
        (prediction_from_text(text, domain=domain), expected_from_record(record))
        for record, text in zip(records, texts, strict=True)
    ]
    result = score_task(pairs, domain=domain)
    result["domains"] = [domain]
    return result


def evaluate_model_on_records(
    model_path: Path | str,
    dataset_path: Path | str,
    quantization: str = "auto",
    max_new_tokens: int = 512,
    batch_size: int = 4,
    generator: Callable[[list], list[str]] | None = None,
) -> dict:
    """Load the trained model (or accept a served-model generator) and run task eval.

    ``generator`` overrides local model loading: pass
    :func:`build_remote_generator` to score a model served behind an
    OpenAI-compatible endpoint. The report is only persisted next to locally
    stored weights; served models have no local directory to write into.
    """
    from brainforge.dataset.writer import read_jsonl

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    records = read_jsonl(dataset_path)
    if generator is None:
        generator = _build_generator(model_path, quantization, max_new_tokens, batch_size)
    result = run_task_eval(records, generator)
    # Hub ids (base-model evaluation) resolve outside the local filesystem:
    # the report is only persisted next to locally stored weights.
    if model_path.is_dir():
        (model_path / "task_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _flatten_message(user_message: str | list[str]) -> str:
    """Normalize a user message (possibly a chat-style list) to a single string."""
    if isinstance(user_message, list):
        return "\n\n".join(user_message)
    return user_message


def _build_provider(base_url: str, api_key: str | None) -> Any:
    """Build an OpenAI-compatible provider for a served model endpoint."""
    from brainforge.config.models import ProviderConfig
    from brainforge.providers.registry import build_provider
    from brainforge.types import ProviderType

    return build_provider(
        "task-eval-remote",
        ProviderConfig(
            type=ProviderType.OPENAI,
            base_url=base_url,
            api_key=api_key or "not-needed",
        ),
    )


def build_remote_generator(
    base_url: str,
    model: str,
    max_tokens: int = 512,
    api_key: str | None = None,
    temperature: float = 0.7,
    concurrency: int = 4,
) -> Callable[[list], list[str]]:
    """Generate task answers against a served model (OpenAI-compatible endpoint).

    One chat completion per record prompt; the served model may batch internally.
    Prompts are flattened through ``_flatten_message`` like the local generator,
    and responses are taken from the assistant ``content`` (a thinking trace in
    ``reasoning_content`` is the model's, not the answer).
    """
    from concurrent.futures import ThreadPoolExecutor

    from brainforge.providers.base import ChatMessage, ChatRequest

    provider = _build_provider(base_url, api_key)

    def one(prompt: str) -> str:
        response = provider.complete(
            ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            model,
        )
        content = response.content or ""
        if not content.strip():
            raise ProviderError(
                f"model '{model}' returned an empty completion for a task-eval prompt"
            )
        return content

    def generate(user_messages: list) -> list[str]:
        prompts = [_flatten_message(item) for item in user_messages]
        if len(prompts) <= 1:
            return [one(prompt) for prompt in prompts]
        with ThreadPoolExecutor(max_workers=min(concurrency, len(prompts))) as pool:
            return list(pool.map(one, prompts))

    return generate


def _conversations(batch: list[str]) -> list[list[dict]]:
    """One single-message conversation per prompt.

    A flat list of messages would merge the whole batch into ONE sequence
    (the model then answers all prompts at once): found by the first real
    eval run, where a 4-prompt batch produced a single 1965-token input.
    """
    return [[{"role": "user", "content": prompt}] for prompt in batch]


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
                    _conversations(batch),
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
