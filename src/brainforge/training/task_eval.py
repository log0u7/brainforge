"""Task-level evaluation of a fine-tuned student model on a held-out split.

Unlike ``qlora.evaluate`` (loss + perplexity), this harness scores the model's
actual task output (verdict + CWE classification) against the judge-verified
ground truth stored in each dataset record's assistant message. The domain's
scoring semantics live in its pack (``DomainPack.eval_score`` hook, ADR-0022):
a domain is evaluable when its pack implements the hook.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brainforge.domains.base import DomainPack
from brainforge.domains.registry import get_or_generic
from brainforge.errors import BrainforgeError, ProviderError
from brainforge.providers.base import extract_json

_EVAL_SCORE_ERROR = (
    "domain pack '{domain}' does not implement eval_score; "
    "task evaluation is not supported for this domain"
)


def _eval_pack(domain: str) -> DomainPack:
    """Resolve the domain pack for a task evaluation, rejecting domains
    whose pack does not implement the eval_score hook (ADR-0022)."""
    pack = get_or_generic(domain)
    if type(pack).eval_score is DomainPack.eval_score:
        raise BrainforgeError(_EVAL_SCORE_ERROR.format(domain=domain))
    return pack


def expected_from_record(record: dict) -> dict:
    """Extract the judge-verified ground truth from the record's assistant
    answer, mapped to the domain pack's eval shape.

    Security records yield {vulnerability_found, abstain, cwe}; other
    domains yield whatever their pack's eval_shape maps the verdict to.
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
    pack = get_or_generic(str(record.get("domain") or "security"))
    return pack.eval_shape(data)


def prediction_from_text(text: str, domain: str = "security") -> dict:
    """Map a raw model answer to the domain pack's eval shape; None = unparseable miss."""
    pack = get_or_generic(domain)
    try:
        data = extract_json(text)
    except ProviderError:
        return pack.eval_unparseable()
    return pack.eval_shape(data)


def score_task(pairs: list[tuple[dict, dict]], domain: str = "security") -> dict:
    """Compute accuracy, FP/FN rates, the pack's secondary-metric accuracy
    (see the pack's ``eval_secondary_metrics`` hook) and abstention
    metrics.

    Abstention is a third outcome class (``insufficient_information``): a model
    that abstains where the judge abstained is correct, an abstention on a
    confirmed case is a false negative, and a guess on an abstained case is a
    false positive. ``abstain_recall`` is the graceful-failure metric: of the
    records the judge could not resolve, how many did the model refuse too.

    The per-case verdict comes from the domain pack's ``eval_score`` hook
    (ADR-0022); a domain whose pack does not implement the hook is rejected
    with an explicit error.
    """
    if not pairs:
        raise BrainforgeError("no evaluation pairs")
    pack = _eval_pack(domain)
    n = len(pairs)
    correct = 0
    false_positives = 0
    false_negatives = 0
    abstain_expected = 0
    abstain_predicted = 0
    abstain_correct = 0
    for prediction, expected in pairs:
        view = pack.eval_score(prediction, expected)
        if view is None:
            raise BrainforgeError(_EVAL_SCORE_ERROR.format(domain=domain))
        predicted = view["predicted"]
        actual = view["actual"]
        if actual == "abstain":
            abstain_expected += 1
        if predicted == "abstain":
            abstain_predicted += 1
        if view["outcome"] == "correct":
            correct += 1
            if actual == "abstain":
                abstain_correct += 1
        elif view["outcome"] == "false_negative":
            false_negatives += 1
        else:
            false_positives += 1
    metrics = {
        "n_records": n,
        "accuracy": correct / n,
        "false_positive_rate": false_positives / n,
        "false_negative_rate": false_negatives / n,
        "abstain_rate": abstain_predicted / n,
        "abstain_recall": (abstain_correct / abstain_expected)
        if abstain_expected
        else None,
    }
    metrics.update(pack.eval_secondary_metrics(pairs))
    return metrics


def _budget_warning(records: list[dict], max_new_tokens: int) -> str | None:
    """Warn when the generation budget cannot fit the longest expected answer.

    A token budget below the longest ground-truth completion (about one token
    per four characters) cuts every such answer mid-JSON: the outputs come
    back unparseable and accuracy reads 0 by construction. Found by the
    CTF eval runs, where 512- and 1024-token budgets truncated 1145-2168
    token answers.
    """
    longest_chars = 0
    longest_id = None
    for record in records:
        content = (record.get("messages") or [{}])[-1].get("content") or ""
        if len(content) > longest_chars:
            longest_chars, longest_id = len(content), record.get("id")
    estimated = longest_chars // 4
    if estimated > max_new_tokens:
        return (
            f"max_new_tokens {max_new_tokens} is below the longest ground-truth "
            f"completion (record '{longest_id}' needs about {estimated} tokens): "
            f"answers will be cut and scored unparseable, raise --max-new-tokens"
        )
    return None


def run_task_eval(
    records: list[dict],
    generate: Callable[[list], list[str]],
    max_new_tokens: int = 512,
) -> dict:
    """Score generated answers against expected ones.

    ``generate`` receives the list of per-record user messages (one item per
    record, in order) and must return one raw model text per record, in the
    same order. Batched generators amortize ``model.generate`` across records.
    """
    if not records:
        raise BrainforgeError("empty evaluation dataset")
    budget_warning = _budget_warning(records, max_new_tokens)
    if budget_warning is not None:
        logging.getLogger(__name__).warning(budget_warning)
    domains = {
        str(record["domain"]) for record in records if record.get("domain") is not None
    }
    if len(domains) > 1:
        raise BrainforgeError(
            f"mixed domains not supported: run one domain per evaluation, found: {sorted(domains)}"
        )
    domainless = [record for record in records if not record.get("domain")]
    if domains and domainless:
        raise BrainforgeError(
            f"{len(domainless)} record(s) without a domain in a {sorted(domains)[0]} evaluation:"
            " every record must declare its domain or the scoring would"
            " silently mis-score it"
        )
    domain = domains.pop() if domains else "security"
    _eval_pack(domain)
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
    enable_thinking: bool | None = None,
) -> dict:
    """Load the trained model (or accept a served-model generator) and run task eval.

    ``generator`` overrides local model loading: pass
    :func:`build_remote_generator` to score a model served behind an
    OpenAI-compatible endpoint. ``enable_thinking`` controls Qwen3-style
    reasoning at generation time (both paths). The report is only persisted
    next to locally stored weights; served models have no local directory
    to write into.
    """
    from brainforge.dataset.writer import read_jsonl

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    records = read_jsonl(dataset_path)
    if generator is None:
        generator = _build_generator(
            model_path,
            quantization,
            max_new_tokens,
            batch_size,
            enable_thinking=enable_thinking,
        )
    result = run_task_eval(records, generator, max_new_tokens=max_new_tokens)
    # Hub ids (base-model evaluation) resolve outside the local filesystem:
    # the report is only persisted next to locally stored weights.
    if model_path.is_dir():
        (model_path / "task_eval.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
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
    enable_thinking: bool | None = None,
) -> Callable[[list], list[str]]:
    """Generate task answers against a served model (OpenAI-compatible endpoint).

    One chat completion per record prompt; the served model may batch internally.
    Prompts are flattened through ``_flatten_message`` like the local generator,
    and responses are taken from the assistant ``content`` (a thinking trace in
    ``reasoning_content`` is the model's, not the answer). ``enable_thinking``
    rides the OpenAI ``extra_body`` as ``chat_template_kwargs`` (Qwen3 thinking
    control on llama.cpp): thinking traces can consume the whole token budget
    and leave ``content`` empty.
    """
    from concurrent.futures import ThreadPoolExecutor

    from brainforge.providers.base import ChatMessage, ChatRequest

    provider = _build_provider(base_url, api_key)
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if enable_thinking is not None
        else None
    )

    def one(prompt: str) -> str:
        response = provider.complete(
            ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
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


def _generation_template_kwargs(enable_thinking: bool | None) -> dict:
    """Chat-template kwargs for Qwen3-style thinking control.

    ``enable_thinking=False`` skips the reasoning block at generation time;
    thinking traces can otherwise consume the entire completion budget and
    leave the scored answer empty. Templates that do not know the kwarg
    ignore it silently.
    """
    if enable_thinking is None:
        return {}
    return {"enable_thinking": enable_thinking}


def _build_generator(
    model_path: Path,
    quantization: str,
    max_new_tokens: int,
    batch_size: int = 4,
    attn_implementation: str = "sdpa",
    enable_thinking: bool | None = None,
) -> Callable[[list], list[str]]:
    import torch

    from brainforge.training.hardware import plan_for_inference
    from brainforge.training.models import load_student_model

    model, tokenizer = load_student_model(
        model_path, plan_for_inference(quantization, attn_implementation)
    )
    # Left padding keeps every prompt in the batch aligned at the same position.
    tokenizer.padding_side = "left"
    template_kwargs = _generation_template_kwargs(enable_thinking)

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
                    **template_kwargs,
                ).to(model.device)
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
                generated = output[:, inputs["input_ids"].shape[1] :]
                replies.extend(
                    tokenizer.batch_decode(generated, skip_special_tokens=True)
                )
        return replies

    return generate
