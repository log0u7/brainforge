import json

from brainforge.dataset.writer import DatasetRecord


def validate_record(record: dict) -> list[str]:
    errors: list[str] = []
    try:
        parsed = DatasetRecord.model_validate(record)
    except Exception as exc:
        return [f"schema validation failed: {exc}"]
    messages = parsed.messages
    if not messages:
        return ["record has no messages"]
    if messages[0].role != "user":
        errors.append("first message must be from the user")
    if messages[-1].role != "assistant":
        errors.append("last message must be from the assistant")
    for message in messages:
        if message.role not in {"user", "assistant", "system"}:
            errors.append(f"unexpected message role '{message.role}'")
        if not message.content.strip():
            errors.append(f"empty content in '{message.role}' message")
    try:
        json.loads(messages[-1].content)
    except json.JSONDecodeError as exc:
        errors.append(f"assistant content is not valid JSON: {exc}")
    metadata = parsed.metadata
    for key in ("source", "teachers", "quality", "recitation_risk"):
        if key not in metadata:
            errors.append(f"metadata is missing '{key}'")
    if metadata.get("quality", {}).get("passed") is not True:
        errors.append("metadata quality gate did not pass")
    return errors


def validate_dataset(records: list[dict]) -> dict:
    valid: list[dict] = []
    invalid: list[dict] = []
    for index, record in enumerate(records):
        errors = validate_record(record)
        if errors:
            invalid.append({"index": index, "id": record.get("id"), "errors": errors})
        else:
            valid.append(record)
    return {"valid": valid, "invalid": invalid}
