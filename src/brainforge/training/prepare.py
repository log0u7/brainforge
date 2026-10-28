from pathlib import Path

from brainforge.dataset.split import postcutoff_warning, split_with_postcutoff
from brainforge.dataset.validation import validate_dataset
from brainforge.dataset.writer import DatasetRecord, read_jsonl, write_jsonl
from brainforge.errors import BrainforgeError


def tool_turn_warning(dataset_path: Path | str) -> str | None:
    """Return a warning when the split holds agentic multi-turn records.

    Training uses conversational prompt-completion with a completion-only
    loss: the final assistant message is the completion, everything before
    it is masked prompt. Agentic records that interleave assistant tool
    calls with ``tool`` observations lose the training signal of their
    intermediate assistant turns (only the last one is scored). Returns
    ``None`` when every record is single-turn or the split is absent.
    """
    path = Path(dataset_path)
    if not path.is_file():
        return None
    flagged = 0
    for record in read_jsonl(path):
        messages = record.get("messages") or []
        roles = [message.get("role") for message in messages]
        if "tool" in roles or roles.count("assistant") > 1:
            flagged += 1
    if not flagged:
        return None
    return (
        f"{flagged} record(s) contain tool observations or multiple assistant"
        " turns; prompt-completion training scores only the final assistant"
        " turn, so intermediate turns receive no loss. Split agentic records"
        " into per-turn examples if their intermediate turns must be learned."
    )


def prepare(
    dataset_path: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    raw_records = read_jsonl(dataset_path)
    if not raw_records:
        raise BrainforgeError(f"dataset is empty: {dataset_path}")
    checked = validate_dataset(raw_records)
    if checked["invalid"]:
        first = checked["invalid"][0]
        raise BrainforgeError(
            f"dataset contains {len(checked['invalid'])} invalid records; "
            f"first error at index {first['index']} ({first['id']}): {first['errors']}"
        )
    records = [DatasetRecord.model_validate(record) for record in checked["valid"]]
    splits = split_with_postcutoff(records, train_ratio, val_ratio, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int | str] = {}
    for name in ("train", "validation", "test", "test_postcutoff"):
        split_records = splits.get(name, [])
        write_jsonl(output / f"{name}.jsonl", split_records)
        stats[name] = len(split_records)
    warning = postcutoff_warning(splits)
    if warning:
        stats["warning"] = warning
    import json

    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
