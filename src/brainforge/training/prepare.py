from pathlib import Path

from brainforge.dataset.split import postcutoff_warning, split_with_postcutoff
from brainforge.dataset.validation import validate_dataset
from brainforge.dataset.writer import DatasetRecord, read_jsonl, write_jsonl
from brainforge.errors import BrainforgeError


def tool_turn_warning(dataset_path: Path | str, assistant_only_loss: bool) -> str | None:
    """Return a warning when the split holds agentic multi-turn records but
    the run trains without ``assistant_only_loss``.

    Agentic records interleave assistant tool calls with ``tool``
    observations or carry several assistant turns. Without
    ``assistant_only_loss`` the trainer scores those observation tokens
    too, so the flag matters exactly here. Returns ``None`` when the flag
    is set, the records are single-turn, or the split is absent.
    """
    if assistant_only_loss:
        return None
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
        " turns; training without assistant_only_loss scores those tokens too."
        " Set training.assistant_only_loss=true for agentic datasets"
        " (requires a chat template with generation markers)."
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
