import hashlib

from brainforge.dataset.writer import DatasetRecord


def group_key(record: DatasetRecord) -> str:
    source = record.metadata.get("source", {})
    if not isinstance(source, dict):
        return "unknown"
    for key in ("repository", "path", "url"):
        value = source.get(key)
        if value:
            return str(value)
    return str(source.get("type", "unknown"))


def _group_bucket(group: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    train_boundary = int(train_ratio * 100)
    val_boundary = int((train_ratio + val_ratio) * 100)
    if bucket < train_boundary:
        return "train"
    if bucket < val_boundary:
        return "validation"
    return "test"


def split_dataset(
    records: list[DatasetRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[DatasetRecord]]:
    splits: dict[str, list[DatasetRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in records:
        splits[_group_bucket(group_key(record), seed, train_ratio, val_ratio)].append(record)
    return splits


def is_postcutoff(record: DatasetRecord) -> bool:
    return record.metadata.get("recitation_risk") is False


def postcutoff_records(records: list[DatasetRecord]) -> list[DatasetRecord]:
    return [record for record in records if is_postcutoff(record)]


def split_with_postcutoff(
    records: list[DatasetRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    min_postcutoff: int = 20,
) -> dict[str, list[DatasetRecord]]:
    splits = split_dataset(records, train_ratio, val_ratio, seed)
    splits["test_postcutoff"] = postcutoff_records(records)
    return splits


def postcutoff_warning(
    splits: dict[str, list[DatasetRecord]], min_postcutoff: int = 20
) -> str | None:
    count = len(splits.get("test_postcutoff", []))
    if count < min_postcutoff:
        return (
            f"only {count} post-cutoff evaluation cases available "
            f"(minimum recommended: {min_postcutoff}); add more recent sources"
        )
    return None
