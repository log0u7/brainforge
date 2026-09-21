import json

import pytest

from brainforge.dataset.writer import DatasetRecord
from brainforge.errors import BrainforgeError
from brainforge.training.prepare import prepare


def make_record(index: int, date: str = "2026-08-01") -> dict:
    return DatasetRecord(
        id=f"case-{index:03d}",
        domain="security",
        messages=[
            {"role": "user", "content": "review this"},
            {"role": "assistant", "content": '{"verdict": "confirmed", "cwe": "CWE-78"}'},
        ],
        metadata={
            "source": {"type": "git", "date": date},
            "teachers": [],
            "quality": {"passed": True},
            "recitation_risk": False,
        },
    ).model_dump()


def write_dataset(tmp_path, records):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_prepare_empty_dataset_raises(tmp_path):
    path = write_dataset(tmp_path, [])
    with pytest.raises(BrainforgeError, match="empty"):
        prepare(path, tmp_path / "out")


def test_prepare_invalid_records_raise(tmp_path):
    records = [make_record(0), {"id": "bad", "messages": []}]
    path = write_dataset(tmp_path, records)
    with pytest.raises(BrainforgeError, match="1 invalid record") as exc:
        prepare(path, tmp_path / "out")
    assert "index 1" in str(exc.value)
    assert not (tmp_path / "out").exists()


def test_prepare_writes_all_splits_and_stats(tmp_path):
    records = [make_record(index) for index in range(6)]
    path = write_dataset(tmp_path, records)
    target = tmp_path / "prepared"
    stats = prepare(path, target)
    for name in ("train", "validation", "test", "test_postcutoff"):
        assert (target / f"{name}.jsonl").is_file()
    counts = {name: stats[name] for name in ("train", "validation", "test", "test_postcutoff")}
    assert sum(counts.values()) == len(records)
    assert (target / "stats.json").is_file()
    assert json.loads((target / "stats.json").read_text()) == stats


def test_prepare_flags_postcutoff_warning_for_small_dataset(tmp_path):
    records = [make_record(index) for index in range(3)]
    path = write_dataset(tmp_path, records)
    stats = prepare(path, tmp_path / "out")
    assert "warning" in stats
