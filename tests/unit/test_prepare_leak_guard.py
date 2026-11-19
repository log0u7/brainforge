"""Anti-leak guard: train/eval ID overlap must fail loudly at prepare time.

The grpo-translated eval split shares 5/7 record IDs with the mr22 train
split (discovered manually after the 8B run). A committed eval protocol
input must be leak-free by construction: prepare() rejects an eval split
whose record IDs overlap its train split.
"""

import json

import pytest

from brainforge.training.prepare import assert_no_id_overlap


def _write_records(path, ids):
    path.write_text(
        "\n".join(
            json.dumps({"id": i, "messages": [{"role": "user", "content": "x"}]}) for i in ids
        )
        + "\n",
        encoding="utf-8",
    )


def test_overlap_raises(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    _write_records(train, ["a", "b", "c"])
    _write_records(eval_, ["c", "d"])
    with pytest.raises(ValueError, match="c"):
        assert_no_id_overlap(train, eval_)


def test_disjoint_passes(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    _write_records(train, ["a", "b"])
    _write_records(eval_, ["c", "d"])
    assert_no_id_overlap(train, eval_)  # no exception


def test_missing_eval_file_is_noop(tmp_path):
    train = tmp_path / "train.jsonl"
    _write_records(train, ["a"])
    assert_no_id_overlap(train, tmp_path / "absent.jsonl")  # no exception
