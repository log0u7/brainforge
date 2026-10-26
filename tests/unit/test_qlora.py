def test_messages_dataset_tolerates_empty_validation_file(tmp_path):
    """prepare can legitimately emit a 0-record split (tiny datasets): an
    existing-but-empty validation.jsonl must behave like a missing file
    (no eval dataset), not raise 'empty dataset split'."""
    import pytest

    from brainforge.training.qlora import _messages_dataset

    empty = tmp_path / "validation.jsonl"
    empty.write_text("", encoding="utf-8")
    try:
        result = _messages_dataset(empty)
    except Exception as exc:
        pytest.fail(f"empty validation file raised: {exc}")
    assert result is None
