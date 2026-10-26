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


def test_model_init_kwargs_pins_device_map():
    """The default auto dispatch sends quantized modules to the CPU on a
    busy 10GB card; the trainer kwargs pin the resolved device instead."""
    from types import SimpleNamespace

    from brainforge.training.qlora import _model_init_kwargs

    plan = SimpleNamespace(attn_implementation="sdpa", device="cuda:0")
    kwargs = _model_init_kwargs(plan)
    assert kwargs == {"attn_implementation": "sdpa", "device_map": "cuda:0"}
