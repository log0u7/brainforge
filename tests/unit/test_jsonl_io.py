"""Streaming JSONL reader: bounded RAM for large KD datasets."""

import json

from brainforge.dataset.writer import file_digest, iter_jsonl


def test_iter_jsonl_yields_records_one_at_a_time(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [{"id": "a"}, {"id": "b"}]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert list(iter_jsonl(path)) == records


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('\n{"id": "a"}\n\n{"id": "b"}\n')
    assert list(iter_jsonl(path)) == [{"id": "a"}, {"id": "b"}]


def test_iter_jsonl_does_not_materialize_the_file(tmp_path):
    """Lazy by contract: nothing read until the iterator is consumed."""
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "a"}\n')
    stream = iter_jsonl(path)
    assert not isinstance(stream, list)
    next(iter(stream))


def test_file_digest_is_content_sensitive(tmp_path):
    """Content digest: from_generator cache keys must change when data changes."""
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "a"}\n')
    first = file_digest(path)
    path.write_text('{"id": "b"}\n')
    assert file_digest(path) != first
    path.write_text('{"id": "a"}\n')
    assert file_digest(path) == first


def test_file_digest_is_hex_sha256(tmp_path):
    import hashlib

    path = tmp_path / "data.jsonl"
    path.write_text("payload\n")
    assert file_digest(path) == hashlib.sha256(b"payload\n").hexdigest()
