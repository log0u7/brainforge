import json

from typer.testing import CliRunner

from brainforge.cli import dataset_cmd

runner = CliRunner()


def _write_jsonl(path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_filter_keeps_only_sidecar_ids(tmp_path):
    records = [
        {"id": "r1", "text": "keep-1"},
        {"id": "r2", "text": "drop"},
        {"id": "r3", "text": "keep-2"},
    ]
    sidecar = [{"id": "r1"}, {"id": "r3"}]
    _write_jsonl(tmp_path / "records.jsonl", records)
    _write_jsonl(tmp_path / "sidecar.jsonl", sidecar)
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    kept = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert kept == [{"id": "r1", "text": "keep-1"}, {"id": "r3", "text": "keep-2"}]
    # compact + non-ASCII preserved
    assert (
        out.read_text(encoding="utf-8")
        == '{"id": "r1", "text": "keep-1"}\n{"id": "r3", "text": "keep-2"}\n'
    )
    assert "2/3" in result.output
    assert "2" in result.output  # sidecar entry count


def test_filter_preserves_unicode_compact(tmp_path):
    records = [{"id": "r1", "text": "café ☕"}]
    sidecar = [{"id": "r1"}]
    _write_jsonl(tmp_path / "records.jsonl", records)
    _write_jsonl(tmp_path / "sidecar.jsonl", sidecar)
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == '{"id": "r1", "text": "café ☕"}\n'


def test_filter_out_dir_writes_train_and_kd(tmp_path):
    records = [{"id": "r1"}, {"id": "r2"}]
    sidecar = [{"id": "r1", "kd": True}, {"id": "rX", "kd": False}]
    _write_jsonl(tmp_path / "records.jsonl", records)
    _write_jsonl(tmp_path / "sidecar.jsonl", sidecar)
    out_dir = tmp_path / "train_dir"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    train = (out_dir / "train.jsonl").read_text(encoding="utf-8")
    kd = (out_dir / "kd.jsonl").read_text(encoding="utf-8")
    assert train == '{"id": "r1"}\n'
    # all sidecar entries, same order, untouched
    assert kd == '{"id": "r1", "kd": true}\n{"id": "rX", "kd": false}\n'
    assert sorted(p.name for p in out_dir.iterdir()) == ["kd.jsonl", "train.jsonl"]


def test_filter_out_dir_replaces_existing(tmp_path):
    records = [{"id": "r1"}]
    sidecar = [{"id": "r1"}]
    _write_jsonl(tmp_path / "records.jsonl", records)
    _write_jsonl(tmp_path / "sidecar.jsonl", sidecar)
    out_dir = tmp_path / "train_dir"
    out_dir.mkdir()
    (out_dir / "train.jsonl").write_text("stale", encoding="utf-8")
    (out_dir / "kd.jsonl").write_text("stale", encoding="utf-8")

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (out_dir / "stale.jsonl").exists()
    assert (out_dir / "train.jsonl").is_file()
    assert (out_dir / "kd.jsonl").is_file()


def test_filter_rejects_both_out_and_out_dir(tmp_path):
    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "r.jsonl"),
            "--sidecar",
            str(tmp_path / "s.jsonl"),
            "--out",
            str(tmp_path / "o.jsonl"),
            "--out-dir",
            str(tmp_path / "d"),
        ],
    )
    assert result.exit_code == 1
    assert "--out" in result.output or "out-dir" in result.output


def test_filter_requires_out_or_out_dir(tmp_path):
    result = runner.invoke(
        dataset_cmd.app,
        ["filter", "--dataset", str(tmp_path / "r.jsonl"), "--sidecar", str(tmp_path / "s.jsonl")],
    )
    assert result.exit_code == 1
    assert "--out" in result.output or "out-dir" in result.output


def test_filter_refuses_out_same_as_dataset(tmp_path):
    records = [{"id": "r1"}]
    sidecar = [{"id": "r1"}]
    _write_jsonl(tmp_path / "records.jsonl", records)
    _write_jsonl(tmp_path / "sidecar.jsonl", sidecar)
    same = tmp_path / "records.jsonl"
    before = same.read_text(encoding="utf-8")

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(same),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(same),
        ],
    )

    assert result.exit_code == 1
    assert same.read_text(encoding="utf-8") == before


def test_filter_skips_blank_lines(tmp_path):
    (tmp_path / "records.jsonl").write_text('{"id": "r1"}\n\n{"id": "r2"}\n', encoding="utf-8")
    (tmp_path / "sidecar.jsonl").write_text('{"id": "r1"}\n\n', encoding="utf-8")
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == '{"id": "r1"}\n'


def test_filter_malformed_record_line_exits_cleanly(tmp_path):
    (tmp_path / "records.jsonl").write_text('{"id": "r1"}\nnot-json\n', encoding="utf-8")
    _write_jsonl(tmp_path / "sidecar.jsonl", [{"id": "r1"}])
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 1
    assert "not-json" in result.output or "line 2" in result.output
    assert not out.exists()


def test_filter_malformed_sidecar_line_exits_cleanly(tmp_path):
    _write_jsonl(tmp_path / "records.jsonl", [{"id": "r1"}])
    (tmp_path / "sidecar.jsonl").write_text('{"id": "r1"}\n{oops\n', encoding="utf-8")
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 1
    assert not out.exists()


def test_filter_sidecar_entry_missing_id_dropped(tmp_path):
    _write_jsonl(tmp_path / "records.jsonl", [{"id": "r1"}])
    _write_jsonl(tmp_path / "sidecar.jsonl", [{"no_id": True}, {"id": "r1"}])
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "without string id" in result.output
    kept = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [r["id"] for r in kept] == ["r1"]


def test_filter_truncates_long_ids_in_summary(tmp_path):
    long_id = "x" * 80
    _write_jsonl(tmp_path / "records.jsonl", [{"id": long_id}])
    _write_jsonl(tmp_path / "sidecar.jsonl", [{"id": long_id}])
    out = tmp_path / "filtered.jsonl"

    result = runner.invoke(
        dataset_cmd.app,
        [
            "filter",
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--sidecar",
            str(tmp_path / "sidecar.jsonl"),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert ("x" * 40) in result.output
    assert long_id not in result.output


def test_filter_rejects_non_object_line(tmp_path):
    recs = tmp_path / "recs.jsonl"
    recs.write_text("[1, 2, 3]\n", encoding="utf-8")
    side = tmp_path / "kd.jsonl"
    side.write_text('{"id": "a"}\n', encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        dataset_cmd.app,
        ["filter", "-d", str(recs), "-s", str(side), "--out", str(tmp_path / "o.jsonl")],
    )
    assert result.exit_code != 0, result.output


def test_filter_refuses_out_dir_with_unexpected_content(tmp_path):
    recs = tmp_path / "recs.jsonl"
    recs.write_text('{"id": "a", "text": "x"}\n', encoding="utf-8")
    side = tmp_path / "kd.jsonl"
    side.write_text('{"id": "a"}\n', encoding="utf-8")
    out_dir = tmp_path / "snap"
    out_dir.mkdir()
    (out_dir / "precious.txt").write_text("keep me", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        dataset_cmd.app, ["filter", "-d", str(recs), "-s", str(side), "--out-dir", str(out_dir)]
    )
    assert result.exit_code != 0, result.output
    assert (out_dir / "precious.txt").exists()


def test_filter_refuses_out_equal_to_sidecar(tmp_path):
    side = tmp_path / "kd.jsonl"
    side.write_text('{"id": "a"}\n', encoding="utf-8")
    recs = tmp_path / "recs.jsonl"
    recs.write_text('{"id": "a", "text": "x"}\n', encoding="utf-8")
    result = runner.invoke(
        dataset_cmd.app, ["filter", "-d", str(recs), "-s", str(side), "--out", str(side)]
    )
    assert result.exit_code != 0, result.output
    assert side.read_text(encoding="utf-8") == '{"id": "a"}\n'


def test_filter_drops_non_string_ids(tmp_path):
    recs = tmp_path / "recs.jsonl"
    recs.write_text(
        '{"id": "a", "text": "x"}\n{"text": "no-id"}\n{"id": 7, "text": "y"}\n',
        encoding="utf-8",
    )
    side = tmp_path / "kd.jsonl"
    side.write_text('{"id": "a"}\n{"id": null}\n{"id": 7}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        dataset_cmd.app, ["filter", "-d", str(recs), "-s", str(side), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    kept = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [r["id"] for r in kept] == ["a"]


def test_filter_rejects_non_utf8_input(tmp_path):
    recs = tmp_path / "recs.jsonl"
    recs.write_bytes(b'{"id": "a", "text": "\xff\xfe"}\n')
    side = tmp_path / "kd.jsonl"
    side.write_text('{"id": "a"}\n', encoding="utf-8")
    result = runner.invoke(
        dataset_cmd.app,
        ["filter", "-d", str(recs), "-s", str(side), "--out", str(tmp_path / "o.jsonl")],
    )
    assert result.exit_code != 0, result.output
