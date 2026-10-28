"""CLI tests for the `brainforge ops` sub-app."""

import json
import os
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brainforge.cli.ops_cmd import app

runner = CliRunner()


def _state(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- ops status -------------------------------------------------------------


def test_status_prints_done_total_and_current(tmp_path):
    _state(tmp_path, "jobA", {"done": 3, "total": 10, "current": "case-42 summary"})
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "jobA" in result.output
    assert "3/10" in result.output
    assert "case-42 summary" in result.output
    assert "STALE" not in result.output


def test_status_defaults_done_zero_total_minus_one(tmp_path):
    _state(tmp_path, "bare", {})
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "bare" in result.output
    assert "0/-1" in result.output


def test_status_truncates_current_to_60_chars(tmp_path):
    long = "x" * 100
    _state(tmp_path, "long", {"done": 1, "total": 2, "current": long})
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "x" * 60 in result.output
    assert "x" * 61 not in result.output


def test_status_flags_stale_file(tmp_path):
    path = _state(tmp_path, "old", {"done": 1, "total": 2})
    old = time.time() - 10_000
    os.utime(path, (old, old))
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "STALE" in result.output


def test_status_custom_stale_after(tmp_path):
    path = _state(tmp_path, "mid", {"done": 1, "total": 2})
    mtime = time.time() - 100
    os.utime(path, (mtime, mtime))
    stale = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path), "--stale-after", "50"])
    fresh = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path), "--stale-after", "500"])
    assert "STALE" in stale.output
    assert "STALE" not in fresh.output


def test_status_skips_malformed_state_file(tmp_path):
    bad = tmp_path / "broken.state.json"
    bad.write_text("{not json", encoding="utf-8")
    _state(tmp_path, "good", {"done": 1, "total": 1})
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "broken" in result.output  # warning line mentions the file
    assert "good" in result.output  # healthy file still reported


@pytest.mark.skipif(sys.platform == "win32", reason="chmod semantics")
def test_status_skips_unreadable_state_file(tmp_path):
    locked = _state(tmp_path, "locked", {"done": 1, "total": 1})
    locked.chmod(0o000)
    try:
        result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    finally:
        locked.chmod(0o644)
    assert result.exit_code == 0, result.output
    assert "locked" in result.output


def test_status_missing_dir_friendly_exit_zero(tmp_path):
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path / "nope")])
    assert result.exit_code == 0
    assert "no state" in result.output.lower() or "not found" in result.output.lower()


def test_status_no_state_files_friendly_exit_zero(tmp_path):
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no state" in result.output.lower()


def test_status_reports_log_paths(tmp_path):
    _state(tmp_path, "job", {"done": 1, "total": 1})
    log = tmp_path / "run.log"
    log.write_text("line0\nline1\n\nline2\nline3\n", encoding="utf-8")
    result = runner.invoke(app, ["status", "--datasets-dir", str(tmp_path), "--logs", str(log)])
    assert result.exit_code == 0, result.output
    assert "run.log" in result.output
    assert "line2" in result.output
    assert "line3" in result.output  # last 3 non-empty lines, blank dropped
    assert "line0" not in result.output


# --- ops log ----------------------------------------------------------------


def test_log_missing_file_exit_zero(tmp_path):
    result = runner.invoke(app, ["log", str(tmp_path / "absent.log")])
    assert result.exit_code == 0
    assert "absent" in result.output or "not found" in result.output


def test_log_tail_last_n_non_empty_lines(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n\n", encoding="utf-8")
    result = runner.invoke(app, ["log", str(log), "--lines", "5"])
    assert result.exit_code == 0, result.output
    for i in range(16, 21):
        assert f"line{i}" in result.output
    assert "line15" not in result.output
    assert "line20" in result.output  # blank trailing line ignored


def test_log_default_lines_twelve(tmp_path):
    log = tmp_path / "run.log"
    lines = [f"line{i}" for i in range(1, 21)]
    log.write_text("\n".join(lines), encoding="utf-8")
    result = runner.invoke(app, ["log", str(log)])
    assert result.exit_code == 0, result.output
    assert "line9" in result.output  # last 12 = lines 9..20
    assert "line8" not in result.output


def test_log_decodes_utf16_le(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("alpha\nbeta\n", encoding="utf-16-le")
    result = runner.invoke(app, ["log", str(log)])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output


def test_log_falls_back_to_cp1252(tmp_path):
    # Odd byte count defeats utf-8 and utf-16-le; 0xE9 is valid cp1252 (é).
    log = tmp_path / "run.log"
    log.write_bytes(b"caf\xe9\n")
    result = runner.invoke(app, ["log", str(log)])
    assert result.exit_code == 0, result.output
    assert "café" in result.output


def test_log_marks_error_lines(tmp_path):
    log = tmp_path / "run.log"
    body = "\n".join(["started ok", "Traceback (most recent call last):", "boom"])
    log.write_text(body, encoding="utf-8")
    result = runner.invoke(app, ["log", str(log)])
    assert result.exit_code == 0, result.output
    assert "Traceback" in result.output
    assert "boom" in result.output


def test_log_replaces_undecodable_bytes(tmp_path):
    log = tmp_path / "run.log"
    # Odd byte count defeats utf-16-le; 0x81 is invalid utf-8 and undefined cp1252.
    log.write_bytes(b"x\x81y")
    result = runner.invoke(app, ["log", str(log)])
    assert result.exit_code == 0, result.output
    assert "\ufffd" in result.output
