"""Unit tests for `brainforge pipeline kd-pass` (per-case subprocess + assembly)."""

import json
import subprocess
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from brainforge.cli import app, pipeline_cmd

runner = CliRunner()


def _record(rec_id: str, fallback: bool = False) -> dict[str, Any]:
    return {"id": rec_id, "metadata": {"fallback": fallback}}


def _entry(rec_id: str) -> dict[str, Any]:
    return {"id": rec_id, "model": "teacher", "positions": [[0, 0.9], [1, 0.1]]}


class _Proc:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def _write_case(cases_dir: Path, name: str) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / f"{name}.json").write_text("{}", encoding="utf-8")


def _write_scratch(parts: Path, stem: str, records: list[dict[str, Any]]) -> None:
    parts.mkdir(parents=True, exist_ok=True)
    scratch = parts / f"kdgen-{stem}.jsonl"
    scratch.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _read_ids(path: Path) -> list[str]:
    return [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _install_fake_runner(monkeypatch: Any, plans: dict[str, dict[str, Any]]) -> list[list[str]]:
    """Patch the command prefix and subprocess.run with a plan-driven fake.

    Plans map case stem to: records/entries to write as the subprocess output,
    an optional rc (non-zero failure), an optional timeout flag, or an "error"
    message turned into an AssertionError (subprocess must not run).
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        capture_output: bool = False,
        text: bool = False,
        errors: str | None = None,
        timeout: int | None = None,
    ) -> _Proc:
        calls.append(list(cmd))
        stem = Path(cmd[cmd.index("-i") + 1]).stem
        scratch = Path(cmd[cmd.index("-o") + 1])
        plan = plans[stem]
        if plan.get("error"):
            raise AssertionError(plan["error"])
        if plan.get("records") is not None:
            scratch.parent.mkdir(parents=True, exist_ok=True)
            scratch.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plan["records"]),
                encoding="utf-8",
            )
        if plan.get("entries") is not None:
            sidecar = Path(str(scratch).removesuffix(".jsonl") + ".kd.jsonl")
            sidecar.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in plan["entries"]),
                encoding="utf-8",
            )
        if plan.get("timeout"):
            raise subprocess.TimeoutExpired(cmd, timeout)
        return _Proc(plan.get("rc", 0))

    monkeypatch.setattr(pipeline_cmd, "_kd_command_prefix", lambda: ["bf-fake"])
    monkeypatch.setattr(pipeline_cmd.subprocess, "run", fake_run)
    return calls


def _invoke(tmp_path: Path, extra: tuple[str, ...] = ()) -> Any:
    return runner.invoke(
        app,
        [
            "pipeline",
            "kd-pass",
            "--cases",
            str(tmp_path / "cases"),
            "--out-dir",
            str(tmp_path / "ds"),
            *extra,
        ],
    )


def test_fresh_run_assembles_train_and_kd(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    _write_case(cases, "b")
    plans = {
        "a": {
            "records": [_record("a-ok"), _record("a-nopos")],
            "entries": [_entry("a-ok")],
        },
        "b": {"records": [_record("b-ok")], "entries": [_entry("b-ok")]},
    }
    calls = _install_fake_runner(monkeypatch, plans)
    config = tmp_path / "cfg.json"
    result = _invoke(tmp_path, ("--config", str(config)))

    assert result.exit_code == 0, result.output
    out = tmp_path / "ds"
    assert _read_ids(out / "train.jsonl") == ["a-ok", "b-ok"]
    assert _read_ids(out / "kd.jsonl") == ["a-ok", "b-ok"]
    assert "without positions" in result.output
    assert "2 cases (2 ok, 0 skipped, 0 failed)" in result.output
    assert "--no-rag" in calls[0]
    assert calls[0][-2:] == ["-c", str(config)]


def test_resume_skips_existing_nonempty_scratch(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    _write_case(cases, "b")
    parts = tmp_path / "ds-parts"
    _write_scratch(parts, "a", [_record("a-kept")])
    (parts / "kdgen-a.kd.jsonl").write_text(json.dumps(_entry("a-kept")) + "\n", encoding="utf-8")
    before = (parts / "kdgen-a.jsonl").read_text(encoding="utf-8")
    plans = {
        "a": {"error": "case a must not rerun"},
        "b": {"records": [_record("b-ok")], "entries": [_entry("b-ok")]},
    }
    calls = _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert Path(calls[0][calls[0].index("-i") + 1]).stem == "b"
    assert (parts / "kdgen-a.jsonl").read_text(encoding="utf-8") == before
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["a-kept", "b-ok"]
    assert "1 skipped" in result.output


def test_resume_reruns_empty_scratch_file(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    (tmp_path / "ds-parts").mkdir()
    (tmp_path / "ds-parts" / "kdgen-a.jsonl").write_text("", encoding="utf-8")
    plans = {"a": {"records": [_record("a-ok")], "entries": [_entry("a-ok")]}}
    calls = _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["a-ok"]


def test_nonzero_rc_deletes_partial_and_continues(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    _write_case(cases, "b")
    plans = {
        "a": {"rc": 3, "records": [_record("a-partial")]},
        "b": {"records": [_record("b-ok")], "entries": [_entry("b-ok")]},
    }
    _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    parts = tmp_path / "ds-parts"
    assert not (parts / "kdgen-a.jsonl").exists()
    assert not (parts / "kdgen-a.kd.jsonl").exists()
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["b-ok"]
    assert "FAILED (exit 3)" in result.output
    assert "1 failed" in result.output


def test_timeout_deletes_partial_and_continues(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    _write_case(cases, "b")
    plans = {
        "a": {"timeout": True, "records": [_record("a-partial")]},
        "b": {"records": [_record("b-ok")], "entries": [_entry("b-ok")]},
    }
    _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path, ("--kd-timeout", "7"))

    assert result.exit_code == 0, result.output
    parts = tmp_path / "ds-parts"
    assert not (parts / "kdgen-a.jsonl").exists()
    assert not (parts / "kdgen-a.kd.jsonl").exists()
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["b-ok"]
    assert "TIMEOUT" in result.output


def test_default_keeps_fallback_records(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    plans = {
        "a": {
            "records": [_record("a-fb", fallback=True), _record("a-ok")],
            "entries": [_entry("a-fb"), _entry("a-ok")],
        }
    }
    _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["a-fb", "a-ok"]
    assert _read_ids(tmp_path / "ds" / "kd.jsonl") == ["a-fb", "a-ok"]


def test_only_solved_filters_fallback_records(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    plans = {
        "a": {
            "records": [_record("a-fb", fallback=True), _record("a-ok")],
            "entries": [_entry("a-fb"), _entry("a-ok")],
        }
    }
    _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path, ("--only-solved",))

    assert result.exit_code == 0, result.output
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["a-ok"]
    assert _read_ids(tmp_path / "ds" / "kd.jsonl") == ["a-ok"]


def test_dedupe_keeps_first_by_id(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    _write_case(cases, "b")
    plans = {
        "a": {"records": [_record("dup")], "entries": [_entry("dup")]},
        "b": {"records": [_record("dup")], "entries": [_entry("dup")]},
    }
    _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert _read_ids(tmp_path / "ds" / "train.jsonl") == ["dup"]
    assert _read_ids(tmp_path / "ds" / "kd.jsonl") == ["dup"]


def test_scratch_dir_default_naming(tmp_path: Any, monkeypatch: Any) -> None:
    cases = tmp_path / "cases"
    _write_case(cases, "a")
    plans = {"a": {"records": [_record("a-ok")], "entries": [_entry("a-ok")]}}
    _install_fake_runner(monkeypatch, plans)
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    parts = tmp_path / "ds-parts"
    assert parts.is_dir()
    assert (parts / "kdgen-a.jsonl").is_file()
    assert (parts / "kdgen-a.kd.jsonl").is_file()


def test_missing_cases_dir_errors(tmp_path: Any, monkeypatch: Any) -> None:
    _install_fake_runner(monkeypatch, {})
    result = _invoke(tmp_path)

    assert result.exit_code == 1
    assert "cases directory not found" in result.output


def test_no_case_files_errors(tmp_path: Any, monkeypatch: Any) -> None:
    (tmp_path / "cases").mkdir()
    _install_fake_runner(monkeypatch, {})
    result = _invoke(tmp_path)

    assert result.exit_code == 1
    assert "no cases found" in result.output


def test_command_prefix_resolution(monkeypatch: Any) -> None:
    monkeypatch.setattr(pipeline_cmd.shutil, "which", lambda name: "/usr/bin/brainforge")
    assert pipeline_cmd._kd_command_prefix() == ["/usr/bin/brainforge"]
    monkeypatch.setattr(pipeline_cmd.shutil, "which", lambda name: None)
    assert pipeline_cmd._kd_command_prefix() == []


def test_command_prefix_falls_back_to_sibling_exe(tmp_path: Any, monkeypatch: Any) -> None:
    """Editable install + no PATH entry: the exe next to sys.executable works."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    exe = scripts / "brainforge.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(pipeline_cmd.shutil, "which", lambda name: None)
    monkeypatch.setattr(pipeline_cmd.sys, "executable", str(scripts / "python.exe"))
    assert pipeline_cmd._kd_command_prefix() == [str(exe)]


def test_command_prefix_sibling_missing_still_empty(tmp_path: Any, monkeypatch: Any) -> None:
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    monkeypatch.setattr(pipeline_cmd.shutil, "which", lambda name: None)
    monkeypatch.setattr(pipeline_cmd.sys, "executable", str(scripts / "python.exe"))
    assert pipeline_cmd._kd_command_prefix() == []
