"""CLI: dataset enrich-flags stamps verified flags in place."""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from brainforge.cli import app


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def test_enrich_flags_cli_stamps_and_rewrites(tmp_path: Any) -> None:

    dataset = tmp_path / "ds.jsonl"
    record = {
        "id": "ctf-lictor-fmtstr-001",
        "domain": "ctf",
        "messages": [
            {"role": "user", "content": "Solve this CTF challenge."},
            {"role": "assistant", "content": json.dumps({"verdict": "partial", "flag": None})},
        ],
        "metadata": {},
    }
    _write(dataset, [record])
    case = {
        "id": "ctf-lictor-fmtstr-001",
        "metadata": {"tier": "verified", "expected_flag": "FLAG{the_format_is_the_weapon}"},
    }
    _write(tmp_path / "cases" / "ctf-lictor-fmtstr-001.json", [case])
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "dataset",
            "enrich-flags",
            str(dataset),
            "--cases-dir",
            str(tmp_path / "cases"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1/1 records" in result.output
    line = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
    assert json.loads(line["messages"][-1]["content"])["flag"] == "FLAG{the_format_is_the_weapon}"
    assert line["metadata"]["flag_verified"] is True


def test_enrich_flags_cli_missing_dataset(tmp_path: Any) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["dataset", "enrich-flags", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 1
    assert "file not found" in result.output
