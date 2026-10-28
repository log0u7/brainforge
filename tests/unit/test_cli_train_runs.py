"""Tests for `brainforge train runs`: summary table over run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from brainforge.cli.train_cmd import app

runner = CliRunner()


def _make_run(
    root: Path,
    name: str,
    *,
    backend: str = "sft",
    strategy: str = "qlora",
    rows: int = 3,
    epochs: float = 3.0,
    steps: int = 14,
    eval_loss: float | None = None,
    perplexity: float | None = None,
    n_records: int | None = None,
    alpha: float | None = None,
    temp: float | None = None,
) -> None:
    run_dir = root / "experiments" / name
    run_dir.mkdir(parents=True)
    summary: dict[str, Any] = {
        "backend": backend,
        "effective_strategy": strategy,
        "train_rows": rows,
        "epochs": epochs,
        "log_history": [dict(loss=1.0) for _ in range(steps + 1)],
    }
    if alpha is not None:
        summary["kd_alpha"] = alpha
    if temp is not None:
        summary["kd_temperature"] = temp
    (run_dir / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if eval_loss is not None:
        ev: dict[str, Any] = {"eval_loss": eval_loss}
        if perplexity is not None:
            ev["perplexity"] = perplexity
        if n_records is not None:
            ev["n_records"] = n_records
        (run_dir / "eval.json").write_text(json.dumps(ev), encoding="utf-8")


def _invoke(root: Path) -> Any:
    import os

    cwd = os.getcwd()
    os.chdir(root)
    try:
        return runner.invoke(app, ["runs"])
    finally:
        os.chdir(cwd)


def test_runs_lists_summary_and_eval(tmp_path: Any) -> None:
    _make_run(
        tmp_path,
        "run-a",
        backend="kd",
        strategy="qlora",
        rows=17,
        steps=36,
        alpha=0.5,
        temp=2.0,
        eval_loss=2.999,
        perplexity=20.07,
        n_records=26,
    )
    result = _invoke(tmp_path)
    assert result.exit_code == 0
    assert "run-a" in result.output
    assert "kd" in result.output
    assert "17" in result.output
    assert "36" in result.output
    assert "0.5" in result.output
    assert "20.07" in result.output
    assert "26" in result.output


def test_runs_without_eval_still_lists_run(tmp_path: Any) -> None:
    _make_run(tmp_path, "run-b", rows=31, steps=53)
    result = _invoke(tmp_path)
    assert result.exit_code == 0
    assert "run-b" in result.output
    assert "31" in result.output


def test_runs_empty_dir_prints_notice(tmp_path: Any) -> None:
    result = _invoke(tmp_path)
    assert result.exit_code == 0
    assert "no runs" in result.output.lower()


def test_runs_skips_dirs_without_summary(tmp_path: Any) -> None:
    (tmp_path / "experiments" / "junk").mkdir(parents=True)
    _make_run(tmp_path, "run-c", rows=3)
    result = _invoke(tmp_path)
    assert result.exit_code == 0
    assert "run-c" in result.output
    assert "junk" not in result.output


def test_runs_sorted_by_name(tmp_path: Any) -> None:
    _make_run(tmp_path, "zeta", rows=1)
    _make_run(tmp_path, "alpha", rows=2)
    result = _invoke(tmp_path)
    assert result.output.index("alpha") < result.output.index("zeta")
