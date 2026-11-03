"""Tests for the SFT trails miner.

``scripts/build_sft_trails.py`` is not part of the ``brainforge``
package: the tests load it directly from the repository layout. The
contract under test is the contamination one: only the training-
eligible lab originals (the 6 leaked targets) may be mined from the
benchmark run dirs; every eval-challenge directory is skipped and an
all-eval run produces zero records and a nonzero exit.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_sft_trails.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_sft_trails", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run_dir(root: Path) -> Path:
    """One run dir with one training-eligible target and one eval target."""
    run = root / "run-test"
    eligible = run / "fmtstr"
    eligible.mkdir(parents=True)
    (eligible / "trail-fmtstr.jsonl").write_text(
        json.dumps({"run": "r1", "type": "plan", "selected": "fmtstr_write"}) + "\n",
        encoding="utf-8",
    )
    (eligible / "target.facts").write_text('vuln("fmtstr").\n', encoding="utf-8")
    (eligible / "run.json").write_text(
        json.dumps(
            {
                "status": "flagged",
                "attempts": [
                    {
                        "technique": "fmtstr_write",
                        "executed": True,
                        "success": True,
                        "flag": "FLAG{fmtstr_train}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evaled = run / "uaf_heap-002"
    evaled.mkdir(parents=True)
    (evaled / "trail-uaf_heap-002.jsonl").write_text(
        json.dumps({"run": "r2", "type": "plan", "selected": "fmtstr_write"}) + "\n",
        encoding="utf-8",
    )
    (evaled / "run.json").write_text(
        json.dumps({"status": "flagged", "attempts": []}), encoding="utf-8"
    )
    return run


def test_mine_writes_only_training_eligible_targets(tmp_path: Path) -> None:
    module = _load_script()
    run = _fake_run_dir(tmp_path)
    out = tmp_path / "trails.jsonl"
    summary = module.mine_run_dirs([run], out)
    assert summary["records"] == 1
    assert summary["skipped_eval"] == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == "lab-fmtstr-001"
    assert "uaf_heap" not in out.read_text(encoding="utf-8")


def test_mine_record_is_an_agentic_trajectory(tmp_path: Path) -> None:
    module = _load_script()
    run = _fake_run_dir(tmp_path)
    out = tmp_path / "trails.jsonl"
    module.mine_run_dirs([run], out)
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["messages"][0]["role"] == "user"
    verdict = json.loads(record["messages"][-1]["content"])
    assert verdict["flag"] == "FLAG{fmtstr_train}"
    assert verdict["verdict"] == "solved"
    assert record["metadata"]["flag_verified"] is True


def test_mine_all_eval_run_produces_zero_records_and_nonzero_exit(tmp_path: Path) -> None:
    module = _load_script()
    run = tmp_path / "run-eval-only"
    evaled = run / "srop-001"
    evaled.mkdir(parents=True)
    (evaled / "trail-srop-001.jsonl").write_text(
        json.dumps({"run": "r", "type": "plan", "selected": "ret2libc"}) + "\n",
        encoding="utf-8",
    )
    (evaled / "run.json").write_text(
        json.dumps({"status": "flagged", "attempts": []}), encoding="utf-8"
    )
    out = tmp_path / "trails.jsonl"
    summary = module.mine_run_dirs([run], out)
    assert summary["records"] == 0
    assert summary["skipped_eval"] == 1
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--run-dir", str(run), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr


def test_unknown_target_dir_is_skipped_and_reported(tmp_path: Path) -> None:
    """Neither eval nor training-eligible: skipped, never written, and
    reported so the operator can decide what it is."""
    module = _load_script()
    run = tmp_path / "run-unknown"
    mystery = run / "mystery-target"
    mystery.mkdir(parents=True)
    (mystery / "trail-mystery-target.jsonl").write_text("{}", encoding="utf-8")
    out = tmp_path / "trails.jsonl"
    summary = module.mine_run_dirs([run], out)
    assert summary["records"] == 0
    assert summary["skipped_unknown"] == ["mystery-target"]
