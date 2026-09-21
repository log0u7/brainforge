"""Cross-boundary integration tests via subprocess on the real filesystem.

Opt-in via BRAINFORCE_IT=1 (marker ``integration``): they run the installed
CLI as a subprocess with the mock provider, no API keys and no GPU.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("BRAINFORCE_IT") != "1",
        reason="integration tests are opt-in via BRAINFORCE_IT=1",
    ),
]


def run_cli(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", "from brainforge.cli import app; app()", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    config = json.loads((REPO_ROOT / "config" / "config.json").read_text())
    config["providers"]["mock_only"] = {"type": "mock"}
    for model in config["models"].values():
        model["provider"] = "mock_only"
    config["judge_independence"] = "off"
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.json").write_text(json.dumps(config))
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for case_file in (REPO_ROOT / "examples" / "cases").glob("*.json"):
        shutil.copy(case_file, raw / case_file.name)
    return tmp_path


def test_cli_chain_pipeline_validate_prepare(workspace: Path):
    dataset = workspace / "datasets" / "security_dataset.jsonl"
    run_cli(
        workspace,
        "pipeline",
        "run",
        "security_dataset",
        "--input",
        str(workspace / "data" / "raw"),
        "--output",
        str(dataset),
        "--provider",
        "mock",
    )
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert records

    run_cli(workspace, "dataset", "validate", str(dataset))

    prepared = workspace / "prepared"
    run_cli(workspace, "train", "prepare", str(dataset), "--output", str(prepared))
    splits = {}
    for name in ("train", "validation", "test", "test_postcutoff"):
        assert (prepared / f"{name}.jsonl").is_file(), name
        splits[name] = sum(
            1 for line in (prepared / f"{name}.jsonl").read_text().splitlines() if line.strip()
        )
    stats = json.loads((prepared / "stats.json").read_text())
    for name, count in splits.items():
        assert stats[name] == count, name
    assert sum(splits.values()) == len(records)
