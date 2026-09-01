import json
from pathlib import Path

import pytest


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    config = json.loads((Path(__file__).parents[2] / "config" / "config.json").read_text())
    config["providers"]["mock_only"] = {"type": "mock"}
    for model in config["models"].values():
        model["provider"] = "mock_only"
    config["judge_independence"] = "off"
    (tmp_path / "config" / "config.json").write_text(json.dumps(config))
    (tmp_path / "data" / "raw").mkdir(parents=True)
    examples = Path(__file__).parents[2] / "examples" / "cases"
    for case_file in examples.glob("*.json"):
        (tmp_path / "data" / "raw" / case_file.name).write_text(case_file.read_text())
    return tmp_path
