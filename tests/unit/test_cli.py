import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()


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


def test_config_validate_ok(project_env):
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0, result.output
    assert "configuration is valid" in result.output


def test_config_providers_table(project_env):
    result = runner.invoke(app, ["config", "providers"])
    assert result.exit_code == 0
    assert "openrouter" in result.output


def test_config_schema_check_ok(project_env):
    result = runner.invoke(app, ["config", "schema", "--write"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["config", "schema", "--check"])
    assert result.exit_code == 0
    assert "in sync" in result.output


def test_models_list(project_env):
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "judge" in result.output


def test_roles_list(project_env):
    result = runner.invoke(app, ["roles", "list"])
    assert result.exit_code == 0
    assert "security_teacher" in result.output


def test_pipeline_run_mock(project_env):
    result = runner.invoke(
        app,
        ["pipeline", "run", "security_dataset", "--input", "data/raw", "--provider", "mock"],
    )
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output
    dataset = project_env / "datasets" / "security_dataset.jsonl"
    assert dataset.is_file()
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert len(records) == 3
    risk_flags = [record["metadata"]["recitation_risk"] for record in records]
    assert not all(risk_flags)
    assert any(risk_flags)


def test_pipeline_run_unknown_pipeline(project_env):
    result = runner.invoke(app, ["pipeline", "run", "ghost", "--provider", "mock"])
    assert result.exit_code == 1


def test_teacher_run_and_ensemble(project_env):
    result = runner.invoke(
        app,
        [
            "teacher",
            "run",
            "--role",
            "security_teacher",
            "--case",
            "case-cmd-injection-001",
            "--cases-dir",
            "data/raw",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["teacher", "ensemble", "--case", "case-cmd-injection-001", "--cases-dir", "data/raw"],
    )
    assert result.exit_code == 0, result.output


def test_dataset_validate(project_env):
    runner.invoke(
        app,
        ["pipeline", "run", "security_dataset", "--input", "data/raw", "--provider", "mock"],
    )
    result = runner.invoke(app, ["dataset", "validate", "datasets/security_dataset.jsonl"])
    assert result.exit_code == 0, result.output
    assert "dataset is valid" in result.output


def test_dataset_inspect(project_env):
    runner.invoke(
        app,
        ["pipeline", "run", "security_dataset", "--input", "data/raw", "--provider", "mock"],
    )
    result = runner.invoke(app, ["dataset", "inspect", "datasets/security_dataset.jsonl"])
    assert result.exit_code == 0, result.output
    assert "recitation_risk" in result.output


def test_train_prepare(project_env):
    runner.invoke(
        app,
        ["pipeline", "run", "security_dataset", "--input", "data/raw", "--provider", "mock"],
    )
    result = runner.invoke(app, ["train", "prepare", "datasets/security_dataset.jsonl"])
    assert result.exit_code == 0, result.output
    assert (project_env / "datasets" / "security_dataset_prepared" / "train.jsonl").is_file()
    assert (
        project_env / "datasets" / "security_dataset_prepared" / "test_postcutoff.jsonl"
    ).is_file()


def test_train_run_stub(project_env):
    result = runner.invoke(app, ["train", "run"])
    assert result.exit_code == 1
    assert "phase 2" in result.output


def test_rag_index_and_search_hashing(project_env):
    result = runner.invoke(app, ["rag", "index", "data/raw", "--backend", "hashing"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["rag", "search", "subprocess shell=True", "--backend", "hashing"])
    assert result.exit_code == 0, result.output
    assert "Search" in result.output
