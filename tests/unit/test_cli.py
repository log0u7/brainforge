import json

from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()


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


def test_train_run_fails_without_gpu(project_env):
    result = runner.invoke(app, ["train", "run"])
    assert result.exit_code == 1
    assert "training failed" in result.output


def test_rag_index_and_search_hashing(project_env):
    result = runner.invoke(app, ["rag", "index", "data/raw", "--backend", "hashing"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["rag", "search", "subprocess shell=True", "--backend", "hashing"])
    assert result.exit_code == 0, result.output
    assert "Search" in result.output
