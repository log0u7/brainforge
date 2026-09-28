import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brainforge.cli import app
from brainforge.errors import BrainforgeError
from brainforge.training.qlora import extract_messages, train_qlora

runner = CliRunner()


def test_extract_messages_keeps_only_messages():
    records = [
        {
            "id": "a",
            "domain": "security",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {},
        },
        {
            "id": "b",
            "domain": "coding",
            "messages": [{"role": "user", "content": "yo"}],
            "metadata": {},
        },
    ]
    rows = extract_messages(records)
    assert rows == [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "yo"}]},
    ]


def test_extract_messages_rejects_empty():
    with pytest.raises(BrainforgeError, match="empty dataset split"):
        extract_messages([])


def test_extract_messages_rejects_missing_messages():
    with pytest.raises(BrainforgeError, match="index 1"):
        extract_messages(
            [
                {"id": "a", "messages": [{"role": "user", "content": "hi"}]},
                {"id": "b", "metadata": {}},
            ]
        )


def test_train_qlora_fails_cleanly_without_gpu():
    summary = {"id": "x", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(BrainforgeError) as excinfo:
        train_qlora(_training_config(), [summary], "out")
    assert "torch" in str(excinfo.value) or "CUDA" in str(excinfo.value)


def _training_config():
    from brainforge.config.models import TrainingConfig

    return TrainingConfig()


def test_cli_train_run_fails_cleanly_without_gpu(project_env, monkeypatch):
    dataset_dir = project_env / "datasets" / "prepared"
    dataset_dir.mkdir(parents=True)
    split = [{"id": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {}}]
    for name in ("train.jsonl", "validation.jsonl"):
        (dataset_dir / name).write_text("\n".join(json.dumps(r) for r in split) + "\n")
    result = runner.invoke(app, ["train", "run", "--dataset-dir", str(dataset_dir)])
    assert result.exit_code == 1
    assert "training failed" in result.output


def test_cli_train_run_resume_flag_fails_cleanly_without_gpu(project_env):
    dataset_dir = project_env / "datasets" / "prepared"
    dataset_dir.mkdir(parents=True)
    split = [{"id": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {}}]
    for name in ("train.jsonl", "validation.jsonl"):
        (dataset_dir / name).write_text("\n".join(json.dumps(r) for r in split) + "\n")
    result = runner.invoke(app, ["train", "run", "--dataset-dir", str(dataset_dir), "--resume"])
    assert result.exit_code == 1
    assert "training failed" in result.output


def test_latest_run_dir_uses_mtime_not_name(project_env):
    import os
    import time

    from brainforge.cli.train_cmd import latest_run_dir

    base = project_env / "experiments"
    name_latest = base / "run-20260101-000000"
    mtime_latest = base / "run-20251231-235959"
    name_latest.mkdir(parents=True)
    mtime_latest.mkdir(parents=True)
    os.utime(name_latest, (0, 0))
    os.utime(mtime_latest, (time.time(), time.time()))
    assert latest_run_dir().name == "run-20251231-235959"


def test_latest_run_dir_respects_config_output_dir(project_env):
    from brainforge.cli.train_cmd import latest_run_dir

    (project_env / "experiments").mkdir(exist_ok=True)
    run_dir = project_env / "artifacts" / "run-20260920-000001"
    run_dir.mkdir(parents=True)
    with (project_env / "config" / "config.json").open("r") as handle:
        config = json.load(handle)
    config["training"]["output_dir"] = "artifacts"
    (project_env / "config" / "config.json").write_text(json.dumps(config))
    assert latest_run_dir().name == "run-20260920-000001"


def test_train_qlora_resume_without_checkpoint_fails_fast(tmp_path):
    output_dir = tmp_path / "run"
    with pytest.raises(BrainforgeError, match="no checkpoint"):
        train_qlora(_training_config(), tmp_path, output_dir, resume=True)


def test_latest_checkpoint_sorts_numerically(tmp_path):
    from brainforge.training.qlora import _latest_checkpoint

    (tmp_path / "checkpoint-999").mkdir()
    (tmp_path / "checkpoint-1000").mkdir()
    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"


def test_latest_checkpoint_ignores_non_numeric_names(tmp_path):
    from brainforge.training.qlora import _latest_checkpoint

    (tmp_path / "checkpoint-42").mkdir()
    (tmp_path / "checkpoint-not-a-step").mkdir()
    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint-42"


def test_cli_train_run_resume_reuses_latest_run(project_env, monkeypatch):
    calls = {}

    def fake_train_qlora(config, dataset_dir, output_dir, resume=False):
        calls["output_dir"] = Path(output_dir)
        calls["resume"] = resume
        raise BrainforgeError("stop")

    monkeypatch.setattr("brainforge.training.qlora.train_qlora", fake_train_qlora)
    latest = project_env / "experiments" / "run-20260920-000001"
    latest.mkdir(parents=True)
    result = runner.invoke(app, ["train", "run", "--resume"])
    assert result.exit_code == 1
    assert calls.get("output_dir") == latest
    assert calls["resume"] is True


def test_cli_train_run_missing_dataset_fails_cleanly(project_env, monkeypatch):
    import os

    def boom(config, dataset_dir, output_dir, resume=False):
        raise FileNotFoundError(os.path.join(str(dataset_dir), "train.jsonl"))

    monkeypatch.setattr("brainforge.training.qlora.train_qlora", boom)
    result = runner.invoke(app, ["train", "run"])
    assert result.exit_code == 1
    assert "training failed" in result.output
    assert "Traceback" not in result.output


def test_cli_train_task_eval_fails_cleanly(project_env):
    result = runner.invoke(app, ["train", "task-eval", "--model", str(project_env / "nope")])
    assert result.exit_code == 1
    assert "task evaluation failed" in result.output


def test_cli_train_chat_missing_model_fails(project_env):
    result = runner.invoke(app, ["train", "chat", "--model", str(project_env / "nope")])
    assert result.exit_code == 1
    assert "chat failed" in result.output


def test_evaluate_signature_accepts_quantization():
    import inspect

    from brainforge.training.qlora import evaluate

    params = inspect.signature(evaluate).parameters
    assert "quantization" in params
    assert params["quantization"].default == "4bit"


def test_smoke_cpu_skips_without_torch(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    from brainforge.training import smoke_cpu

    assert smoke_cpu.main() == 0
    assert "skipping" in capsys.readouterr().out.lower()


def test_smoke_cpu_fails_when_deps_missing(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    from brainforge.training import smoke_cpu

    assert smoke_cpu.main() == 1
