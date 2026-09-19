import json

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
