"""Quickstart command tests: one-shot cases -> dataset -> trained student."""

import json
from pathlib import Path

from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()


def test_cli_quickstart_runs_all_steps(project_env, monkeypatch):
    calls: list[str] = []

    def fake_execute(cfg, pipeline_name, input_dir, output, **kwargs):
        calls.append(f"pipeline:{pipeline_name}")
        dataset = Path(output)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "case-x",
            "domain": "security",
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "{}"},
            ],
            "metadata": {},
        }
        dataset.write_text(json.dumps(record) + "\n")
        return {"accepted": 1, "rejected": 0}, 1

    def fake_prepare(dataset, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
        calls.append("prepare")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        split = [
            {
                "id": "case-x",
                "messages": [{"role": "user", "content": "u"}],
                "metadata": {},
            }
        ]
        for name in ("train", "validation", "test", "test_postcutoff"):
            (target / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in split) + "\n")
        return {"train": 1, "validation": 1, "test": 1, "test_postcutoff": 1}

    def fake_train_student(training, dataset_dir, output_dir, resume=False, kd=False):
        calls.append("train")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"base_model": training.base_model, "output_dir": str(output_dir)}

    def fake_evaluate(
        model, dataset, quantization="4bit", batch_size=4, attn_implementation="sdpa"
    ):
        calls.append("evaluate")
        return {"model": str(model), "n_records": 1, "eval_loss": 0.5, "perplexity": 1.6}

    monkeypatch.setattr("brainforge.cli.pipeline_cmd.execute_pipeline", fake_execute)
    monkeypatch.setattr("brainforge.training.prepare.prepare", fake_prepare)
    monkeypatch.setattr("brainforge.cli.train_cmd.train_student", fake_train_student)
    monkeypatch.setattr("brainforge.training.qlora.evaluate", fake_evaluate)

    result = runner.invoke(
        app,
        [
            "quickstart",
            "run",
            "--domain",
            "security",
            "--input",
            "data/raw",
            "--provider",
            "mock",
            "--epochs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == ["pipeline:security_dataset", "prepare", "train", "evaluate"]
    assert "quickstart done" in result.output
    assert "next steps" in result.output.lower()
    assert "export" in result.output.lower()


def test_cli_quickstart_unknown_domain_fails_cleanly(project_env):
    result = runner.invoke(
        app, ["quickstart", "run", "--domain", "kubernetes", "--input", "data/raw"]
    )
    assert result.exit_code == 1
    assert "quickstart failed" in result.output
    assert "Traceback" not in result.output


def test_cli_quickstart_empty_input_fails_cleanly(project_env):
    result = runner.invoke(
        app, ["quickstart", "run", "--domain", "security", "--input", "data/empty"]
    )
    assert result.exit_code == 1
    assert "quickstart failed" in result.output


def test_cli_quickstart_output_override(project_env, monkeypatch):
    seen = {}

    def fake_execute(cfg, pipeline_name, input_dir, output, **kwargs):
        seen["dataset"] = str(output)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(
            json.dumps(
                {
                    "id": "case-x",
                    "domain": "security",
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "{}"},
                    ],
                    "metadata": {},
                }
            )
            + "\n"
        )
        return {"accepted": 1, "rejected": 0}, 1

    def fake_prepare(dataset, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        row = {"id": "x", "messages": [{"role": "user", "content": "u"}], "metadata": {}}
        for name in ("train", "validation", "test", "test_postcutoff"):
            (target / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
        return {"train": 1}

    def fake_train_student(training, dataset_dir, output_dir, resume=False, kd=False):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"output_dir": str(output_dir)}

    def fake_evaluate(*args, **kwargs):
        return {"n_records": 1, "eval_loss": 0.5, "perplexity": 1.6}

    monkeypatch.setattr("brainforge.cli.pipeline_cmd.execute_pipeline", fake_execute)
    monkeypatch.setattr("brainforge.training.prepare.prepare", fake_prepare)
    monkeypatch.setattr("brainforge.cli.train_cmd.train_student", fake_train_student)
    monkeypatch.setattr("brainforge.training.qlora.evaluate", fake_evaluate)

    result = runner.invoke(
        app,
        [
            "quickstart",
            "run",
            "--domain",
            "security",
            "--input",
            "data/raw",
            "--provider",
            "mock",
            "--output",
            "datasets/custom.jsonl",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["dataset"] == "datasets/custom.jsonl"
