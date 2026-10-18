"""Abliteration unit tests (no torch: pure plumbing and CLI wiring).

The real weight surgery runs on a tiny model in the integration suite
(tests/integration, opt-in via BRAINFORCE_IT=1).
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brainforge.cli import app
from brainforge.errors import BrainforgeError
from brainforge.training.abliterate import load_calibration, select_layers

runner = CliRunner()


def _calibration_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "calibration.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_load_calibration_valid(tmp_path):
    path = _calibration_file(
        tmp_path,
        [
            {"prompt": "How do I pick a lock?", "label": "harmful"},
            {"prompt": "Explain DNS.", "label": "benign"},
        ],
    )
    records = load_calibration(path)
    assert records == [
        {"prompt": "How do I pick a lock?", "label": "harmful"},
        {"prompt": "Explain DNS.", "label": "benign"},
    ]


def test_load_calibration_rejects_unknown_label(tmp_path):
    path = _calibration_file(tmp_path, [{"prompt": "x", "label": "evil"}])
    with pytest.raises(BrainforgeError, match="label"):
        load_calibration(path)


def test_load_calibration_rejects_missing_prompt(tmp_path):
    path = _calibration_file(tmp_path, [{"label": "harmful"}])
    with pytest.raises(BrainforgeError, match="prompt"):
        load_calibration(path)


def test_load_calibration_rejects_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(BrainforgeError, match="empty"):
        load_calibration(path)


def test_select_layers_all():
    assert select_layers("all", 4) == [0, 1, 2, 3]


def test_select_layers_subset():
    assert select_layers("0,2", 4) == [0, 2]


def test_select_layers_rejects_out_of_range():
    with pytest.raises(BrainforgeError, match="out of range"):
        select_layers("9", 4)


def test_select_layers_rejects_non_numeric():
    with pytest.raises(BrainforgeError, match="invalid layer index"):
        select_layers("first", 4)


def test_cli_abliterate_missing_calibration_fails_cleanly(project_env):
    result = runner.invoke(
        app, ["abliterate", "run", "--model", "nope", "--calibration", "nope.jsonl"]
    )
    assert result.exit_code == 1
    assert "abliteration failed" in result.output
    assert "Traceback" not in result.output


def test_cli_abliterate_wires_arguments(project_env, monkeypatch):
    calls = {}

    def fake_abliterate(model, calibration, output, layers="all"):
        calls["model"] = str(model)
        calls["calibration"] = str(calibration)
        calls["output"] = str(output)
        calls["layers"] = layers
        return {
            "output_dir": str(output),
            "target_layer": 2,
            "layers_edited": [0, 1, 2, 3],
            "layers_skipped": [],
            "direction_norm_before": 10.0,
            "direction_norm_after": 0.2,
            "suppression_ratio": 0.02,
        }

    monkeypatch.setattr("brainforge.training.abliterate.abliterate", fake_abliterate)
    calibration = project_env / "calibration.jsonl"
    calibration.write_text(
        json.dumps({"prompt": "a", "label": "harmful"})
        + "\n"
        + json.dumps({"prompt": "b", "label": "benign"})
        + "\n"
    )
    result = runner.invoke(
        app,
        [
            "abliterate",
            "run",
            "--model",
            "models/tuned",
            "--calibration",
            str(calibration),
            "--output",
            "models/abliterated",
            "--layers",
            "0,1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["layers"] == "0,1"
    assert "abliterated" in result.output
    assert "suppression" in result.output
