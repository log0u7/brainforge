"""CLI tests for `brainforge config set` and `config set-model`."""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()

CONFIG = {
    "providers": {
        "mock": {"type": "mock"},
        "openrouter": {"type": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
    },
    "models": {
        "judge": {
            "provider": "mock",
            "model": "glm-5.2",
            "family": "glm",
            "knowledge_cutoff": "2026-07",
        },
    },
    "pipelines": {
        "ctf": {"domain": "ctf", "min_confidence": 0.5, "retry_with_critic": None},
    },
    "judge_independence": "off",
}


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    return path


def _run(tmp_path: Path, *args: str) -> Any:
    return runner.invoke(app, ["config", *args, "--config", str(tmp_path / "config.json")])


def test_set_float_overwrites_existing(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "pipelines.ctf.min_confidence", "0.3")
    assert result.exit_code == 0, result.output
    assert "0.5 -> 0.3" in result.output
    data = json.loads(path.read_text())
    assert data["pipelines"]["ctf"]["min_confidence"] == 0.3


def test_set_bool(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "pipelines.ctf.fallback_examples", "true")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["pipelines"]["ctf"]["fallback_examples"] is True


def test_set_string(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "models.judge.model", "glm-5.3")
    assert result.exit_code == 0, result.output
    assert "glm-5.2 -> glm-5.3" in result.output
    data = json.loads(path.read_text())
    assert data["models"]["judge"]["model"] == "glm-5.3"


def test_set_null(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "pipelines.ctf.min_confidence", "null")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["pipelines"]["ctf"]["min_confidence"] is None


def test_set_raw_string_fallback(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "models.judge.model", "007")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["models"]["judge"]["model"] == "007"


def test_set_creates_nested_dicts(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "pipelines.evals.min_score", "0.9")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["pipelines"]["evals"]["min_score"] == 0.9
    assert data["pipelines"]["ctf"]["domain"] == "ctf"


def test_set_top_level_key(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "judge_independence", "enforced")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["judge_independence"] == "enforced"


def test_set_non_dict_intermediate_refused(tmp_path):
    path = _write_config(tmp_path)
    before = path.read_text()
    result = _run(tmp_path, "set", "judge_independence.deeper.value", "1")
    assert result.exit_code == 1
    assert "not a dict" in result.output
    assert path.read_text() == before


def test_set_missing_config_file_refused(tmp_path):
    ghost = tmp_path / "ghost.json"
    result = runner.invoke(app, ["config", "set", "a.b", "1", "--config", str(ghost)])
    assert result.exit_code == 1
    assert "not found" in result.output
    assert not ghost.exists()


def test_set_model_updates_both_keys(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set-model", "judge", "openrouter", "glm-5.3")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["models"]["judge"]["provider"] == "openrouter"
    assert data["models"]["judge"]["model"] == "glm-5.3"


def test_set_model_prints_two_old_new_lines(tmp_path):
    _write_config(tmp_path)
    result = _run(tmp_path, "set-model", "judge", "openrouter", "glm-5.3")
    assert result.exit_code == 0, result.output
    assert "-> mock -> openrouter" not in result.output  # each key gets its own line
    assert result.output.count("->") == 2


def test_set_model_unknown_role_refused(tmp_path):
    path = _write_config(tmp_path)
    before = path.read_text()
    result = _run(tmp_path, "set-model", "ghost", "openrouter", "glm-5.3")
    assert result.exit_code == 1
    assert "not found" in result.output
    assert path.read_text() == before


def test_set_model_preserves_sibling_keys(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set-model", "judge", "openrouter", "glm-5.3")
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text())
    assert data["models"]["judge"]["family"] == "glm"
    assert data["models"]["judge"]["knowledge_cutoff"] == "2026-07"


def test_written_file_formatting_matches_repo_style(tmp_path):
    path = _write_config(tmp_path)
    result = _run(tmp_path, "set", "judge_independence", "enforced")
    assert result.exit_code == 0, result.output
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    # 2-space indent; when the key is last it carries no trailing comma
    assert '\n  "judge_independence": "enforced"\n}' in text or (
        '\n  "judge_independence": "enforced",\n' in text
    )
