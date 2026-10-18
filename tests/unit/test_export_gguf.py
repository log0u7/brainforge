"""GGUF export and Ollama packaging tests.

llama.cpp is shelled out to; these tests fake subprocess.run and
shutil.which so no external tool is needed.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from brainforge.cli import app
from brainforge.errors import BrainforgeError
from brainforge.training.gguf import _find_llama_cpp, export_gguf

runner = CliRunner()


@pytest.fixture
def fake_llama_cpp(tmp_path, monkeypatch):
    """A fake llama.cpp checkout with both tools present."""
    llama_cpp = tmp_path / "llama.cpp"
    llama_cpp.mkdir()
    (llama_cpp / "convert_hf_to_gguf.py").write_text("print('convert')")
    (llama_cpp / "llama-quantize").write_text("#!/bin/sh\n")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    return llama_cpp, calls


def test_find_llama_cpp_explicit_path(fake_llama_cpp):
    llama_cpp, _ = fake_llama_cpp
    assert _find_llama_cpp(llama_cpp) == llama_cpp


def test_find_llama_cpp_env_var(fake_llama_cpp, monkeypatch):
    llama_cpp, _ = fake_llama_cpp
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama_cpp))
    assert _find_llama_cpp() == llama_cpp


def test_find_llama_cpp_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(BrainforgeError, match=r"llama\.cpp not found"):
        _find_llama_cpp()


def test_export_gguf_runs_convert_and_quantize(fake_llama_cpp, tmp_path):
    llama_cpp, calls = fake_llama_cpp
    merged = tmp_path / "merged"
    merged.mkdir()
    result = export_gguf(merged, tmp_path / "out", quant="Q4_K_M", llama_cpp_path=llama_cpp)
    assert len(calls) == 2
    convert_cmd, quantize_cmd = calls
    assert convert_cmd[0] == sys.executable
    assert "convert_hf_to_gguf.py" in convert_cmd[1]
    assert str(merged) in convert_cmd
    assert "--outtype" in convert_cmd
    assert "llama-quantize" in quantize_cmd[0]
    assert "Q4_K_M" in quantize_cmd
    modelfile = tmp_path / "out" / "Modelfile"
    assert modelfile.read_text() == "FROM ./model-q4_k_m.gguf\n"
    assert result["gguf"] == str(tmp_path / "out" / "model-q4_k_m.gguf")
    assert result["ollama"] is None


def test_export_gguf_custom_quant(fake_llama_cpp, tmp_path):
    llama_cpp, calls = fake_llama_cpp
    merged = tmp_path / "merged"
    merged.mkdir()
    result = export_gguf(merged, tmp_path / "out", quant="Q8_0", llama_cpp_path=llama_cpp)
    assert "Q8_0" in calls[1]
    assert result["quant"] == "Q8_0"


def test_export_gguf_ollama_registration(fake_llama_cpp, tmp_path, monkeypatch):
    llama_cpp, calls = fake_llama_cpp
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/ollama" if name == "ollama" else None
    )
    merged = tmp_path / "merged"
    merged.mkdir()
    result = export_gguf(
        merged, tmp_path / "out", llama_cpp_path=llama_cpp, ollama_name="triage-bot"
    )
    assert calls[-1][:3] == ["ollama", "create", "triage-bot"]
    assert result["ollama"] == "created: ollama run triage-bot"


def test_export_gguf_ollama_skipped_without_binary(fake_llama_cpp, tmp_path, monkeypatch):
    llama_cpp, calls = fake_llama_cpp
    monkeypatch.setattr("shutil.which", lambda name: None)
    merged = tmp_path / "merged"
    merged.mkdir()
    result = export_gguf(
        merged, tmp_path / "out", llama_cpp_path=llama_cpp, ollama_name="triage-bot"
    )
    assert len(calls) == 2
    assert result["ollama"].startswith("skipped")


def test_export_gguf_command_failure_raises(fake_llama_cpp, tmp_path, monkeypatch):
    llama_cpp, _ = fake_llama_cpp

    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("subprocess.run", failing_run)
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(BrainforgeError, match="failed with exit code"):
        export_gguf(merged, tmp_path / "out", llama_cpp_path=llama_cpp)


def test_cli_export_gguf_format(project_env, monkeypatch):
    calls = {}

    def fake_merge(model, output):
        calls["merge"] = Path(output)
        return {"model": str(model), "output_dir": str(output)}

    def fake_gguf(merged_dir, output, quant="Q4_K_M", llama_cpp_path=None, ollama_name=None):
        calls["gguf"] = {
            "merged_dir": Path(merged_dir),
            "output": Path(output),
            "quant": quant,
            "ollama_name": ollama_name,
        }
        return {
            "output_dir": str(output),
            "quant": quant,
            "gguf": str(output / "model.gguf"),
            "modelfile": str(output / "Modelfile"),
            "ollama": None,
        }

    monkeypatch.setattr("brainforge.training.qlora.export", fake_merge)
    monkeypatch.setattr("brainforge.training.gguf.export_gguf", fake_gguf)
    result = runner.invoke(
        app,
        [
            "train",
            "export",
            "--model",
            "models/run-1",
            "--format",
            "gguf",
            "--quant",
            "Q5_K_M",
            "--ollama",
            "triage-bot",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["gguf"]["quant"] == "Q5_K_M"
    assert calls["gguf"]["ollama_name"] == "triage-bot"
    assert calls["gguf"]["merged_dir"] == calls["merge"]
