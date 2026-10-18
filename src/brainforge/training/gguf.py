"""GGUF conversion and Ollama packaging for trained students.

Shells out to llama.cpp (``convert_hf_to_gguf.py`` + ``llama-quantize``) and
optionally registers the result with Ollama. llama.cpp is an external tool:
locate it via ``--llama-cpp-path``, the ``LLAMA_CPP_DIR`` env var, or PATH;
fail fast with an install pointer when missing.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from brainforge.errors import BrainforgeError

_INSTALL_POINTER = (
    "llama.cpp not found; pass --llama-cpp-path, set LLAMA_CPP_DIR, or add "
    "llama.cpp to PATH (git clone https://github.com/ggml-org/llama.cpp)"
)


def _find_llama_cpp(explicit: Path | str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    on_path = shutil.which("convert_hf_to_gguf.py")
    if on_path:
        candidates.append(Path(on_path).parent)
    for candidate in candidates:
        if (candidate / "convert_hf_to_gguf.py").exists():
            return candidate
    raise BrainforgeError(_INSTALL_POINTER)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    except FileNotFoundError as exc:
        raise BrainforgeError(f"command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        label = " ".join(cmd[:2])
        raise BrainforgeError(f"{label} failed with exit code {exc.returncode}") from exc


def export_gguf(
    merged_model_dir: Path | str,
    output_dir: Path | str,
    quant: str = "Q4_K_M",
    llama_cpp_path: Path | str | None = None,
    ollama_name: str | None = None,
) -> dict:
    """Convert a merged HF model to a quantized GGUF and optionally register it.

    Writes ``model-<quant>.gguf`` and a ``Modelfile`` into ``output_dir``.
    When ``ollama_name`` is given and the ``ollama`` binary is available,
    registers the model under that name.
    """
    merged_model_dir = Path(merged_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llama_cpp = _find_llama_cpp(llama_cpp_path)
    quant_gguf = output_dir / f"model-{quant.lower()}.gguf"
    _run(
        [
            sys.executable,
            str(llama_cpp / "convert_hf_to_gguf.py"),
            str(merged_model_dir),
            "--outfile",
            str(output_dir / "model-f16.gguf"),
            "--outtype",
            "f16",
        ]
    )
    _run(
        [
            str(llama_cpp / "llama-quantize"),
            str(output_dir / "model-f16.gguf"),
            str(quant_gguf),
            quant,
        ]
    )
    modelfile = output_dir / "Modelfile"
    modelfile.write_text(f"FROM ./{quant_gguf.name}\n", encoding="utf-8")
    result = {
        "model": str(merged_model_dir),
        "output_dir": str(output_dir),
        "quant": quant,
        "gguf": str(quant_gguf),
        "modelfile": str(modelfile),
        "ollama": None,
    }
    if ollama_name:
        if shutil.which("ollama") is None:
            result["ollama"] = "skipped: ollama binary not found"
        else:
            _run(["ollama", "create", ollama_name, "-f", str(modelfile)], cwd=output_dir)
            result["ollama"] = f"created: ollama run {ollama_name}"
    return result
