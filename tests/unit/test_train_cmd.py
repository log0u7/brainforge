"""CLI tests for `brainforge train bench` and `brainforge train consolidate`."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()


def _fake_env(monkeypatch) -> None:
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = type(
        "AutoModelForCausalLM",
        (),
        {"_model_mapping": {"llama": "LlamaForCausalLM"}},
    )
    for name, module in (
        ("torch", torch),
        ("transformers", transformers),
        ("peft", types.ModuleType("peft")),
        ("trl", types.ModuleType("trl")),
        ("datasets", types.ModuleType("datasets")),
        ("bitsandbytes", types.ModuleType("bitsandbytes")),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_train_bench_writes_results_files(project_env, monkeypatch) -> None:
    _fake_env(monkeypatch)
    import brainforge.training.benchmark as bench_mod

    fake_results = [
        {"arm": "no-kd", "accuracy": 0.5, "passed": 3, "total": 10},
        {"arm": "sidecar", "accuracy": 0.7, "passed": 7, "total": 10},
    ]
    calls: dict[str, object] = {}

    def fake_benchmark(**kwargs):
        calls.update(kwargs)
        return fake_results

    monkeypatch.setattr(bench_mod, "run_three_arm_benchmark", fake_benchmark)
    monkeypatch.setattr(bench_mod, "render_markdown_table", lambda results: "# bench")
    monkeypatch.setattr(
        "brainforge.config.load_config", lambda _config: SimpleNamespace(training=None)
    )

    output = project_env / "experiments" / "bench"
    result = runner.invoke(
        app,
        ["train", "bench", "--output", str(output), "--dataset-dir", str(project_env)],
    )
    assert result.exit_code == 0, result.output
    assert "# bench" in result.output
    assert calls["output_dir"] == output
    assert (output / "results.json").exists()
    assert (output / "results.md").exists()
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload == fake_results
    assert (output / "results.md").read_text(encoding="utf-8") == "# bench\n"


def test_train_benchmark_module_exports_contract() -> None:
    import brainforge.training.benchmark as bench_mod

    assert hasattr(bench_mod, "run_three_arm_benchmark")
    assert hasattr(bench_mod, "render_markdown_table")
