"""CLI tests for `brainforge train bench` and `brainforge train consolidate`."""

import json
import sys
import types
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
    monkeypatch.setattr(
        bench_mod, "render_markdown_table", lambda results, domain="security": "# bench"
    )
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


def _fake_eval_result(fake_result):
    def fake_eval(*args, **kwargs):
        return dict(fake_result)

    return fake_eval


def test_task_eval_display_metrics_come_from_the_pack(project_env, monkeypatch) -> None:
    """ADR-0022 D3 cleanup: the task-eval display shows the domain pack's
    secondary metrics (eval_secondary_metrics hook), not a hardcoded list."""
    _fake_env(monkeypatch)
    import brainforge.training.task_eval as task_eval_mod

    for domain, pack_metric, other_metric in (
        ("ctf", "flag_accuracy", "cwe_accuracy"),
        ("security", "cwe_accuracy", "flag_accuracy"),
    ):
        fake_result = {
            "n_records": 2,
            "accuracy": 0.5,
            "false_positive_rate": 0.25,
            "false_negative_rate": 0.25,
            pack_metric: 0.5,
            "domains": [domain],
        }
        monkeypatch.setattr(
            task_eval_mod,
            "evaluate_model_on_records",
            _fake_eval_result(fake_result),
        )
        result = runner.invoke(app, ["train", "task-eval", "--model", "x"])
        assert result.exit_code == 0, result.output
        assert pack_metric in result.output
        assert other_metric not in result.output
