"""Quickstart: cases -> distilled dataset -> trained student, in one command.

Chains the existing pipeline, prepare and train steps so a new domain is
usable in minutes: ``brainforge quickstart run --domain ticket_triage``.
The training step requires the training extra; the device and quantization
resolve automatically (use ``--device cpu`` for CPU-only machines). The
dataset steps run anywhere (use ``--provider mock`` for a zero-cost dry run).
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from brainforge.cli._common import clamp_errors, console
from brainforge.config.defaults import DEFAULT_DATASETS_DIR
from brainforge.errors import BrainforgeError

if TYPE_CHECKING:
    from brainforge.config.models import Config

app = typer.Typer(
    help="One-shot: cases -> distilled dataset -> trained student.", no_args_is_help=True
)


def _pipeline_for_domain(cfg: "Config", domain: str) -> str:
    matches = [name for name, pipeline in cfg.pipelines.items() if pipeline.domain == domain]
    if not matches:
        available = sorted({pipeline.domain for pipeline in cfg.pipelines.values()})
        raise BrainforgeError(
            f"no pipeline configured for domain '{domain}'"
            f" (configured domains: {', '.join(available)});"
            " add a pipeline to your config (see docs/reference/configuration.md)"
        )
    return str(sorted(matches)[0])


@app.command("run")
@clamp_errors("quickstart")
def run(
    domain: str = typer.Option(..., "--domain", "-D", help="Domain pack (e.g. ticket_triage)"),
    input_dir: Path = typer.Option(Path("data/raw"), "--input", "-i", help="Cases directory"),
    provider: str = typer.Option(
        None, "--provider", help="Provider override (e.g. mock for a zero-cost dry run)"
    ),
    base_model: str = typer.Option("", "--base-model", "-b", help="Override config base model"),
    epochs: int = typer.Option(1, "--epochs", "-e"),
    strategy: str = typer.Option("qlora", "--strategy", help="qlora or full"),
    backend: str = typer.Option("auto", "--backend", help="auto, trl or unsloth"),
    device: str = typer.Option("auto", "--device", help="auto, cuda, xpu, mps or cpu"),
    quantization: str = typer.Option("auto", "--quantization", help="auto, 4bit, 8bit or none"),
    kd: bool = typer.Option(False, "--kd", help="Use the kd.jsonl sidecar if present"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Dataset output path (default: datasets/<domain>.jsonl)"
    ),
    config: str = typer.Option(None, "--config", "-c"),
) -> None:
    """Distill teachers into a student in one command."""
    from brainforge.cli.pipeline_cmd import execute_pipeline
    from brainforge.cli.train_cmd import train_student
    from brainforge.config import load_config
    from brainforge.config.models import ProviderConfig
    from brainforge.providers.mock import MockProvider
    from brainforge.training.prepare import prepare

    cfg = load_config(config)
    pipeline_name = _pipeline_for_domain(cfg, domain)
    dataset_path = output or (DEFAULT_DATASETS_DIR / f"{domain}.jsonl")
    prepared_dir = DEFAULT_DATASETS_DIR / "prepared"

    console.print(f"[bold]quickstart[/bold] domain={domain} pipeline={pipeline_name}")

    provider_override = None
    if provider is not None:
        provider_override = MockProvider(provider, ProviderConfig(type="mock"))
    (stats, case_count) = execute_pipeline(
        cfg,
        pipeline_name,
        input_dir,
        dataset_path,
        provider_override=provider_override,
        kd=kd,
    )
    console.print(
        f"[green]dataset[/green] {stats['accepted']} accepted,"
        f" {stats['rejected']} rejected ({case_count} cases) -> {dataset_path}"
    )

    console.print("[bold]preparing splits[/bold]")
    prepare(dataset_path, prepared_dir)

    training = cfg.training
    updates: dict = {}
    if epochs:
        updates["epochs"] = epochs
    if base_model:
        updates["base_model"] = base_model
    updates["strategy"] = strategy
    updates["backend"] = backend
    updates["device"] = device
    updates["quantization"] = quantization
    training = training.model_copy(update=updates)
    run_dir = Path(training.output_dir) / f"quickstart-{datetime.now():%Y%m%d-%H%M%S}"
    summary = train_student(training, prepared_dir, run_dir, kd=kd)
    console.print(f"[green]trained[/green] -> {summary['output_dir']}")

    eval_note = ""
    try:
        from brainforge.training.qlora import evaluate

        result = evaluate(summary["output_dir"], prepared_dir / "test_postcutoff.jsonl")
        console.print(
            f"[green]evaluated[/green] {result['n_records']} records,"
            f" eval_loss={result['eval_loss']:.4f}"
        )
    except Exception as exc:  # evaluation is best-effort (e.g. empty holdout)
        eval_note = f" (evaluation skipped: {exc})"

    console.print("[bold]quickstart done[/bold]" + eval_note)
    console.print("next steps:")
    console.print(
        f"  brainforge train export --model {summary['output_dir']} --format gguf --ollama <name>"
    )
    console.print(f"  brainforge train chat --model {summary['output_dir']}")
    console.print("  brainforge train run --strategy full   # heavier decision model")
