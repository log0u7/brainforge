from pathlib import Path

import typer

from brainforge.cli._common import console, err_console
from brainforge.errors import BrainforgeError

app = typer.Typer(help="Training preparation (QLoRA ships in phase 2).", no_args_is_help=True)


@app.command("prepare")
def prepare(
    dataset: Path = typer.Argument(..., help="Path to dataset.jsonl"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    seed: int = typer.Option(42, "--seed"),
):
    """Validate, split and export a dataset for TRL training."""
    from brainforge.training.prepare import prepare

    target = output_dir or Path("datasets") / (dataset.stem + "_prepared")
    try:
        stats = prepare(dataset, target, train_ratio, val_ratio, seed)
    except BrainforgeError as exc:
        err_console.print(f"[red]prepare failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]prepared dataset[/green] -> {target}")
    for name, count in stats.items():
        console.print(f"  {name}: {count}")


@app.command("run")
def run(
    config: str = typer.Option(None, "--config", "-c"),
):
    """Start QLoRA training (phase 2, not implemented yet)."""
    from brainforge.config import load_config
    from brainforge.errors import ConfigError
    from brainforge.training.qlora import train_qlora

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        train_qlora(cfg.training, "datasets/prepared", Path(cfg.training.output_dir))
    except BrainforgeError as exc:
        err_console.print(f"[yellow]not available:[/yellow] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("evaluate")
def evaluate():
    """Evaluate a trained student (phase 2)."""
    from brainforge.training.qlora import evaluate as evaluate_stub

    try:
        evaluate_stub("models/student", "datasets/test_postcutoff.jsonl")
    except BrainforgeError as exc:
        err_console.print(f"[yellow]not available:[/yellow] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("export")
def export():
    """Export a trained student (phase 2)."""
    from brainforge.training.qlora import export as export_stub

    try:
        export_stub("models/student", "models/export")
    except BrainforgeError as exc:
        err_console.print(f"[yellow]not available:[/yellow] {exc}")
        raise typer.Exit(code=1) from exc
