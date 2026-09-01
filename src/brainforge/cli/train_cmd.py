from datetime import datetime
from pathlib import Path

import typer

from brainforge.cli._common import console, err_console
from brainforge.errors import BrainforgeError

app = typer.Typer(
    help="Training: prepare datasets, run QLoRA, evaluate, export.", no_args_is_help=True
)


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
    dataset_dir: Path = typer.Option(
        Path("datasets/prepared"),
        "--dataset-dir",
        "-d",
        help="Directory with train/validation splits",
    ),
    epochs: int = typer.Option(
        0, "--epochs", "-e", help="Override config epochs (0 = keep config)"
    ),
    base_model: str = typer.Option("", "--base-model", "-b", help="Override config base model"),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output dir (default: <config.output_dir>/<run name>)",
    ),
    config: str = typer.Option(None, "--config", "-c"),
):
    """Run QLoRA training (requires a CUDA GPU and the training extra)."""
    from brainforge.config import load_config
    from brainforge.errors import ConfigError
    from brainforge.training.qlora import train_qlora

    try:
        cfg = load_config(config)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    training = cfg.training
    updates = {}
    if epochs:
        updates["epochs"] = epochs
    if base_model:
        updates["base_model"] = base_model
    if updates:
        training = training.model_copy(update=updates)
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_dir = output or Path(training.output_dir) / run_name
    try:
        summary = train_qlora(training, dataset_dir, output_dir)
    except BrainforgeError as exc:
        err_console.print(f"[red]training failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]training done[/green] -> {summary['output_dir']}")
    for key, value in summary.items():
        console.print(f"  {key}: {value}")


@app.command("evaluate")
def evaluate(
    model: Path = typer.Option(
        Path("experiments"), "--model", "-m", help="Trained adapter directory"
    ),
    dataset: Path = typer.Option(
        Path("datasets/prepared/test_postcutoff.jsonl"),
        "--dataset",
        "-d",
        help="Eval split (JSONL)",
    ),
):
    """Evaluate a trained student (loss + perplexity, written to eval.json)."""
    from brainforge.training.qlora import evaluate as evaluate_model

    try:
        result = evaluate_model(model, dataset)
    except BrainforgeError as exc:
        err_console.print(f"[red]evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]evaluated[/green] {result['n_records']} records")
    console.print(f"  eval_loss: {result['eval_loss']:.4f}")
    console.print(f"  perplexity: {result['perplexity']:.4f}")


@app.command("export")
def export(
    model: Path = typer.Option(
        Path("experiments"), "--model", "-m", help="Trained adapter directory"
    ),
    output: Path = typer.Option(
        Path("models/export"), "--output", "-o", help="Merged model output directory"
    ),
):
    """Merge the LoRA adapter into the base model and export it standalone."""
    from brainforge.training.qlora import export as export_model

    try:
        result = export_model(model, output)
    except BrainforgeError as exc:
        err_console.print(f"[red]export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]exported[/green] -> {result['output_dir']}")
