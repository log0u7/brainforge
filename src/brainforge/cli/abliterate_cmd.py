from pathlib import Path

import typer

from brainforge.cli._common import clamp_errors, console

app = typer.Typer(
    help="Abliteration: remove refusal directions from model weights (experimental).",
    no_args_is_help=True,
)


@app.command("run")
@clamp_errors("abliteration")
def run(
    model: Path = typer.Option(..., "--model", "-m", help="HF model id or local directory"),
    calibration: Path = typer.Option(
        ..., "--calibration", help="JSONL of {prompt, label: harmful|benign} records"
    ),
    output: Path = typer.Option(
        Path("models/abliterated"), "--output", "-o", help="Abliterated model directory"
    ),
    layers: str = typer.Option(
        "all", "--layers", help="all or comma-separated layer indices to ablate"
    ),
) -> None:
    """Remove the refusal direction from a model's weights (experimental)."""
    from brainforge.training.abliterate import abliterate

    stats = abliterate(model, calibration, output, layers=layers)
    console.print(f"[green]abliterated[/green] -> {stats['output_dir']}")
    console.print(f"  layers edited: {stats['layers_edited']}")
    if stats["layers_skipped"]:
        console.print(f"  layers skipped: {stats['layers_skipped']}")
    console.print(
        f"  direction norm: {stats['direction_norm_before']:.3f} ->"
        f" {stats['direction_norm_after']:.3f}"
    )
    console.print(f"  suppression ratio: {stats['suppression_ratio']:.3f}")
