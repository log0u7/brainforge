from pathlib import Path

import typer
from rich.table import Table

from brainforge.cli._common import console, err_console
from brainforge.dataset.dedup import find_duplicates
from brainforge.dataset.validation import validate_dataset
from brainforge.dataset.writer import DatasetRecord, read_jsonl
from brainforge.errors import BrainforgeError

app = typer.Typer(help="Dataset validation, inspection and splitting.", no_args_is_help=True)


def _load_records(path: Path) -> list[DatasetRecord]:
    if not path.is_file():
        err_console.print(f"[red]dataset not found:[/red] {path}")
        raise typer.Exit(code=1)
    raw = read_jsonl(path)
    if not raw:
        err_console.print(f"[red]dataset is empty:[/red] {path}")
        raise typer.Exit(code=1)
    return [DatasetRecord.model_validate(record) for record in raw]


@app.command("build")
def build(
    pipeline_name: str = typer.Argument(..., help="Pipeline name from the configuration"),
    input_dir: Path = typer.Option(Path("data/raw"), "--input", "-i"),
    output: Path = typer.Option(None, "--output", "-o"),
    provider: str = typer.Option("mock", "--provider", help="Provider override (default: mock)"),
    config: str = typer.Option(None, "--config", "-c"),
) -> None:
    """Build a dataset by running a pipeline (defaults to the mock provider)."""
    from brainforge.cli.pipeline_cmd import run as pipeline_run

    pipeline_run(
        pipeline_name=pipeline_name,
        input_dir=input_dir,
        output=output,
        provider=provider,
        fresh=False,
        rag=True,
        rag_backend="fastembed",
        experiment=None,
        config=config,
    )


@app.command("validate")
def validate(
    path: Path = typer.Argument(..., help="Path to dataset.jsonl"),
) -> None:
    """Validate every record of a dataset."""
    raw = read_jsonl(path) if path.is_file() else []
    if not raw:
        err_console.print(f"[red]dataset not found or empty:[/red] {path}")
        raise typer.Exit(code=1)
    result = validate_dataset(raw)
    invalid = result["invalid"]
    if invalid:
        for item in invalid[:10]:
            err_console.print(
                f"[red]invalid record {item['index']} ({item['id']}):[/red] {item['errors']}"
            )
        err_console.print(f"[red]{len(invalid)} invalid records[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]dataset is valid[/green] ({len(raw)} records)")


@app.command("inspect")
def inspect(
    path: Path = typer.Argument(..., help="Path to dataset.jsonl"),
) -> None:
    """Show dataset statistics: contamination, agreement, duplicates."""
    records = _load_records(path)
    total = len(records)
    risk = sum(1 for r in records if r.metadata.get("recitation_risk"))
    postcutoff = total - risk
    table = Table(title=f"Dataset: {path}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("records", str(total))
    table.add_row("recitation_risk", f"{risk} ({risk / total:.1%})")
    table.add_row("post-cutoff (primary eval signal)", f"{postcutoff} ({postcutoff / total:.1%})")
    agreement_totals: dict[str, list[bool]] = {}
    for record in records:
        for role, agreed in record.metadata.get("teacher_agreement", {}).items():
            agreement_totals.setdefault(role, []).append(bool(agreed))
    for role, values in agreement_totals.items():
        rate = sum(values) / len(values) if values else 0.0
        flag = " [red](>=95%: check correlated bias)[/red]" if rate >= 0.95 else ""
        table.add_row(f"agreement {role}", f"{rate:.1%}{flag}")
    try:
        duplicates = find_duplicates(records)
        table.add_row("duplicates", str(len(duplicates)))
    except BrainforgeError:
        pass
    console.print(table)


@app.command("split")
def split(
    path: Path = typer.Argument(..., help="Path to dataset.jsonl"),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    seed: int = typer.Option(42, "--seed"),
    config: str = typer.Option(None, "--config", "-c"),
) -> None:
    """Split a dataset into train/validation/test (+ post-cutoff holdout)."""
    from brainforge.training.prepare import prepare

    target = output_dir or path.parent / (path.stem + "_split")
    try:
        stats = prepare(path, target, train_ratio, val_ratio, seed)
    except BrainforgeError as exc:
        err_console.print(f"[red]split failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]split written[/green] -> {target}")
    for name, count in stats.items():
        console.print(f"  {name}: {count}")
