import json
import shutil
from pathlib import Path

import typer
from rich.table import Table

from brainforge.cli._common import clamp_errors, console, err_console
from brainforge.dataset.dedup import find_duplicates
from brainforge.dataset.validation import validate_dataset
from brainforge.dataset.writer import DatasetRecord, read_jsonl
from brainforge.errors import BrainforgeError

app = typer.Typer(help="Dataset validation, inspection and splitting.", no_args_is_help=True)


def _read_jsonl_generic(path: Path) -> list[dict]:
    """Read a JSONL file as raw dicts, skipping blank lines (no schema validation)."""
    if not path.is_file():
        err_console.print(f"[red]file not found:[/red] {path}")
        raise typer.Exit(code=1)
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            err_console.print(f"[red]invalid JSON at {path}:{lineno}:[/red] {line[:80]}")
            raise BrainforgeError(str(exc)) from exc
        if not isinstance(entry, dict):
            err_console.print(f"[red]not a JSON object at {path}:{lineno}:[/red] {line[:80]}")
            raise BrainforgeError("expected a JSON object per line")
        entries.append(entry)
    return entries


@app.command("filter")
@clamp_errors("dataset filter")
def filter(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Input records JSONL"),
    sidecar: Path = typer.Option(..., "--sidecar", "-s", help="Sidecar JSONL with an id field"),
    out: Path = typer.Option(None, "--out", help="Write filtered records JSONL here"),
    out_dir: Path = typer.Option(None, "--out-dir", help="Write train.jsonl + kd.jsonl here"),
) -> None:
    """Keep records whose id appears in the sidecar; copy the sidecar for training."""
    if out and out_dir:
        err_console.print("[red]--out and --out-dir are mutually exclusive[/red]")
        raise typer.Exit(code=1)
    if not out and not out_dir:
        err_console.print("[red]provide exactly one of --out or --out-dir[/red]")
        raise typer.Exit(code=1)
    if out and out.resolve() == dataset.resolve():
        err_console.print(f"[red]--out must not overwrite the input dataset:[/red] {out}")
        raise typer.Exit(code=1)

    sidecar_entries = _read_jsonl_generic(sidecar)
    ids = set()
    for index, entry in enumerate(sidecar_entries):
        if "id" not in entry:
            err_console.print(f"[red]sidecar entry {index} has no id[/red]")
            raise typer.Exit(code=1)
        ids.add(entry["id"])
    records = _read_jsonl_generic(dataset)
    kept = [record for record in records if record.get("id") in ids]

    written = out or out_dir / "train.jsonl"
    if out_dir:
        if out_dir.exists():
            unexpected = [
                p.name for p in out_dir.iterdir() if p.name not in ("train.jsonl", "kd.jsonl")
            ]
            if unexpected:
                err_console.print(
                    f"[red]refusing to replace {out_dir}: unexpected content[/red] {unexpected}"
                )
                raise typer.Exit(code=1)
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
    with written.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if out_dir:
        kd = out_dir / "kd.jsonl"
        with kd.open("w", encoding="utf-8") as handle:
            for entry in sidecar_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(f"[green]records kept[/green] {len(kept)}/{len(records)}")
    console.print(f"[green]sidecar entries[/green] {len(sidecar_entries)}")
    display = [str(record_id)[:40] for record_id in sorted(ids)]
    console.print(f"[green]ids[/green] {display}")


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
