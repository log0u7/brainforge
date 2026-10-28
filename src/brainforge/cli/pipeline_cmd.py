import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.table import Table

from brainforge.cli._common import (
    CONFIG_OPTION,
    clamp_errors,
    console,
    err_console,
    load_config_or_exit,
)
from brainforge.config.defaults import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_DATASETS_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_RAG_INDEX_DIR,
)
from brainforge.config.models import Config
from brainforge.domains import get_or_generic
from brainforge.errors import BrainforgeError
from brainforge.models.registry import ModelRegistry
from brainforge.pipeline.engine import PipelineEngine, run_pipeline
from brainforge.providers.base import Provider
from brainforge.rag.embeddings import get_backend
from brainforge.rag.retrieval import Retriever
from brainforge.roles.registry import RoleRegistry

app = typer.Typer(help="Run dataset generation pipelines.", no_args_is_help=True)


def _build_retriever(rag: bool, backend_name: str, index_dir: Path) -> Retriever | None:
    if not rag:
        return None
    index_db = index_dir / "index.db"
    if not index_db.exists():
        return None
    backend = get_backend(backend_name)
    return Retriever(index_dir, backend)


def execute_pipeline(
    cfg: Config,
    pipeline_name: str,
    input_dir: Path,
    output: Path,
    provider_override: Provider | None = None,
    fresh: bool = False,
    rag: bool = True,
    rag_backend: str = "fastembed",
    rejected_dir: Path | None = None,
    usage_log_dir: Path | None = DEFAULT_LOG_DIR,
    kd: bool = False,
) -> tuple[dict, int]:
    cache_path = None if fresh else DEFAULT_CACHE_DIR
    roles = RoleRegistry(cfg, ModelRegistry(cfg, cache_path=cache_path))
    pipeline = cfg.pipelines[pipeline_name]
    pack = get_or_generic(pipeline.domain)
    retriever = _build_retriever(rag, rag_backend, DEFAULT_RAG_INDEX_DIR)
    usage_logger = None
    if usage_log_dir is not None and provider_override is None:
        from brainforge.providers.observability import UsageLogger

        usage_logger = UsageLogger(usage_log_dir)
    engine = PipelineEngine(
        cfg,
        pipeline_name,
        roles,
        pack,
        retriever=retriever,
        usage_logger=usage_logger,
        provider_override=provider_override,
        kd=kd,
    )
    from brainforge.dataset.case_builder import build_cases_from_path

    cases = build_cases_from_path(input_dir)
    if not cases:
        raise BrainforgeError(f"no cases found in {input_dir}")
    kd_sidecar = Path(f"{output}".removesuffix(".jsonl") + ".kd.jsonl") if kd else None
    return (
        run_pipeline(engine, cases, output, rejected_dir, kd_sidecar=kd_sidecar),
        len(cases),
    )


@app.command("run")
def run(
    pipeline_name: str = typer.Argument(..., help="Pipeline name from the configuration"),
    input_dir: Path = typer.Option(DEFAULT_DATA_DIR / "raw", "--input", "-i"),
    output: Path = typer.Option(None, "--output", "-o", help="Output dataset.jsonl path"),
    provider: str = typer.Option(None, "--provider", help="Override all providers (e.g. mock)"),
    fresh: bool = typer.Option(False, "--fresh", help="Bypass the response cache"),
    rag: bool = typer.Option(True, "--rag/--no-rag", help="Use the local RAG index if present"),
    rag_backend: str = typer.Option("fastembed", "--rag-backend"),
    experiment: str | None = typer.Option(None, "--experiment", help="Create an experiment report"),
    kd: bool = typer.Option(
        False,
        "--kd",
        help=(
            "Record judge top-k logprobs into a <dataset>.kd.jsonl sidecar"
            " (requires a provider that serves logprobs)"
        ),
    ),
    config: str = CONFIG_OPTION,
) -> None:
    """Execute a pipeline over the cases in --input and write the dataset."""
    cfg = load_config_or_exit(config)
    if pipeline_name not in cfg.pipelines:
        err_console.print(f"[red]unknown pipeline:[/red] {pipeline_name}")
        raise typer.Exit(code=1)
    output_path = output or (DEFAULT_DATASETS_DIR / f"{pipeline_name}.jsonl")
    provider_override = None
    if provider is not None:
        from brainforge.config.models import ProviderConfig
        from brainforge.providers.mock import MockProvider

        provider_override = MockProvider(provider, ProviderConfig(type="mock"))
    try:
        (stats, case_count) = execute_pipeline(
            cfg,
            pipeline_name,
            input_dir,
            output_path,
            provider_override=provider_override,
            fresh=fresh,
            rag=rag,
            rag_backend=rag_backend,
            rejected_dir=DEFAULT_DATA_DIR / "rejected",
            kd=kd,
        )
    except BrainforgeError as exc:
        err_console.print(f"[red]pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    summary = (
        f"{stats['accepted']} accepted, {stats['rejected']} rejected"
        f" ({case_count} cases) -> {output_path}"
    )
    if stats.get("fallback"):
        summary = (
            f"{stats['accepted']} accepted, {stats['fallback']} fallback, "
            f"{stats['rejected']} rejected ({case_count} cases) -> {output_path}"
        )
    if stats.get("errors"):
        summary += f" [yellow]({stats['errors']} provider errors skipped)[/yellow]"
    console.print(f"[green]{pipeline_name}[/green]: {summary}")
    if experiment:
        from brainforge.dataset.writer import read_jsonl
        from brainforge.experiments.manager import create_experiment

        run_dir = create_experiment(
            experiment,
            cfg,
            dataset_path=output_path,
            records=read_jsonl(output_path),
        )
        console.print(f"[green]experiment saved:[/green] {run_dir}")


@app.command("status")
def status(
    path: Path = typer.Argument(
        ...,
        help="Dataset path or pipeline state file (.state.json) written by `pipeline run`",
    ),
) -> None:
    """Print the live progress of a pipeline run from its state file."""
    state_file = path if path.name.endswith(".state.json") else Path(f"{path}.state.json")
    if not state_file.is_file():
        err_console.print(f"[red]no state file:[/red] {state_file} (run `pipeline run` first)")
        raise typer.Exit(code=1)
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]corrupt state file:[/red] {state_file}: {exc}")
        raise typer.Exit(code=1) from exc
    status_value = state.get("status", "unknown")
    done = state.get("done", 0)
    total = state.get("total_cases", 0)
    console.print(
        f"[bold]{state_file}[/bold]: [bold]{status_value}[/bold] "
        f"({done}/{total} cases done, "
        f"{state.get('accepted', 0)} accepted, "
        f"{state.get('rejected', 0)} rejected, "
        f"{state.get('errors', 0)} errors)"
    )
    current = state.get("current_case")
    if current is not None:
        console.print(f"[yellow]running:[/yellow] {current}")
    table = Table(title=f"Pipeline run: {state_file}")
    table.add_column("case")
    table.add_column("status")
    table.add_column("duration_ms")
    table.add_column("gate_reasons")
    for entry in state.get("cases", []):
        table.add_row(
            str(entry.get("id")),
            str(entry.get("status")),
            str(entry.get("duration_ms")),
            str(entry.get("gate_reasons")),
        )
    console.print(table)


def _kd_command_prefix() -> list[str]:
    """Resolve how to launch brainforge itself from a subprocess."""
    import brainforge

    main_py = Path(brainforge.__file__ or "").parent / "__main__.py"
    if main_py.is_file():
        return [sys.executable, "-m", "brainforge"]
    exe = shutil.which("brainforge")
    if exe is None:
        # Editable installs on Windows may not have the venv Scripts dir on
        # PATH; the entry-point exe sits next to sys.executable.
        sibling = Path(sys.executable).with_name("brainforge.exe")
        exe = str(sibling) if sibling.is_file() else None
    return [exe] if exe is not None else []


def _discard_scratch(scratch_file: Path) -> None:
    scratch_file.unlink(missing_ok=True)
    sidecar = Path(str(scratch_file).removesuffix(".jsonl") + ".kd.jsonl")
    sidecar.unlink(missing_ok=True)


@app.command("kd-pass")
@clamp_errors("kd-pass")
def kd_pass(
    cases_dir: Path = typer.Option(..., "--cases", "-i", help="Directory of case JSON files"),
    out_dir: Path = typer.Option(..., "--out-dir", "-o", help="Output dataset directory"),
    scratch_dir: Path | None = typer.Option(
        None, "--scratch-dir", help="Per-case outputs directory (default: <out-dir>-parts)"
    ),
    only_solved: bool = typer.Option(
        False, "--only-solved", help="Keep only non-fallback (solved) records"
    ),
    kd_timeout: int = typer.Option(900, "--kd-timeout", help="Seconds per case subprocess"),
    config: str = CONFIG_OPTION,
) -> None:
    """Run `pipeline run ctf --kd --no-rag` per case and assemble the KD dataset."""
    if not cases_dir.is_dir():
        raise BrainforgeError(f"cases directory not found: {cases_dir}")
    case_files = sorted(cases_dir.glob("*.json"))
    if not case_files:
        raise BrainforgeError(f"no cases found in {cases_dir}")
    parts = scratch_dir if scratch_dir is not None else Path(f"{out_dir}-parts")
    parts.mkdir(parents=True, exist_ok=True)
    prefix = _kd_command_prefix()
    if not prefix:
        err_console.print(
            "[red]no brainforge executable:[/red]"
            " install the package (brainforge.cli:main) or add brainforge/__main__.py"
        )
        raise typer.Exit(code=1)
    total = len(case_files)
    succeeded = 0
    skipped = 0
    failed = 0
    for index, case_file in enumerate(case_files, 1):
        scratch_file = parts / f"kdgen-{case_file.stem}.jsonl"
        if scratch_file.is_file() and scratch_file.stat().st_size > 0:
            skipped += 1
            console.print(f"[dim]{index}/{total} {case_file.stem}: skip (resume)[/dim]")
            continue
        cmd = [
            *prefix,
            "pipeline",
            "run",
            "ctf",
            "-i",
            str(case_file),
            "-o",
            str(scratch_file),
            "--kd",
            "--no-rag",
        ]
        if config is not None:
            cmd.extend(["-c", config])
        try:
            proc: subprocess.CompletedProcess[str] = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace", timeout=kd_timeout
            )
        except subprocess.TimeoutExpired:
            failed += 1
            err_console.print(
                f"[yellow]{index}/{total} {case_file.stem}: TIMEOUT (> {kd_timeout}s)[/yellow]"
            )
            _discard_scratch(scratch_file)
            continue
        if proc.returncode != 0:
            failed += 1
            err_console.print(
                f"[yellow]{index}/{total} {case_file.stem}:"
                f" FAILED (exit {proc.returncode})[/yellow]"
            )
            _discard_scratch(scratch_file)
            continue
        succeeded += 1
        console.print(f"[dim]{index}/{total} {case_file.stem}: kd ok[/dim]")

    records: list[dict] = []
    sidecar: list[dict] = []
    seen: set[str] = set()
    no_positions = 0
    for case_file in case_files:
        scratch_file = parts / f"kdgen-{case_file.stem}.jsonl"
        if not scratch_file.is_file():
            continue
        entries: dict[str, dict] = {}
        sidecar_path = Path(str(scratch_file).removesuffix(".jsonl") + ".kd.jsonl")
        if sidecar_path.is_file():
            for line in sidecar_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                entry_id = entry.get("id")
                if entry_id is not None:
                    entries[entry_id] = entry
        for line in scratch_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if only_solved and record.get("metadata", {}).get("fallback"):
                continue
            record_id = record.get("id")
            if record_id is None or record_id in seen:
                continue
            if record_id not in entries:
                no_positions += 1
                continue
            seen.add(record_id)
            records.append(record)
            sidecar.append(entries[record_id])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (out_dir / "kd.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in sidecar),
        encoding="utf-8",
    )
    console.print(
        f"[green]kd-pass[/green]: {total} cases ({succeeded} ok, {skipped} skipped,"
        f" {failed} failed), {len(records)} records, {len(sidecar)} sidecar entries,"
        f" {no_positions} without positions -> {out_dir}"
    )
