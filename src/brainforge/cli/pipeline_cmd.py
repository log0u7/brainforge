import json
from pathlib import Path

import typer
from rich.table import Table

from brainforge.cli._common import CONFIG_OPTION, console, err_console, load_config_or_exit
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
