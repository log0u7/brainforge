from pathlib import Path

import typer

from brainforge.cli._common import CONFIG_OPTION, console, err_console, load_config_or_exit
from brainforge.config.defaults import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_DATASETS_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_RAG_INDEX_DIR,
)
from brainforge.domains import get_or_generic
from brainforge.errors import BrainforgeError
from brainforge.models.registry import ModelRegistry
from brainforge.pipeline.engine import PipelineEngine, run_pipeline
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
    cfg,
    pipeline_name: str,
    input_dir: Path,
    output: Path,
    provider_override=None,
    fresh: bool = False,
    rag: bool = True,
    rag_backend: str = "fastembed",
    rejected_dir: Path | None = None,
    usage_log_dir: Path | None = DEFAULT_LOG_DIR,
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
    )
    from brainforge.dataset.case_builder import build_cases_from_path

    cases = build_cases_from_path(input_dir)
    if not cases:
        raise BrainforgeError(f"no cases found in {input_dir}")
    return run_pipeline(engine, cases, output, rejected_dir), len(cases)


@app.command("run")
def run(
    pipeline_name: str = typer.Argument(..., help="Pipeline name from the configuration"),
    input_dir: Path = typer.Option(DEFAULT_DATA_DIR / "raw", "--input", "-i"),
    output: Path = typer.Option(None, "--output", "-o", help="Output dataset.jsonl path"),
    provider: str = typer.Option(None, "--provider", help="Override all providers (e.g. mock)"),
    fresh: bool = typer.Option(False, "--fresh", help="Bypass the response cache"),
    rag: bool = typer.Option(True, "--rag/--no-rag", help="Use the local RAG index if present"),
    rag_backend: str = typer.Option("fastembed", "--rag-backend"),
    experiment: str = typer.Option(None, "--experiment", help="Create an experiment report"),
    config: str = CONFIG_OPTION,
):
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
        )
    except BrainforgeError as exc:
        err_console.print(f"[red]pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]{pipeline_name}[/green]: {stats['accepted']} accepted, "
        f"{stats['rejected']} rejected ({case_count} cases) -> {output_path}"
    )
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
