from pathlib import Path

import typer
from rich.table import Table

from brainforge.cli._common import CONFIG_OPTION, console, err_console, load_config_or_exit
from brainforge.config.defaults import DEFAULT_DATA_DIR, DEFAULT_RAG_INDEX_DIR
from brainforge.errors import BrainforgeError
from brainforge.rag.embeddings import get_backend
from brainforge.rag.retrieval import Retriever

app = typer.Typer(help="Local RAG index and search.", no_args_is_help=True)


@app.command("index")
def index(
    path: Path = typer.Argument(DEFAULT_DATA_DIR / "raw", help="Directory or file to index"),
    index_dir: Path = typer.Option(DEFAULT_RAG_INDEX_DIR, "--index-dir"),
    backend_name: str = typer.Option("fastembed", "--backend", help="fastembed or hashing"),
    config: str = CONFIG_OPTION,
):
    """Index documents from a directory into the local vector store."""
    load_config_or_exit(config)
    try:
        backend = get_backend(backend_name)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    retriever = Retriever(index_dir, backend)
    try:
        stats = retriever.index(path)
    except BrainforgeError as exc:
        err_console.print(f"[red]indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]indexed[/green] {stats['documents']} documents,"
        f" {stats['chunks']} chunks -> {index_dir}"
    )


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--k", help="Number of results"),
    index_dir: Path = typer.Option(DEFAULT_RAG_INDEX_DIR, "--index-dir"),
    backend_name: str = typer.Option("fastembed", "--backend"),
    config: str = CONFIG_OPTION,
):
    """Search the local RAG index."""
    load_config_or_exit(config)
    retriever = Retriever(index_dir, get_backend(backend_name))
    results = retriever.search(query, k=k)
    if not results:
        console.print("[yellow]no results (empty or missing index?)[/yellow]")
        return
    table = Table(title=f"Search: {query[:50]}")
    table.add_column("Score")
    table.add_column("Source")
    table.add_column("Chunk")
    table.add_column("Text")
    for result in results:
        table.add_row(
            f"{result.score:.4f}",
            result.source,
            result.chunk_id,
            result.text[:100].replace("\n", " "),
        )
    console.print(table)
