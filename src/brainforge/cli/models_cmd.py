import typer
from rich.table import Table

from brainforge.cli._common import CONFIG_OPTION, console, load_config_or_exit
from brainforge.errors import BrainforgeError

app = typer.Typer(help="Model registry operations.", no_args_is_help=True)


@app.command("list")
def list_models(config: str = CONFIG_OPTION):
    """List configured models."""
    cfg = load_config_or_exit(config)
    table = Table(title="Models")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model ID")
    table.add_column("Family")
    table.add_column("Knowledge cutoff")
    for name, model in cfg.models.items():
        table.add_row(
            name,
            model.provider,
            model.model,
            model.family or "-",
            model.knowledge_cutoff or "-",
        )
    console.print(table)


@app.command("test")
def test_model(
    model_name: str = typer.Argument(..., help="Model name from the registry"),
    config: str = CONFIG_OPTION,
):
    """Send a tiny request to a model to verify connectivity."""
    from brainforge.providers.base import ChatMessage, ChatRequest

    cfg = load_config_or_exit(config)
    if model_name not in cfg.models:
        console.print(f"[red]unknown model:[/red] {model_name}")
        raise typer.Exit(code=1)
    from brainforge.models.registry import ModelRegistry

    registry = ModelRegistry(cfg, cache_path=None)
    resolved = registry.resolve(model_name)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Reply with the single word: pong")],
        temperature=0.0,
        max_tokens=16,
    )
    try:
        response = resolved.provider.complete(request, resolved.definition.model)
    except BrainforgeError as exc:
        console.print(f"[red]model test failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]ok[/green] ({response.usage.input_tokens} in / {response.usage.output_tokens} out)"
    )
    content = (response.content or "").strip()
    if content:
        console.print(f"reply: {content[:120]}")
