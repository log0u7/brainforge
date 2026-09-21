import typer
from rich.table import Table

from brainforge.cli._common import CONFIG_OPTION, console, load_config_or_exit
from brainforge.config.loader import default_schema_path, schema_matches, write_schema
from brainforge.errors import ConfigError
from brainforge.providers.registry import build_provider

app = typer.Typer(help="Configuration validation and inspection.", no_args_is_help=True)


@app.command("validate")
def validate(config: str = CONFIG_OPTION) -> None:
    """Validate the configuration file and dry-build all providers."""
    cfg = load_config_or_exit(config)
    for name, provider_config in cfg.providers.items():
        try:
            build_provider(name, provider_config)
        except ConfigError as exc:
            console.print(f"[red]provider '{name}' failed to build:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    violations = cfg.check_judge_independence()
    if violations:
        for violation in violations:
            console.print(f"[yellow]judge independence warning:[/yellow] {violation}")
    console.print("[green]configuration is valid[/green]")


@app.command("providers")
def providers(config: str = CONFIG_OPTION) -> None:
    """List configured providers."""
    cfg = load_config_or_exit(config)
    table = Table(title="Providers")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Base URL")
    table.add_column("API style")
    table.add_column("API key")
    for name, provider in cfg.providers.items():
        table.add_row(
            name,
            provider.type.value,
            provider.base_url or "(default)",
            provider.api_style.value,
            "set" if provider.api_key else "(none)",
        )
    console.print(table)


@app.command("schema")
def schema(
    write: bool = typer.Option(False, "--write", help="Write schema.json from the models"),
    check: bool = typer.Option(False, "--check", help="Verify schema.json is in sync"),
    config: str = CONFIG_OPTION,
) -> None:
    """Export or verify the JSON Schema generated from the config models."""

    schema_path = default_schema_path(config)
    if write:
        path = write_schema(schema_path)
        console.print(f"[green]schema written:[/green] {path}")
        return
    if not schema_path.exists():
        console.print(f"[red]schema not found:[/red] {schema_path} (run with --write)")
        raise typer.Exit(code=1)
    if schema_matches(schema_path):
        console.print("[green]schema.json is in sync[/green]")
        return
    console.print("[red]schema.json is out of sync[/red]")
    raise typer.Exit(code=1)


@app.command("path")
def path(config: str = CONFIG_OPTION) -> None:
    """Print the resolved configuration file path."""
    from brainforge.config import resolve_config_path

    console.print(resolve_config_path(config))
