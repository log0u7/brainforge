from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from brainforge.cli._common import CONFIG_OPTION, clamp_errors, console, load_config_or_exit
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


def _parse_scalar(value: str) -> Any:
    """Parse as JSON scalar first (0.3 -> float, true -> bool, null -> None);
    fall back to the raw string on JSONDecodeError."""
    import json

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if parsed is None or isinstance(parsed, (bool, int, float, str)):
        return parsed
    raise ConfigError(
        f"value must be a JSON scalar or a raw string, got structured value: {value!r}"
    )


def _load_raw(config_path: str | None) -> tuple[Path, dict[str, Any]]:
    """Read the raw JSON config as a dict, refusing missing files (no stray creation)."""
    import json

    from brainforge.config import resolve_config_path

    path = resolve_config_path(config_path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be an object, got {type(raw).__name__}")
    return path, raw


def _set_leaf(raw: dict[str, Any], dotted_path: str, value: Any) -> Any:
    """Navigate/create nested dicts and set the leaf.

    Missing intermediate keys are created as dicts; an existing intermediate
    that is not a dict is refused (would silently shadow structured config).
    """
    keys = dotted_path.split(".")
    node = raw
    for index, key in enumerate(keys[:-1]):
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        elif not isinstance(child, dict):
            raise ConfigError(
                f"cannot descend into '{'.'.join(keys[: index + 1])}'"
                f" (segment '{key}' is not a dict)"
            )
        node = child
    old = node.get(keys[-1])
    node[keys[-1]] = value
    return old


def _write_raw(path: Path, raw: dict[str, Any]) -> None:
    import json

    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


@app.command("set")
@clamp_errors("config set")
def set_value(
    dotted_path: str = typer.Argument(..., help="Dotted path, e.g. pipelines.ctf.min_confidence"),
    value: str = typer.Argument(
        ...,
        help=(
            "Parsed as JSON first (0.3 -> float, true -> bool, null -> None,"
            ' "007" -> string); on JSONDecodeError treated as a raw string.'
        ),
    ),
    config: str = CONFIG_OPTION,
) -> None:
    """Set a nested configuration value by dotted path."""
    new_value = _parse_scalar(value)
    path, raw = _load_raw(config)
    old = _set_leaf(raw, dotted_path, new_value)
    _write_raw(path, raw)
    leaf = dotted_path.rsplit(".", 1)[-1].lower()
    if "key" in leaf or "secret" in leaf or "token" in leaf or "password" in leaf:
        console.print(f"[green]{dotted_path}:[/green] [old value redacted] -> [new value redacted]")
    else:
        console.print(f"[green]{dotted_path}:[/green] {old} -> {new_value}")


@app.command("set-model")
@clamp_errors("config set-model")
def set_model(
    role: str = typer.Argument(..., help="Model role, e.g. judge or security"),
    provider: str = typer.Argument(..., help="Provider name (must exist under providers)"),
    model: str = typer.Argument(..., help="Model name passed to the provider"),
    config: str = CONFIG_OPTION,
) -> None:
    """Set models.<role>.provider and models.<role>.model in one write."""
    path, raw = _load_raw(config)
    models = raw.get("models")
    if not isinstance(models, dict):
        raise ConfigError("config has no 'models' mapping")
    role_entry = models.get(role)
    if not isinstance(role_entry, dict):
        raise ConfigError(f"model role not found: {role}")
    if provider not in raw.get("providers", {}):
        raise ConfigError(f"unknown provider: {provider}")
    old_provider = _set_leaf(role_entry, "provider", provider)
    old_model = _set_leaf(role_entry, "model", model)
    _write_raw(path, raw)
    console.print(f"[green]models.{role}.provider:[/green] {old_provider} -> {provider}")
    console.print(f"[green]models.{role}.model:[/green] {old_model} -> {model}")
