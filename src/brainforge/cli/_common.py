from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

from brainforge.case import Case
from brainforge.errors import BrainforgeError

if TYPE_CHECKING:
    from brainforge.config.models import Config
    from brainforge.roles.registry import RoleRegistry

console = Console()
err_console = Console(stderr=True)

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Path to config.json")


def clamp_errors(label: str) -> Callable:
    """Turn expected failures (BrainforgeError and OS errors) into a clean exit."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> Any:
            try:
                return fn(*args, **kwargs)
            except (BrainforgeError, OSError) as exc:
                err_console.print(f"[red]{label} failed:[/red] {exc}")
                raise typer.Exit(code=1) from exc

        return wrapper

    return decorator


def load_config_or_exit(config_path: str | None) -> "Config":
    from brainforge.config import load_config
    from brainforge.errors import ConfigError

    try:
        return load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def build_roles(config: "Config", cache_path: Path | str | None = None) -> "RoleRegistry":
    from brainforge.models.registry import ModelRegistry
    from brainforge.roles.registry import RoleRegistry

    return RoleRegistry(config, ModelRegistry(config, cache_path=cache_path))


def find_case(case_ref: str, cases_dir: Path) -> Case:
    from brainforge.dataset.case_builder import build_cases_from_path

    ref_path = Path(case_ref)
    if ref_path.is_file():
        candidates = build_cases_from_path(ref_path)
    else:
        if not cases_dir.exists():
            err_console.print(f"[red]cases directory not found:[/red] {cases_dir}")
            raise typer.Exit(code=1)
        candidates = build_cases_from_path(cases_dir)
    matches = [case for case in candidates if case.id == case_ref or case.id.endswith(case_ref)]
    if not matches:
        err_console.print(
            f"[red]case not found:[/red] {case_ref} (searched {len(candidates)} cases)"
        )
        raise typer.Exit(code=1)
    return matches[0]
