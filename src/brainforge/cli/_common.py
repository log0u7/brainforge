from pathlib import Path

import typer
from rich.console import Console

from brainforge.case import Case

console = Console()
err_console = Console(stderr=True)

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Path to config.json")


def load_config_or_exit(config_path: str | None):
    from brainforge.config import load_config
    from brainforge.errors import ConfigError

    try:
        return load_config(config_path)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def build_roles(config, cache_path: Path | str | None = None):
    from brainforge.models.registry import ModelRegistry
    from brainforge.roles.registry import RoleRegistry

    return RoleRegistry(config, ModelRegistry(config, cache_path=cache_path))


def build_roles_without_cache(config):
    return build_roles(config, cache_path=None)


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
