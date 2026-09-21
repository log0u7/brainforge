import json
from pathlib import Path

import typer
from rich.pretty import pretty_repr

from brainforge.cli._common import CONFIG_OPTION, console, find_case, load_config_or_exit
from brainforge.config.defaults import DEFAULT_DATA_DIR
from brainforge.config.models import Config, PipelineDef
from brainforge.domains import get_or_generic
from brainforge.models.registry import ModelRegistry
from brainforge.pipeline.engine import PipelineEngine
from brainforge.roles.registry import RoleRegistry
from brainforge.types import RoleKind

app = typer.Typer(
    help="Run individual teachers or the full ensemble on a case.", no_args_is_help=True
)


def _first_domain(cfg: Config) -> str:
    for pipeline in cfg.pipelines.values():
        return pipeline.domain
    return "generic"


def _build_engine(cfg: Config, roles: RoleRegistry, steps: list[str]) -> PipelineEngine:
    pack = get_or_generic(_first_domain(cfg))
    pipeline_def = PipelineDef(domain=_first_domain(cfg), steps=steps)
    return PipelineEngine(
        cfg,
        "teacher-cli",
        roles,
        pack,
        pipeline=pipeline_def,
    )


@app.command("run")
def run(
    role: str = typer.Option(..., "--role", help="Role to execute"),
    case_ref: str = typer.Option(..., "--case", help="Case id or path to a case file"),
    cases_dir: Path = typer.Option(DEFAULT_DATA_DIR / "raw", "--cases-dir"),
    config: str = CONFIG_OPTION,
) -> None:
    """Run a single role against one case and print its structured analysis."""
    cfg = load_config_or_exit(config)
    if role not in cfg.roles:
        console.print(f"[red]unknown role:[/red] {role}")
        raise typer.Exit(code=1)
    roles = RoleRegistry(cfg, ModelRegistry(cfg, cache_path=None))
    case = find_case(case_ref, cases_dir)
    engine = _build_engine(cfg, roles, [role])
    result = engine.run_case(case)
    data = result.teacher_results.get(role) or result.judge_result
    console.print(pretty_repr(data))


@app.command("ensemble")
def ensemble(
    case_ref: str = typer.Option(..., "--case", help="Case id or path to a case file"),
    cases_dir: Path = typer.Option(DEFAULT_DATA_DIR / "raw", "--cases-dir"),
    config: str = CONFIG_OPTION,
) -> None:
    """Run all teachers plus the judge for one case."""
    cfg = load_config_or_exit(config)
    roles = RoleRegistry(cfg, ModelRegistry(cfg, cache_path=None))
    case = find_case(case_ref, cases_dir)
    teachers = [name for name in cfg.roles if roles.resolve(name).kind != RoleKind.JUDGE]
    judges = [name for name in cfg.roles if roles.resolve(name).kind == RoleKind.JUDGE]
    engine = _build_engine(cfg, roles, teachers + judges)
    result = engine.run_case(case)
    console.print_json(json.dumps(result.judge_result or {}))
    console.print("[dim]teachers:[/dim]")
    for role_name, data in result.teacher_results.items():
        summary = data.get("summary") or data.get("reasoning") or ""
        console.print(f"  [bold]{role_name}[/bold]: {summary[:160]}")
