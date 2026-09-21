import typer
from rich.table import Table

from brainforge.cli._common import CONFIG_OPTION, console, load_config_or_exit

app = typer.Typer(help="Role inspection.", no_args_is_help=True)


@app.command("list")
def list_roles(config: str = CONFIG_OPTION) -> None:
    """List configured roles and their resolution."""
    cfg = load_config_or_exit(config)
    table = Table(title="Roles")
    table.add_column("Role")
    table.add_column("Kind")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Model ID")
    table.add_column("Temp")
    for name, role in cfg.roles.items():
        model = cfg.models[role.model]
        table.add_row(
            name,
            cfg.infer_role_kind(name).value,
            role.model,
            model.provider,
            model.model,
            str(role.temperature),
        )
    console.print(table)
