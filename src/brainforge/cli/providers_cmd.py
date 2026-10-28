import typer
from rich.table import Table

from brainforge.cli._common import (
    CONFIG_OPTION,
    clamp_errors,
    console,
    err_console,
    load_config_or_exit,
)
from brainforge.errors import BrainforgeError, ConfigError
from brainforge.providers.registry import build_provider

app = typer.Typer(help="Provider diagnostics.", no_args_is_help=True)

_PROBE_PROMPT = "Say OK"
_PROBE_MAX_TOKENS = 8


@app.command("check-logprobs")
@clamp_errors("check-logprobs")
def check_logprobs(
    provider: str | None = typer.Option(None, "--provider", help="Provider name from config."),
    model: str | None = typer.Option(None, "--model", help="Model identifier to probe."),
    topk: int = typer.Option(8, "--topk", min=1, help="Top-k alternatives requested per token."),
    all_configured: bool = typer.Option(
        False,
        "--all-configured",
        help="Probe every configured provider/model pair and print a table.",
    ),
    config: str = CONFIG_OPTION,
) -> None:
    """Probe whether a provider serves per-token logprobs (KD teacher signal)."""
    cfg = load_config_or_exit(config)
    if all_configured:
        _check_all(cfg, topk)
        return
    if not provider or not model:
        err_console.print(
            "[red]--provider and --model are required"
            " (or use --all-configured to probe every pair)[/red]"
        )
        raise typer.Exit(code=1)
    if provider not in cfg.providers:
        raise ConfigError(f"provider '{provider}' is not configured")
    built = build_provider(provider, cfg.providers[provider])
    response = _send_probe(built, model, topk)
    positions = response.logprobs or []
    if positions:
        sample = positions[0].topk[:2] or [(positions[0].token, positions[0].logprob)]
        console.print("[green]logprobs served: yes[/green]")
        console.print(f"positions: {len(positions)}")
        console.print(f"sample topk: {sample}")
    else:
        console.print("[yellow]logprobs served: no[/yellow] (no positions returned)")
    console.print(f"latency_ms: {response.latency_ms:.1f}")


def _check_all(cfg, topk: int) -> None:
    """Probe every configured (provider, model) pair; one failure is a row, not an abort."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for model_def in cfg.models.values():
        pair = (model_def.provider, model_def.model)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    pairs.sort()
    built = {}
    table = Table(title="Logprobs probe (KD teacher signal)")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Logprobs")
    table.add_column("Latency ms")
    for provider_name, model in pairs:
        try:
            if provider_name not in built:
                built[provider_name] = build_provider(provider_name, cfg.providers[provider_name])
            response = _send_probe(built[provider_name], model, topk)
            served = "yes" if response.logprobs else "no"
            latency = f"{response.latency_ms:.1f}"
        except BrainforgeError:
            served = "no"
            latency = "-"
        table.add_row(provider_name, model, served, latency)
    console.print(table)


def _send_probe(provider, model: str, topk: int):
    from brainforge.providers.base import ChatMessage, ChatRequest

    request = ChatRequest(
        messages=[ChatMessage(role="user", content=_PROBE_PROMPT)],
        max_tokens=_PROBE_MAX_TOKENS,
    )
    return provider.complete_logprobs(request, model, topk)
