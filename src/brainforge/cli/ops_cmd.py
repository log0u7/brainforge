"""`brainforge ops` sub-app: run-state and log inspection helpers."""

import json
import time
from pathlib import Path

import typer

from brainforge.cli._common import clamp_errors, console, err_console

app = typer.Typer(help="Inspect run state files and job logs.", no_args_is_help=True)

STATE_SUFFIX = ".state.json"
CURRENT_TRUNC = 60
_ENCODINGS = ("utf-8", "utf-16-le", "cp1252")
_ERROR_MARKERS = ("traceback", "failed", "error")


def _decode(data: bytes) -> str:
    """Decode with an encoding fallback chain.

    A successful utf-8 decode containing NUL bytes is treated as a miss so
    BOM-less utf-16-le text (00-terminated ASCII code units) falls through.
    """
    for encoding in _ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text.lstrip("\ufeff")
    return data.decode("utf-8", errors="replace")


def _tail_lines(path: Path, count: int) -> list[str]:
    """Last `count` non-empty lines, trying an encoding fallback chain."""
    if count <= 0:
        return []
    text = _decode(path.read_bytes())
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-count:]


def _state_stem(path: Path) -> str:
    return path.name[: -len(STATE_SUFFIX)]


@app.command("status")
@clamp_errors("ops status")
def status(
    logs: list[Path] = typer.Option([], "--logs", help="Extra log files to summarize"),
    datasets_dir: Path = typer.Option(Path("datasets"), "--datasets-dir"),
    stale_after: int = typer.Option(
        900, "--stale-after", help="Age in seconds after which a run is marked STALE"
    ),
) -> None:
    """Summarize *.state.json run states and optional log tails."""
    if not datasets_dir.is_dir():
        console.print(f"[yellow]datasets directory not found: {datasets_dir}[/yellow]")
        return
    state_files = sorted(datasets_dir.glob("*" + STATE_SUFFIX))
    if not state_files:
        console.print(f"[yellow]no state files in {datasets_dir}[/yellow]")
        return
    now = time.time()
    for path in state_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            err_console.print(f"[yellow]skipping {path.name}: {exc}[/yellow]")
            continue
        if not isinstance(data, dict):
            err_console.print(f"[yellow]skipping {path.name}: not a JSON object[/yellow]")
            continue
        done = data.get("done", 0)
        total = data.get("total", -1)
        current = str(data.get("current") or "")
        age = int(now - path.stat().st_mtime)
        stale = " [red]STALE[/red]" if age > stale_after else ""
        shown = current[:CURRENT_TRUNC] if current else "-"
        console.print(
            f"[bold]{_state_stem(path)}[/bold]: done={done}/{total}"
            f" current={shown} age={age}s{stale}",
            soft_wrap=True,
        )
    for log_path in logs:
        if not log_path.is_file():
            err_console.print(f"[yellow]log not found: {log_path}[/yellow]")
            continue
        try:
            tail = _tail_lines(log_path, 3)
            size = log_path.stat().st_size
            age = int(time.time() - log_path.stat().st_mtime)
        except OSError as exc:
            err_console.print(f"[yellow]skipping log {log_path}: {exc}[/yellow]")
            continue
        console.print(f"[bold]{log_path}[/bold]: size={size}B age={age}s")
        for line in tail:
            console.print(f"  {line}")


@app.command("log")
@clamp_errors("ops log")
def log(
    path: Path = typer.Argument(..., help="Path to the log file"),
    lines: int = typer.Option(12, "--lines", help="Number of trailing non-empty lines"),
) -> None:
    """Tail the last N non-empty lines of a log file."""
    if not path.is_file():
        console.print(f"[yellow]log not found: {path}[/yellow]")
        return
    try:
        tail = _tail_lines(path, lines)
    except OSError as exc:
        err_console.print(f"[yellow]could not read {path}: {exc}[/yellow]")
        return
    for line in tail:
        if any(marker in line.lower() for marker in _ERROR_MARKERS):
            console.print(f"[red]![/red] {line}")
        else:
            console.print(line)
