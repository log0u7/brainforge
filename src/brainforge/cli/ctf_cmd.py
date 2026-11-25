"""CTF domain commands: flag ground truth stamping and trajectory ingest.

These commands live here since ADR-0022 moved the flag ground-truth
logic into the ctf pack: ``dataset enrich-flags`` and
``dataset ingest-trajectories`` are deprecated aliases delegating here,
so ``ctf enrich-flags`` and ``ctf ingest-trajectories`` are the stable
invocations going forward.
"""

import json
from pathlib import Path

import typer

from brainforge.cli._common import console, err_console
from brainforge.errors import BrainforgeError

app = typer.Typer(help="CTF domain: flag ground truth and trajectory ingest.", no_args_is_help=True)


def _read_cases(cases_dir: Path) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(cases_dir.glob("*.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            err_console.print(f"[red]invalid case file {path}:[/red] {exc}")
            raise BrainforgeError(str(exc)) from exc
        if isinstance(case, dict):
            cases.append(case)
    return cases


@app.command("enrich-flags")
def enrich_flags_cmd(
    dataset: Path = typer.Argument(..., help="Path to dataset.jsonl (rewritten in place)"),
    cases_dir: Path = typer.Option(
        Path("examples/cases"), "--cases-dir", help="Directory of case JSON files"
    ),
) -> None:
    """Stamp verified-tier ground-truth flags into a dataset's answers.

    Verified-tier cases carry ``metadata.expected_flag`` (the real flag
    string the gate checks against); the pipeline verifies it but stores
    ``flag: null`` in the record's assistant JSON. This copies the string
    into the record and tags ``flag_verified``, which makes flag_accuracy
    computable at task-eval time.
    """
    from brainforge.cli.dataset_cmd import _read_jsonl_generic
    from brainforge.domains.ctf.flag_ground_truth import enrich_flags

    if not dataset.is_file():
        err_console.print(f"[red]file not found:[/red] {dataset}")
        raise typer.Exit(code=1)
    cases = _read_cases(cases_dir)
    records = _read_jsonl_generic(dataset)
    enriched = enrich_flags(records, cases)
    stamped = sum(1 for r, e in zip(records, enriched, strict=True) if r is not e)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in enriched)
    temp = dataset.with_suffix(dataset.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(dataset)
    console.print(f"[green]flags stamped[/green]: {stamped}/{len(records)} records")


@app.command("ingest-trajectories")
def ingest_trajectories_cmd(
    trajectories: Path = typer.Argument(..., help="Path to the pilot's trajectories.jsonl"),
    dataset_dir: Path = typer.Argument(
        ..., help="GRPO dataset dir (train.jsonl + grpo_flags.jsonl)"
    ),
    cases_dir: Path = typer.Option(
        Path("examples/cases"), "--cases-dir", help="Directory of case JSON files"
    ),
    no_compact: bool = typer.Option(
        False,
        "--no-compact",
        help="Store raw tool observations instead of excerpted ones",
    ),
) -> None:
    """The pilot's execution-verified trajectories become the GRPO
    dataset: train.jsonl merged by id (a re-ingested challenge replaces
    its stale row) and the grpo_flags.jsonl sidecar carrying the flag
    the SERVICE read - the execution is the ground truth. Tool
    observations are excerpted at ingest (compact_tool_observations);
    pass --no-compact to store raw dumps."""
    from brainforge.cli.dataset_cmd import _read_jsonl_generic
    from brainforge.domains.ctf.flag_ground_truth import ingest_trajectories

    if not trajectories.is_file():
        err_console.print(f"[red]file not found:[/red] {trajectories}")
        raise typer.Exit(code=1)
    cases = _read_cases(cases_dir)
    records = _read_jsonl_generic(trajectories)
    rows, flags = ingest_trajectories(records, cases, compact=not no_compact)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "train.jsonl"
    merged: dict[str, dict] = {}
    if train_path.is_file():
        for line in train_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id"):
                merged[row["id"]] = row
    fresh = sum(1 for r in rows if r.get("id") in merged)
    for row in rows:
        if row.get("id"):
            merged[row["id"]] = row
    train_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged.values()),
        encoding="utf-8",
    )

    flags_path = dataset_dir / "grpo_flags.jsonl"
    known: dict[str, dict] = {}
    if flags_path.is_file():
        for line in flags_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("id"):
                known[entry["id"]] = entry
    for entry in flags:
        known[entry["id"]] = entry
    flags_path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in known.values()),
        encoding="utf-8",
    )
    console.print(
        f"[green]trajectories ingested[/green]: {len(rows)} rows "
        f"({fresh} replaced), {len(flags)} flags -> {dataset_dir}"
    )
