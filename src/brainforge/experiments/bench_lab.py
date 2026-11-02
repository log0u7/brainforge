"""The final benchmark harness (ADR-0021).

Reuses the pilot lab loop over an unseen challenge list, measures the
success rate over repetitions for the base and trained models, and
audits contamination: the training challenge IDs must never appear in
the eval set, or the whole benchmark is void.
"""

import json
from pathlib import Path
from typing import Any

from brainforge.errors import BrainforgeError
from brainforge.experiments.pilot import run_pilot


def _record_flag(record: dict) -> str | None:
    """The flag lives in the final assistant verdict message of the pilot
    record - the record never carries an ``outcome`` dict."""
    for message in reversed(record.get("messages") or []):
        if message.get("role") == "assistant":
            try:
                verdict = json.loads(message.get("content") or "")
            except json.JSONDecodeError:
                continue
            if isinstance(verdict, dict):
                return verdict.get("flag")
    return None


def _run_over_targets(
    targets: list[tuple],
    workdir: Path,
    provider: Any,
    model: str,
    repetitions: int,
    pilot_fn: Any,
    expected_flags: dict[str, str],
) -> dict:
    """Run one model over the challenge list ``repetitions`` times and
    tally the success rate: a success is a run whose verdict flag matches
    the challenge's expected flag (a recited training flag on the wrong
    challenge is not a success, it is contamination)."""
    stats = {"attempts": 0, "successes": 0, "per_challenge": {}}  # type: dict
    for _ in range(repetitions):
        for entry in targets:
            records = pilot_fn([entry], workdir, provider, model)
            for record in records:
                # the record's id is the launched challenge; fall back to the
                # entry the harness itself launched if the pilot dropped it
                challenge_id = record.get("id") or entry[0]
                flag = _record_flag(record)
                is_success = bool(
                    challenge_id in expected_flags and flag == expected_flags[challenge_id]
                )
                stats["attempts"] += 1
                if is_success:
                    stats["successes"] += 1
                slot = stats["per_challenge"].setdefault(
                    challenge_id, {"attempts": 0, "successes": 0}
                )
                slot["attempts"] += 1
                if is_success:
                    slot["successes"] += 1
    if stats["attempts"]:
        stats["success_rate"] = stats["successes"] / stats["attempts"]
    else:
        stats["success_rate"] = 0.0
    return stats


def run_bench_lab(
    challenges: list[tuple],
    workdir: Path,
    provider: Any,
    base_model: str,
    trained_model: str,
    repetitions: int,
    training_ids: set[str] | None = None,
    output: Path | None = None,
    pilot_fn: Any = None,
) -> dict:
    """Run the final benchmark over ``challenges`` for the two models.

    ``challenges``: (challenge_id, host:port, binary, material) tuples,
    or the same with a 5th ``expected_flag`` element (the per-target
    oracle flag the pilot verifies against). ``training_ids``: the set
    of challenge IDs used for training - the contamination audit fails
    the benchmark when one of them leaks into the eval list.
    """
    if training_ids is None:
        raise BrainforgeError(
            "bench lab refuses to run without training_ids: the "
            "contamination audit needs the training challenge set"
        )
    eval_ids = [entry[0] for entry in challenges]
    leaked = sorted(set(eval_ids) & set(training_ids))

    targets = list(challenges)
    expected_flags = {entry[0]: entry[4] for entry in targets if len(entry) > 4}
    pilot = pilot_fn or run_pilot

    results: dict = {
        "repetitions": repetitions,
        "challenges": eval_ids,
        "contamination": {
            "clean": not leaked,
            "training_ids_in_eval": leaked,
        },
    }
    for name, model in (("base", base_model), ("trained", trained_model)):
        results[name] = _run_over_targets(
            targets, workdir, provider, model, repetitions, pilot, expected_flags
        )

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows = [
            f"| {name} | {res['successes']}/{res['attempts']} | {res['success_rate']:.2f} |"
            for name, res in (("base", results["base"]), ("trained", results["trained"]))
        ]
        contamination = "clean" if results["contamination"]["clean"] else f"LEAKED: {leaked}"
        (output / "results.md").write_text(
            "# Bench lab results\n\n"
            f"repetitions: {repetitions}\n\n"
            "| model | successes/attempts | success_rate |\n"
            "|---|---|---|\n" + "\n".join(rows) + "\n\n"
            f"contamination: {contamination}\n",
            encoding="utf-8",
        )
    return results
