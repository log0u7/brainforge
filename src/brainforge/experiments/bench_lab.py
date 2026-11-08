"""The final benchmark harness (ADR-0021).

Reuses the pilot lab loop over an unseen challenge list, measures the
success rate over repetitions for the base and trained models, and
audits contamination: the training challenge IDs must never appear in
the eval set, or the whole benchmark is void.
"""

import json
import threading
import time
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


def _run_attempt_capped(
    pilot_fn: Any,
    entry: list[tuple],
    workdir: Path,
    provider: Any,
    model: str,
    attempt_cap: float | None,
    seed: str | None = None,
) -> tuple[list[dict], bool]:
    """Run one pilot attempt under a wall-clock cap in a daemon thread.

    A hung attempt must never block the bench: the MR6-v2 re-bench froze
    for 67 minutes on one wedged provider call. The attempt runs in a
    daemon thread (it dies with the process) and a cap expiry becomes an
    explicit TIMEOUT miss the bench moves on from."""
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["records"] = pilot_fn(entry, workdir, provider, model, seed=seed)
        except Exception as exc:  # re-raised on the caller side
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    if attempt_cap is None:
        done.wait()
    elif not done.wait(timeout=attempt_cap):
        return [], True
    if "error" in box:
        raise box["error"]
    records: list[dict] = box.get("records", [])
    return records, False


def _run_over_targets(
    targets: list[tuple],
    workdir: Path,
    provider: Any,
    model: str,
    repetitions: int,
    pilot_fn: Any,
    expected_flags: dict[str, str],
    attempt_cap: float | None = None,
    seed_base: int = 7,
) -> dict:
    """Run one model over the challenge list ``repetitions`` times and
    tally the success rate: a success is a run whose verdict flag matches
    the challenge's expected flag (a recited training flag on the wrong
    challenge is not a success, it is contamination).

    Each repetition passes its own seed to the pilot (``seed_base + rep``):
    lictor's bandit draws and the exploit build were otherwise identical
    across repetitions, so N reps replayed one trajectory instead of
    drawing N samples. The same rep index gets the same seed for both
    models, which keeps the comparison paired."""
    stats = {"attempts": 0, "successes": 0, "per_challenge": {}}  # type: dict
    for rep in range(repetitions):
        seed = str(seed_base + rep)
        for entry in targets:
            challenge_id = entry[0]
            # the progress lines: the MR6-v2 hang made a stuck challenge
            # invisible for hours (the protocol pipes the output through
            # tail -45 and the trails stayed silent). One flushed line per
            # launch and per outcome - with the elapsed time - keeps the
            # oldest in-flight run readable in the bench log itself.
            print(
                f"[bench {model} rep {rep + 1}/{repetitions}] {challenge_id} launching", flush=True
            )
            started = time.monotonic()
            records, timed_out = _run_attempt_capped(
                pilot_fn, [entry], workdir, provider, model, attempt_cap, seed=seed
            )
            elapsed = time.monotonic() - started
            flag = _record_flag(records[0]) if records else None
            outcome = "TIMEOUT" if timed_out else ("FLAG" if flag else "miss")
            print(
                f"[bench {model} rep {rep + 1}/{repetitions}] {challenge_id}: "
                f"{outcome} (elapsed {elapsed:.0f}s)",
                flush=True,
            )
            for record in records:
                # the record's id is the launched challenge; fall back to the
                # entry the harness itself launched if the pilot dropped it
                challenge_id = record.get("id") or entry[0]
                # model-only mode returns a "success" boolean directly;
                # full mode (pilot) returns a flag in the verdict message
                flag = _record_flag(record)
                record_success = record.get("success")
                if record_success is not None:
                    # model-only mode: success is already computed
                    is_success = bool(record_success)
                else:
                    # full mode: check the flag from the verdict
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


def _contaminating_ids(eval_ids: list[str], training_ids: set[str]) -> list[str]:
    """Training IDs that name the same challenge as one of ``eval_ids``.

    A raw set intersection only catches exact string matches. The GRPO
    lab-loop training records carry ``lab-<technique>-001`` ids while
    the eval manifest names the same lictor lab target
    ``ctf-lictor-<technique>-001``: left unnormalized, the audit
    certifies a contaminated eval set as clean. Every ``lab-`` record
    is therefore evaluated under the eval-namespace spellings it can
    denote; everything else compares literally.

    Facts-dataset training records (``sft-*-facts-<binary>``) name the
    target binary directly; they normalize to ``ctf-lictor-<binary>`` and
    ``ctf-lictor-<binary>-001`` so a training target that leaked into the
    eval set (e.g. ``uaf_train_a`` / ``uaf_train_b``) is caught.
    """
    eval_set = set(eval_ids)
    contaminated: set[str] = set()

    # ID prefixes used by training datasets, stripped to reveal the
    # challenge/binary name they denote. Ordered longest-first so a more
    # specific prefix wins.
    _TRAINING_PREFIXES = (
        "sft-mr8-facts-",
        "sft-verified-facts-mr8-",
        "sft-verified-facts-",
        "sft-facts-",
        "lab-",
    )

    def eval_candidates(tid: str) -> list[str]:
        """Eval-namespace spellings a single training ID can denote."""
        out: list[str] = []
        for prefix in _TRAINING_PREFIXES:
            if tid.startswith(prefix):
                suffix = tid[len(prefix) :]
                out = [f"ctf-lictor-{suffix}", f"ctf-lictor-{suffix}-001"]
                break
        return out

    for tid in training_ids:
        if tid in eval_set:
            contaminated.add(tid)
            continue
        for candidate in eval_candidates(tid):
            if candidate in eval_set:
                contaminated.add(candidate)
    return sorted(contaminated)


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
    mode: str = "full",
    attempt_cap: float | None = None,
    seed_base: int = 7,
) -> dict:
    """Run the final benchmark over ``challenges`` for the two models.

    ``challenges``: (challenge_id, host:port, binary, material) tuples,
    or the same with a 5th ``expected_flag`` element (the per-target
    oracle flag the pilot verifies against). ``training_ids``: the set
    of challenge IDs used for training - the contamination audit fails
    the benchmark when one of them leaks into the eval list.

    ``attempt_cap``: wall-clock cap (seconds) per attempt. A hung attempt
    becomes an explicit TIMEOUT miss and the bench moves on - without a
    cap a single hung attempt can freeze the whole bench.

    ``seed_base``: the repetition ``rep`` runs with seed ``seed_base + rep``
    (lictors bandit draws and the payload build). Repetitions must be
    independent samples, not replays of one trajectory.

    ``mode``: "full" = model + augure + lictor (production system);
    "model" = model alone, no augure/lictor - the model must solve from
    source code alone.
    """
    if training_ids is None:
        raise BrainforgeError(
            "bench lab refuses to run without training_ids: the "
            "contamination audit needs the training challenge set"
        )
    eval_ids = [entry[0] for entry in challenges]
    leaked = _contaminating_ids(eval_ids, training_ids)

    targets = list(challenges)
    expected_flags = {entry[0]: entry[4] for entry in targets if len(entry) > 4}
    if mode == "model":
        from brainforge.experiments.model_only_pilot import run_model_only

        pilot = pilot_fn or run_model_only
    else:
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
            targets,
            workdir,
            provider,
            model,
            repetitions,
            pilot,
            expected_flags,
            attempt_cap,
            seed_base,
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
