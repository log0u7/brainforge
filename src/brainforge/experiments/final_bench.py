"""The MR5 final benchmark runner (ADR-0021).

Assembles the benchmark inputs from repo artifacts and drives
``run_bench_lab``: the oracle-verified challenge cases mapped onto the
live lictor lab targets, the training challenge IDs that feed the
contamination audit, and a provider split that routes each model to
its own llama-server (llama.cpp serves one model per instance, so the
base and the trained student need separate endpoints).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: the live lictor lab listener per target binary (the lictor lab
#: start.sh exposes one socat listener per challenge; the docker lab
#: publishes them all on 127.0.0.1)
LICTOR_LAB_PORTS: dict[str, int] = {
    "fmtstr": 9996,
    "uaf_heap": 9995,
    "shellcode": 9997,
    "ret2plt": 9998,
    "ret2func": 9999,
    "orw": 9994,
    "ret2csu": 9992,
    # the MR5 scale-up targets (lictor #25): the listener per target
    # binary, the port the lab's start.sh socat lines publish
    "ret2libc-001": 9982,
    "ret2libc-002": 9983,
    "fmtstr-002": 9984,
    "ret2func-002": 9985,
    "shellcode-002": 9986,
    "uaf_heap-002": 9987,
    "orw-002": 9988,
    "callme-002": 9989,
    "ret2plt-002": 9990,
    "srop-001": 9991,
    # the MR6 training twins (lictor MR6): same uaf capability class,
    # different programs, socat listeners 19991/19992 - training
    # material for the class transfer, never eval challenges
    "uaf_train_a": 19991,
    "uaf_train_b": 19992,
}


def load_verified_challenges(manifest: Path, cases_dir: Path) -> tuple[list[tuple], list[str]]:
    """Map the oracle-verified manifest cases onto the live lab targets.

    Returns ``(challenges, skipped_ids)``: ``challenges`` are the
    ``(challenge_id, host:port, binary, material, expected_flag)``
    tuples ``run_bench_lab`` consumes; ``skipped_ids`` lists the
    verified cases with no live lab target (the ROP Emporium set ships
    no listener in the current lab) or no case file on disk.
    """
    challenges: list[tuple] = []
    skipped: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        case_id = entry["id"]
        if entry.get("tier") != "verified" or not entry.get("expected_flag"):
            # datasets without a live-flag oracle (the CTF archive sets)
            # never make the eval list: they are reported, not run
            skipped.append(case_id)
            continue
        case_path = cases_dir / f"{case_id}.json"
        if not case_path.is_file():
            skipped.append(case_id)
            continue
        case = json.loads(case_path.read_text(encoding="utf-8"))
        source_path = str((case.get("source") or {}).get("path") or "")
        binary = source_path.rsplit("/", 1)[-1]
        port = LICTOR_LAB_PORTS.get(binary)
        if port is None:
            skipped.append(case_id)
            continue
        material = (case.get("input") or {}).get("code") or ""
        expected_flag = (case.get("metadata") or {}).get("expected_flag")
        challenges.append((case_id, f"127.0.0.1:{port}", binary, material, expected_flag))
    return challenges, skipped


def load_training_ids(*paths: Path) -> set[str]:
    """Collect the challenge IDs the training datasets used.

    Every record the trained model saw is a contamination risk for the
    eval set: ``run_bench_lab`` fails the whole benchmark when one of
    these IDs leaks into the evaluation list.
    """
    ids: set[str] = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id"):
                ids.add(record["id"])
    return ids


class SplitProvider:
    """Route ``complete`` to the right llama-server by model name.

    llama.cpp serves exactly one model per instance, so the final
    benchmark needs the base and the trained student on two separate
    servers: the split routes each model name to its provider instead
    of pointing the whole benchmark at a single endpoint.
    """

    def __init__(self, providers: dict[str, Any]) -> None:
        self._providers = providers

    def complete(self, request: Any, model: str) -> Any:
        return self._providers[model].complete(request, model)


def _build_provider(name: str, base_url: str, api_key: str | None) -> Any:
    from brainforge.config.models import ProviderConfig
    from brainforge.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(
        name, ProviderConfig(type="openai", base_url=base_url, api_key=api_key)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="final-bench",
        description="Run the MR5 final benchmark (ADR-0021) over the live lictor lab.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/ctf_ingest/manifest.jsonl"),
        help="Challenge manifest (JSONL) with verification tiers",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("examples/cases"),
        help="Directory holding the <id>.json challenge case files",
    )
    parser.add_argument(
        "--training-ids",
        action="append",
        type=Path,
        default=[],
        help="JSONL file with the challenge IDs the trained model saw (repeatable)",
    )
    parser.add_argument("--base-url", required=True, help="llama-server base model URL")
    parser.add_argument("--base-api-key", default=None)
    parser.add_argument("--base-model", required=True, help="Model name for the base server")
    parser.add_argument("--trained-url", required=True, help="llama-server trained model URL")
    parser.add_argument("--trained-api-key", default=None)
    parser.add_argument("--trained-model", required=True, help="Model name for the trained server")
    parser.add_argument("--repetitions", type=int, default=1, help="Runs per (model, challenge)")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("experiments/final-bench"),
        help="Pilot workdir and results output directory",
    )
    parser.add_argument(
        "--attempt-cap", type=float, default=600.0, help="Wall-clock cap per attempt (seconds)"
    )
    parser.add_argument("--seed-base", type=int, default=7, help="Base seed for repetitions")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    args = parser.parse_args(argv)

    challenges, skipped = load_verified_challenges(args.manifest, args.cases_dir)
    if skipped:
        print(f"skipped {len(skipped)} verified case(s) with no live lab target:")
        for case_id in skipped:
            print(f"  {case_id}")
    if not challenges:
        print("no runnable challenges: aborting", file=sys.stderr)
        return 1
    print(f"eval set: {len(challenges)} oracle-verified challenge(s)")

    training_ids = load_training_ids(*args.training_ids)
    print(f"contamination audit input: {len(training_ids)} training challenge ID(s)")

    providers: dict[str, Any] = {
        args.base_model: _build_provider("final-bench-base", args.base_url, args.base_api_key),
        args.trained_model: _build_provider(
            "final-bench-trained", args.trained_url, args.trained_api_key
        ),
    }

    if not args.skip_preflight:
        from brainforge.experiments import preflight

        preflight_result = preflight.run_preflight(
            args.base_url, args.trained_url, args.base_model, args.trained_model
        )
        for c in preflight_result.checks:
            print(f"[{c['status'].upper()}] {c['id']}: {c['message']}")
        if not preflight_result.ok:
            print("PREFLIGHT FAILED: refusing to run bench", file=sys.stderr)
            return 2

    from brainforge.experiments.bench_lab import run_bench_lab

    git_sha = None
    with contextlib.suppress(Exception):
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent.parent.parent
            )
            .decode()
            .strip()
        )

    results = run_bench_lab(
        challenges=challenges,
        workdir=args.workdir,
        provider=SplitProvider(providers=providers),
        base_model=args.base_model,
        trained_model=args.trained_model,
        repetitions=args.repetitions,
        training_ids=training_ids,
        output=args.workdir,
        attempt_cap=args.attempt_cap,
        seed_base=args.seed_base,
    )
    results["protocol"] = {
        "git_sha": git_sha,
        "seed_base": args.seed_base,
        "repetitions": args.repetitions,
        "attempt_cap": args.attempt_cap,
        "base_url": args.base_url,
        "trained_url": args.trained_url,
        "base_model": args.base_model,
        "trained_model": args.trained_model,
        "manifest": str(args.manifest),
        "cases_dir": str(args.cases_dir),
    }
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
        (args.workdir / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for name in ("base", "trained"):
        stats = results[name]
        print(f"{name}: {stats['successes']}/{stats['attempts']} = {stats['success_rate']:.2f}")
    contamination = results["contamination"]
    if contamination["clean"]:
        print("contamination: clean")
        return 0
    print(f"contamination: LEAKED {contamination['training_ids_in_eval']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
