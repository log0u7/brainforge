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
import json
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
    # the MR6 eval twins (lictor feat/lab-eval-twins): one clean twin
    # per capability class (fmtstr, orw, ret2plt, ret2func, shellcode,
    # ret2libc, callme, srop) plus the two UAF class-transfer targets.
    # Never in any training dataset: the clean set the verdict is
    # computed on.
    "uaf-eval-001": 19996,
    "uaf-eval-002": 19997,
    "fmtstr-eval": 19998,
    "orw-eval": 19999,
    "ret2plt-eval": 19980,
    "ret2func-eval": 19981,
    "shellcode-eval": 19982,
    "ret2libc-eval": 19983,
    "callme-eval": 19984,
    "srop-eval": 19985,
    # the MR6 eval2 (harder) twins (lictor feat/lab-eval-twins): a
    # second unseen set, harder (no banner leaks, PIE where a leak
    # primitive exists). Ports 20000-20009.
    "uaf-eval2-001": 20000,
    "uaf-eval2-002": 20001,
    "orw-eval2": 20002,
    "ret2plt-eval2": 20003,
    "ret2func-eval2": 20004,
    "shellcode-eval2": 20005,
    "ret2libc-eval2": 20006,
    "callme-eval2": 20007,
    "srop-eval2": 20008,
    "fmtstr-eval2": 20009,
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


def clean_eval_ids(eval_ids: list[str], training_ids: set[str]) -> list[str]:
    """The eval IDs that survive the contamination audit.

    Delegates to ``bench_lab._contaminating_ids`` so the launch-time
    clean list and the post-bench audit share a single normalization:
    a facts-dataset training record ``sft-mr8-facts-uaf_train_a`` and
    the eval target ``ctf-lictor-uaf_train_a-001`` denote the same
    challenge, and a raw intersection alone would certify a
    contaminated eval set as clean. Input order is preserved.
    """
    from brainforge.experiments.bench_lab import _contaminating_ids

    leaked = set(_contaminating_ids(eval_ids, training_ids))
    return [eval_id for eval_id in eval_ids if eval_id not in leaked]


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
        "--mode",
        type=str,
        default="full",
        choices=["full", "model"],
        help="full: model + augure + lictor (production system); model: model alone, "
        "no augure/lictor - the model must solve from source code alone",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("experiments/final-bench"),
        help="Pilot workdir and results output directory",
    )
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

    from brainforge.experiments.bench_lab import run_bench_lab

    results = run_bench_lab(
        challenges=challenges,
        workdir=args.workdir,
        provider=SplitProvider(providers=providers),
        base_model=args.base_model,
        trained_model=args.trained_model,
        repetitions=args.repetitions,
        training_ids=training_ids,
        output=args.workdir,
        mode=args.mode,
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
