#!/usr/bin/env python3
"""Build the SFT fact-suggestion dataset (MR5 follow-up).

One record per out-of-eval lictor lab target: the pilot's exact FACT_PROMPT
with the target's service source as material, the completion is the
ground-truth fact lines the augure decision engine needs from the LLM.
Empty completions are valid records (profiler facts suffice): the model
must learn to emit NOTHING when the ELF already shows everything.

Training set contract (memory dc8e61c1): the training targets must lie
OUTSIDE the 11-challenge clean eval set. The lab original ``uaf_heap``
is EXCLUDED: its material is the exact eval challenge
``ctf-lictor-uaf_heap-001`` (lab/targets/uaf_heap), so training on it
would contaminate the eval (the ID audit cannot catch a material match
under a different ID prefix - the lesson from the 2026-11-03 check).
Every -002 variant's service.c differs from its lab original, so the
remaining targets are clean; callme trains on run.sh (no eval overlap).

Run:  python scripts/build_sft_facts.py [--out datasets/sft-facts/train.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# lictor lab targets (source of the challenge material)
LICTOR_LAB = Path.home() / "projets" / "logout" / "lictor" / "lab" / "targets"

# The pilot's exact prompt: FACT_PROMPT is copied verbatim from
# src/brainforge/experiments/pilot.py so training matches inference.
FACT_PROMPT = """You are feeding an auditable exploitation decision engine.
Read the CTF challenge material below and emit ONLY the fact predicates
you can justify from the source code, one per line, Prolog style:
predicate("value").
Allowed predicates (nothing else: the profiler already emits the
binary facts and the leak addresses are measured separately):
vuln, fmtstr_read, seccomp, allocator, glibc_minor,
function_pointer_on_heap, win_function.
Value domains are strict: fmtstr_read, seccomp and
function_pointer_on_heap take ONLY true or false. vuln takes exactly
one of: "sof", "fmtstr", "uaf". allocator takes "tcache" when the
source shows tcache/heap use, nothing otherwise. glibc_minor takes a
bare integer.
Emit nothing else: no prose, no markdown, no code fences.
Only emit facts the source actually shows.

Challenge material:
{material}"""


def material_for(target: str, lab_dir: Path = LICTOR_LAB) -> str:
    """The challenge source: service.c for lab originals, the run script
    for the emporium targets (callme is a ROP Emporium binary: its
    service.c is the lib's behavior, the run script is the only source
    text the operator sees)."""
    service = lab_dir / target / "service.c"
    if service.exists():
        return service.read_text(encoding="utf-8")
    run = lab_dir / target / "run.sh"
    if run.exists():
        return run.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no material for {target} under {lab_dir}")


# Ground truth per target: the fact lines the decision engine needs
# that the profiler CANNOT show (source-knowledge only). An empty
# list is the truth for targets whose routing runs on profiler facts
# alone - emitting nothing there is correct behavior.
GROUND_TRUTH: dict[str, list[str]] = {
    # fmtstr: printf(input) is a format-string bug; fmtstr_read backs
    # the vuln("fmtstr") corroboration (augure rules require it).
    "fmtstr": ['vuln("fmtstr").', 'fmtstr_read("true").'],
    # orw: the source installs a seccomp filter (execve dead).
    "orw": ['seccomp("true").'],
    # ret2func/ret2plt/shellcode: vuln=sof + plt/nx are profiler facts;
    # the source adds nothing the ELF does not already carry.
    "ret2func": [],
    "ret2plt": [],
    "shellcode": [],
    # callme: ROP Emporium - no source text the profiler lacks.
    "callme": [],
    # uaf_heap is deliberately ABSENT: its material is the eval challenge
    # ctf-lictor-uaf_heap-001 - training on it would contaminate the eval
    # set (see the module docstring).
}

TRAINING_TARGETS = list(GROUND_TRUTH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/sft-facts/train.jsonl")
    ap.add_argument(
        "--lab-dir",
        type=Path,
        default=LICTOR_LAB,
        help="lictor lab targets dir (override for tests: the runner has no lictor checkout)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for target in TRAINING_TARGETS:
        material = material_for(target, lab_dir=args.lab_dir)
        prompt = FACT_PROMPT.format(material=material)
        completion = "\n".join(GROUND_TRUTH[target])
        records.append(
            {
                "id": f"sft-facts-{target}-001",
                "domain": "agentic",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "metadata": {
                    "pipeline": "sft-facts",
                    "target": target,
                    "n_facts": len(GROUND_TRUTH[target]),
                },
            }
        )

    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty = sum(1 for r in records if r["metadata"]["n_facts"] == 0)
    print(f"wrote {len(records)} records to {out} ({empty} empty completions)")


if __name__ == "__main__":
    main()
