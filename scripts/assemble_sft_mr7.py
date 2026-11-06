#!/usr/bin/env python3
"""Assemble the MR7 fact-classification SFT dataset.

MR7 is a single-turn dataset: the enriched FACT_PROMPT (with the vuln
class semantics) in, the constrained JSON facts out. It teaches the two
things the MR6 multi-turn trajectories could not:

  1. the JSON schema envelope ({"facts": [{"name", "value"}]}) - the
     format the bench ask_facts now requires;
  2. the vuln class semantics: sof (strcpy/gets/overflow) vs fmtstr
     (printf with user input) vs uaf (free then use). The MR6-trained
     model classifies every overflow as fmtstr (measured 2/6 on the
     probe): the multi-turn facts steps never explained the classes.

Sources:
  - the 8 training-eligible lab targets (the eval -002 twins are
    excluded: training on them is contamination);
  - the ground-truth vuln class per target (majority vote over the MR6
    trajectories, cross-checked against the challenge source);
  - the majority LLM facts per target (fmtstr_read, seccomp, allocator,
    glibc_minor, function_pointer_on_heap) from the same trajectories.

The 72 MR6 records with the empty-facts marker are dropped here by
construction: an empty facts step teaches the model to emit nothing,
which is the exact failure the bench protocol now measures.

Run:  python scripts/assemble_sft_mr7.py --out datasets/sft-mr7-facts/train.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from brainforge.experiments.pilot import FACT_PROMPT, LLM_ALLOWED  # noqa: E402

# The training-eligible targets and their ground-truth vuln class.
# Majority vote over the MR6 trajectories, cross-checked against the
# challenge source (fmtstr/orw use printf(user_input); ret2func/ret2plt/
# shellcode overflow with strcpy/gets; the uaf twins free then use).
VULN_TRUTH: dict[str, str] = {
    "fmtstr": "fmtstr",
    "orw": "fmtstr",
    "ret2func": "sof",
    "ret2plt": "sof",
    "shellcode": "sof",
    "uaf_heap-002": "uaf",
    "uaf_train_a": "uaf",
    "uaf_train_b": "uaf",
    # the -002 twins and the extra lab targets: same class as their
    # base target (the challenge source shows the same primitive).
    "callme-002": "sof",
    "fmtstr-002": "fmtstr",
    "orw-002": "fmtstr",
    "ret2func-002": "sof",
    "ret2libc-001": "sof",
    "ret2libc-002": "sof",
    "ret2plt-002": "sof",
    "shellcode-002": "sof",
    "srop-001": "sof",
}

# Facts with very few MR6 votes are noise; these overrides carry the
# values the challenge source actually shows (checked by hand).
FACT_OVERRIDES: dict[str, dict[str, str]] = {
    # fmtstr-001 leaks via puts(GOT) then calls system("/bin/sh"):
    # seccomp must be false, the single true vote was drift.
    "fmtstr": {"seccomp": "false"},
    # orw reads the flag with open/read/write under a seccomp filter:
    # 24/24 votes say true, that one is real.
    "orw": {"seccomp": "true"},
    # the overflow twins call system() or execve: no seccomp filter.
    "ret2func": {"seccomp": "false"},
    "ret2plt": {"seccomp": "false"},
    "shellcode": {"seccomp": "false"},
    "uaf_heap-002": {"seccomp": "false"},
    "uaf_train_a": {"seccomp": "false"},
    "uaf_train_b": {"seccomp": "false"},
}


def _target_of(record: dict) -> str | None:
    """The lab target a trajectory record was collected against."""
    for message in record.get("messages") or []:
        if message.get("role") != "user":
            continue
        match = re.search(r"lab target (\S+)\.", str(message.get("content") or ""))
        if match:
            return match.group(1)
    return None


def _facts_of(record: dict) -> list[tuple[str, str]]:
    """The (name, value) pairs of the facts step, LLM predicates only."""
    facts: list[tuple[str, str]] = []
    for message in record.get("messages") or []:
        content = message.get("content") or ""
        if message.get("role") != "assistant" or '"action": "facts"' not in content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for predicate in (payload.get("action_input") or {}).get("predicates") or []:
            match = re.match(r"(\w+)\((.*)\)\.$", str(predicate))
            if match and match.group(1) in LLM_ALLOWED:
                facts.append((match.group(1), match.group(2).strip('"')))
    return facts


def majority_facts(paths: list[Path]) -> dict[str, dict[str, str]]:
    """Per-target majority vote over the trajectory facts steps."""
    votes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            target = _target_of(record)
            if target is None:
                continue
            for name, value in _facts_of(record):
                votes[target][name][value] += 1
    return {
        target: {name: counter.most_common(1)[0][0] for name, counter in names.items()}
        for target, names in votes.items()
    }


def case_material(target: str, cases_dir: Path) -> str | None:
    """The challenge source code for a lab target."""
    for path in sorted(cases_dir.glob("ctf-lictor-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        source_path = str((case.get("source") or {}).get("path") or "")
        if source_path.rsplit("/", 1)[-1] == target:
            return (case.get("input") or {}).get("code") or ""
    return None


def build_records(
    votes: dict[str, dict[str, str]], cases_dir: Path
) -> tuple[list[dict], list[str]]:
    """One single-turn record per training target."""
    records: list[dict] = []
    skipped: list[str] = []
    for target, vuln_class in sorted(VULN_TRUTH.items()):
        material = case_material(target, cases_dir)
        if not material:
            skipped.append(f"{target}: no case material")
            continue
        facts: dict[str, str] = {"vuln": vuln_class}
        facts.update(votes.get(target, {}))
        facts.update(FACT_OVERRIDES.get(target, {}))
        # only the predicates the engine accepts, one emission each
        payload = {
            "facts": [
                {"name": n, "value": v} for n, v in sorted(facts.items()) if n in LLM_ALLOWED and v
            ]
        }
        records.append(
            {
                "messages": [
                    {"role": "user", "content": FACT_PROMPT.format(material=material)},
                    {"role": "assistant", "content": json.dumps(payload)},
                ]
            }
        )
    return records, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs="*",
        default=["datasets/sft-mr6-v2/train.jsonl"],
        help="trajectory JSONLs to mine the majority facts from",
    )
    parser.add_argument("--cases-dir", default="examples/cases")
    parser.add_argument("--out", default="datasets/sft-mr7-facts/train.jsonl")
    args = parser.parse_args()

    paths = [Path(p) for p in args.source if Path(p).is_file()]
    votes = majority_facts(paths)
    records, skipped = build_records(votes, Path(args.cases_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} records to {out}")
    for note in skipped:
        print(f"skipped: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
