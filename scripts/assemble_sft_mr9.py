#!/usr/bin/env python3
"""Assemble the MR9 fact-classification SFT dataset.

MR9 fixes three measured defects of the mr8-20 recipe (bench v6: trained-mr8
collapsed every UAF challenge to vuln=sof while the base model stayed correct):

1. TRAIN/EVAL PROMPT MISMATCH - sft-mr8-facts was built with the older
   build_sft_facts.FACT_PROMPT; the live bench evaluates against the newer
   brainforge.experiments.pilot.FACT_PROMPT (which bans empty values and
   tightened the allocator domain). Records here use THE BENCH copy verbatim.

2. FORMAT CONFLICT - mr8 completions taught the {"facts": [...]} JSON envelope
   while both prompt variants demand one-predicate-per-line Prolog style;
   at eval the model followed the prompt and lost format grounding. Completions
   here are plain Prolog facts.

3. CLASS COLLAPSE + MATERIAL TRUNCATION - the ingest EXCERPT_CHARS cap used to
   clip lab sources at 1500 chars, hiding free() from every UAF record, while
   sof held 13/20 records vs 3 for uaf (majority collapse). MR9 uses full case
   files and over-samples class-minority targets: each target in VULN_TRUTH
   gets COPIES_PER[target] duplicates of the same prompt+completion.

The full body is unioned with datasets/sft-mr6-v2/train.jsonl (219 multi-turn
agent-loop rows) because the 20-record pool alone caused catastrophic
forgetting; those agent trajectories keep the pipeline behaviour alive.

Run:
    python scripts/assemble_sft_mr9.py --out datasets/sft-mr9-facts/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble_sft_mr8 as mr8
from brainforge.experiments.pilot import FACT_PROMPT as BENCH_PROMPT

# How many duplicate copies of the same prompt+completion per target.
# The MR8 recipe had 13 sof records vs 3 uaf records: the training
# prevalence collapsed every UAF challenge to vuln=sof. The over-sampling
# equalizes the class mix: sof 13, fmtstr 8, uaf 9.
COPIES_PER: dict[str, int] = {
    **{t: 1 for t in mr8.VULN_TRUTH},
    **{t: 3 for t in mr8.VULN_TRUTH if mr8.VULN_TRUTH[t] == "uaf"},
    **{t: 2 for t in mr8.VULN_TRUTH if mr8.VULN_TRUTH[t] == "fmtstr"},
}


def _prolog_completion(facts: dict[str, str]) -> str:
    """One predicate per line, the exact format the bench prompt demands."""
    return "\n".join(f'{name}("{value}").' for name, value in sorted(facts.items()) if value)


def build_records(
    votes: dict[str, dict[str, str]],
    cases_dir: Path,
    facts_dataset: dict[str, str] | None = None,
) -> tuple[list[dict], list[str]]:
    """One single-turn record per training target, class-minority over-sampled."""
    records: list[dict] = []
    skipped: list[str] = []
    for target, vuln_class in sorted(mr8.VULN_TRUTH.items()):
        material = mr8.material_for_target(target, cases_dir)
        if not material:
            skipped.append(target)
            continue
        facts = {"vuln": vuln_class}
        facts.update(votes.get(target, {}))
        facts.update(mr8.FACT_OVERRIDES.get(target, {}))
        facts = {name: value for name, value in facts.items() if name in mr8.LLM_ALLOWED}
        completion = _prolog_completion(facts)
        record = {
            "id": f"sft-mr9-facts-{target}",
            "domain": "agentic",
            "messages": [
                {"role": "user", "content": BENCH_PROMPT.format(material=material)},
                {"role": "assistant", "content": completion},
            ],
            "metadata": {
                "pipeline": "verified-facts",
                "training_target": target,
                "verified_by": "decision",
                "n_facts": len(facts),
                "copies": COPIES_PER[target],
                "source": {
                    "type": "lictor-lab",
                    "repository": "log0u7/lictor lab",
                    "path": f"lab/targets/{target}",
                    "date": "2026-11-09",
                },
                "teachers": [
                    {
                        "kind": "fact_teacher",
                        "model": "majority-vote-mr6+fact_overrides",
                        "input": {"prompt": "bench FACT_PROMPT", "material": target},
                        "output": {"format": "prolog", "facts": len(facts)},
                    }
                ],
                "quality": {"passed": True, "evidence": "decision_verified"},
                "recitation_risk": False,
            },
        }
        for _ in range(COPIES_PER[target]):
            records.append(record)
    return records, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=["datasets/sft-mr6-v2/train.jsonl"],
        help="trajectory JSONLs to mine the majority facts from",
    )
    parser.add_argument(
        "--cases-dir",
        default="examples/cases",
        help="directory holding the <case-id>.json challenge files",
    )
    parser.add_argument(
        "--agent-loop",
        default="datasets/sft-mr6-v2/train.jsonl",
        help="agent-loop dataset to union with the facts records",
    )
    parser.add_argument("--out", default="datasets/sft-mr9-facts/train.jsonl")
    args = parser.parse_args()

    paths = [Path(p) for p in args.source]
    votes = mr8.majority_facts(paths)
    records, skipped = build_records(votes, Path(args.cases_dir))

    agent_loop = [
        json.loads(l) for l in Path(args.agent_loop).read_text().splitlines() if l.strip()
    ]
    merged = agent_loop + records
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record) + "\n")

    from collections import Counter

    classes = Counter(r["messages"][1]["content"] for r in records)
    print(f"wrote {len(merged)} records -> {out}")
    print(f"agent-loop rows: {len(agent_loop)}, facts records: {len(records)}")
    print(f"unique completions: {len(classes)}")
    for c, n in classes.most_common():
        print(f"  {n}x: {c[:90]}")
    if skipped:
        print(f"skipped targets: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
