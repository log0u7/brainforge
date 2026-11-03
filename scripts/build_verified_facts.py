#!/usr/bin/env python3
"""Generate decision-verified fact-suggestion SFT records (MR6).

The MR5 SFT failed because the fact-suggestion dataset carried no
execution/decision ground truth: the model learned to emit fact lines
that LOOK valid, and a domain-valid but routing-wrong fact poisons the
augure decision (measured: vuln("fmtstr") on a ret2libc target sent
the plan to fmtstr_write and the exploit died; one degenerate line
repeated ~300x poisoned a whole fact base).

This generator's ground truth is the augure DECISION itself:

  1. the teacher proposes candidate fact lines from the target's
     source material (sampled with temperature for diversity);
  2. each candidate is merged into the target's baseline facts with
     ``lictor suggest`` (the domain gate stays: garbage cannot merge);
  3. ``lictor plan`` runs on the merged fact base - the candidate is
     KEPT only when the merged selection is the target's correct
     technique (a confirming candidate) or when it moves a WRONG
     baseline onto the correct technique (a repairing candidate, the
     most valuable record there is). A candidate that breaks or fails
     to move a correct selection is REJECTED - the Étape 0 poison.

The completion is the kept candidates (possibly empty when every
candidate poisoned the route: the model must learn to emit NOTHING
rather than poison). Every record tags ``verified_by: decision`` so
provenance survives into training (AGENTS.md rule).

CONTAMINATION CONTRACT: uaf_heap is never a training target (its
material is the eval challenge ctf-lictor-uaf_heap-001).

Run:  python scripts/build_verified_facts.py \
        --provider-url https://openrouter.ai/api/v1 \
        --provider-api-key-env OPENROUTER_API_KEY \
        --model qwen/qwen3.5-9b --samples 8 \
        --out datasets/sft-verified-facts/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_facts import FACT_PROMPT, LICTOR_LAB, material_for

# The training-eligible lab originals (the 6 GRPO lab-loop targets).
TRAINING_TARGETS = ["fmtstr", "orw", "ret2csu", "ret2func", "ret2plt", "shellcode"]

# The correct augure technique per target, from the MR5 measured runs
# (the technique the lab loop's successful exploits used). A candidate
# fact is verified AGAINST this selection, not against a domain schema.
CORRECT_TECHNIQUE: dict[str, str] = {
    "fmtstr": "fmtstr_write",
    "orw": "orw",
    "ret2csu": "ret2csu",
    "ret2func": "ret2func",
    "ret2plt": "ret2plt_leak",
    "shellcode": "shellcode",
}


def verify_candidates(
    baseline_facts: str,
    correct_technique: str,
    candidates: list[str],
    workdir: Path,
    runner: Any,
) -> tuple[list[str], list[str]]:
    """Keep the candidates that confirm (or repair) the augure route.

    Returns ``(kept, rejected)``: kept candidate lines in order,
    rejected lines with their observed selection appended for the log.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    rejected: list[str] = []
    merged = baseline_facts
    for candidate in candidates:
        trial = (merged + "\n" + candidate).strip() + "\n"
        (workdir / "target.facts").write_text(trial, encoding="utf-8")
        plan = runner(workdir, ["plan", "target.facts"])
        selected = plan.get("technique")
        if selected == correct_technique:
            kept.append(candidate)
            merged = trial
        else:
            rejected.append(f"{candidate} -> {selected}")
    return kept, rejected


def build_record(target: str, material: str, completion: str, verified_by: str) -> dict:
    """One SFT record in the chat shape the trainer consumes, with the
    decision-verification provenance in the metadata."""
    return {
        "id": f"sft-verified-facts-{target}",
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": FACT_PROMPT.format(material=material)},
            {"role": "assistant", "content": completion},
        ],
        "metadata": {
            "pipeline": "verified-facts",
            "domain": "agentic",
            "training_target": target,
            "verified_by": verified_by,
            "correct_technique": CORRECT_TECHNIQUE.get(target),
            "n_facts": len([x for x in completion.splitlines() if x.strip()]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--provider-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--facts-from",
        type=Path,
        default=Path("experiments/final-bench-results/run-20261103-17ch"),
        help=(
            "Run dir holding the per-target baseline facts (the profiler"
            " facts the pilot harvested); <dir>/<target>/target.facts"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/opencode/verified-facts"))
    args = parser.parse_args()

    from brainforge.config.models import ProviderConfig
    from brainforge.experiments.pilot import ask_facts, run_lictor
    from brainforge.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        "verified-facts",
        ProviderConfig(
            type="openai",
            base_url=args.provider_url,
            api_key=os.environ.get(args.provider_api_key_env),
        ),
    )

    records: list[dict] = []
    for target in TRAINING_TARGETS:
        material = material_for(target, LICTOR_LAB)
        correct = CORRECT_TECHNIQUE[target]
        workdir = args.workdir / target
        # the baseline: the facts the pilot actually harvested for this
        # target in the measured run (profiler + leak); a candidate is
        # verified against the route THOSE facts produce, not against a
        # vacuum
        facts_path = args.facts_from / target / "target.facts"
        baseline_facts = facts_path.read_text(encoding="utf-8") if facts_path.exists() else ""
        kept_total: list[str] = []
        rejected_total: list[str] = []
        for sample in range(args.samples):
            candidates = ask_facts(provider, args.model, material)
            if not candidates:
                continue
            kept, rejected = verify_candidates(
                baseline_facts=baseline_facts,
                correct_technique=correct,
                candidates=candidates,
                workdir=workdir / f"sample-{sample}",
                runner=lambda wd, a: run_lictor(wd, a),
            )
            kept_total.extend(kept)
            rejected_total.extend(rejected)
        # dedup, keep order
        seen: set[str] = set()
        completion_lines = [line for line in kept_total if not (line in seen or seen.add(line))]
        records.append(
            build_record(
                target=target,
                material=material,
                completion="\n".join(completion_lines),
                verified_by="decision",
            )
        )
        print(
            f"[facts] {target}: kept {len(completion_lines)}, "
            f"rejected {len(rejected_total)} (poisons logged)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    print(f"[facts] wrote {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
