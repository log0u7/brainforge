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
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_facts import FACT_PROMPT, LICTOR_LAB, material_for

# The training-eligible lab originals (the GRPO lab-loop targets that
# actually run). ``ret2csu`` is excluded: Emporium multi-file target,
# the single-binary acquire 404s on it, zero records in every run.
TRAINING_TARGETS = [
    "fmtstr",
    "orw",
    "ret2func",
    "ret2plt",
    "shellcode",
    # The MR8 counter-examples: printf-family material (printf/snprintf
    # in the source) with NO format string bug - the mr7 bench failure
    # mode. Their measured baseline route is ret2libc (2026-11-07 fresh
    # acquire+harvest+plan), so any vuln("fmtstr") candidate poisons the
    # route and the decision gate rejects it: the completion teaches
    # emitting nothing on printf-family material.
    "ret2libc_train",
    "srop_train",
    "callme_train",
]

# The correct augure technique per target: the routes the MEASURED
# WINNING runs used (the 2026-11-03 trails with success: true), not
# guessed names. A candidate fact is verified AGAINST this selection.
# The MR8 twins have no winning trail yet (never rolled out): their
# labels are the measured fresh-baseline routes (2026-11-07), the same
# measurement the gate will re-run at build time.
CORRECT_TECHNIQUE: dict[str, str] = {
    "fmtstr": "fmtstr_leak",
    "orw": "orw",
    "ret2func": "ret2func",
    "ret2plt": "ret2plt",
    "shellcode": "shellcode",
    "ret2libc_train": "ret2libc",
    "srop_train": "ret2libc",
    "callme_train": "ret2libc",
}

PACKS_DIR = "/home/logout/projets/github/logout/augure/packs"


def fresh_baseline(target: str, workdir: Path, runner: Any) -> str:
    """Harvest a fresh facts baseline for one target from the live lab.

    The stale run-dir facts file is the last rep's state and diverges
    from the state that produced the winning plan (measured: the
    shellcode file plans ret2libc while the winner ran shellcode).
    This mirrors the pilot's own flow: acquire the binary from the lab
    http server, harvest it into ``target.facts``."""
    tgt = workdir / target
    tgt.mkdir(parents=True, exist_ok=True)
    (tgt / "lictor.yml").write_text(
        f"trail: trail-{target}.jsonl\npacks: {PACKS_DIR}\n", encoding="utf-8"
    )
    runner(tgt, ["acquire", target, "--from", f"http://127.0.0.1:8000/{target}"])
    runner(tgt, ["harvest", target, "--facts", "target.facts"])
    facts_path = tgt / "target.facts"
    return facts_path.read_text(encoding="utf-8") if facts_path.exists() else ""


def plan_technique(outcome: Any) -> str | None:
    """The technique a plan outcome selected, parsed from its stdout.

    ``lictor plan`` emits TEXT (``selected:   fmtstr_leak``), not a
    JSON dict: a bare dict-get returned None and the verification flow
    rejected every candidate as a poison (measured: 0/160 kept with
    correct labels). The pilot parses the same line with a regex; this
    uses the same parse."""
    text = str(outcome.get("stdout") or "") if isinstance(outcome, dict) else ""
    match = re.search(r"selected:\s+(\S+)", text)
    return match.group(1).strip('"') if match else None


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
        selected = plan_technique(plan)
        if selected == correct_technique:
            kept.append(candidate)
            merged = trial
        else:
            rejected.append(f"{candidate} -> {selected}")
    return kept, rejected


def teacher_family(model: str) -> str:
    """The model family for provenance (the assembler's enrichment
    uses family values like 'qwen', 'deepseek'; mirror that vocabulary)."""
    head = model.split("/")[-1].lower()
    for family in ("qwen", "deepseek", "glm", "gemma", "mistral", "llama"):
        if head.startswith(family) or family in head:
            return family
    return "other"


def build_record(
    target: str,
    material: str,
    completion: str,
    verified_by: str,
    teacher_model: str | None = None,
) -> dict:
    """One SFT record in the chat shape the trainer consumes, with the
    decision-verification provenance in the metadata.

    ``teacher_model``: the model that PROPOSED the candidates (the
    builder's --model). Writing it keeps the assembler's hardcoded
    fallback (deepseek-v4-pro, a guess about past runs) from standing
    in for the real teacher - provenance must be measured, not guessed."""
    metadata: dict[str, Any] = {
        "pipeline": "verified-facts",
        "domain": "agentic",
        "training_target": target,
        "verified_by": verified_by,
        "correct_technique": CORRECT_TECHNIQUE.get(target),
        "n_facts": len([x for x in completion.splitlines() if x.strip()]),
    }
    if teacher_model:
        metadata["teachers"] = [
            {
                "role": "fact_teacher",
                "kind": "teacher",
                "provider": "openrouter",
                "model": teacher_model,
                "family": teacher_family(teacher_model),
            }
        ]
    return {
        "id": f"sft-verified-facts-{target}",
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": FACT_PROMPT.format(material=material)},
            {"role": "assistant", "content": completion},
        ],
        "metadata": metadata,
    }


def sample_facts(provider: Any, model: str, material: str) -> list[str]:
    """One teacher call, network-failure tolerant.

    The 2026-11-07 run measured the failure mode: a single hung teacher
    call (the provider's wall-clock deadline) crashed the whole build at
    target 2/8 after target 1 had produced verified records. A teacher
    sample is a hint, not ground truth - the decision gate owns the
    verdict. A failed call yields no candidates: the sample is skipped,
    the build continues."""
    from brainforge.experiments.pilot import ask_facts

    try:
        return ask_facts(provider, model, material)
    except (TimeoutError, ConnectionError) as exc:
        print(f"[facts] teacher call failed ({exc}); skipping sample")
        return []


def provider_from_args(args: argparse.Namespace) -> Any:
    """The OpenAI-compatible teacher provider with the CLI knobs."""
    from brainforge.config.models import ProviderConfig
    from brainforge.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(
        "verified-facts",
        ProviderConfig(
            type="openai",
            base_url=args.provider_url,
            api_key=os.environ.get(args.provider_api_key_env),
            timeout=args.timeout,
            max_retries=args.max_retries,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-url", required=True)
    parser.add_argument("--provider-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-call wall-clock budget in seconds (a slow reasoning teacher needs more)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="provider retries on transient failures (keep low: the gate owns quality)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/opencode/verified-facts"))
    args = parser.parse_args()

    from brainforge.experiments.pilot import run_lictor

    provider = provider_from_args(args)

    records: list[dict] = []
    for target in TRAINING_TARGETS:
        material = material_for(target, LICTOR_LAB)
        correct = CORRECT_TECHNIQUE[target]
        workdir = args.workdir / target
        # the baseline: a FRESH harvest from the live lab (the same
        # flow the pilot runs); a candidate is verified against the
        # route the fresh facts produce, not against a stale run-dir
        # file (measured: the on-disk state diverges from the winning
        # plan's state)
        baseline_facts = fresh_baseline(target, args.workdir, lambda wd, a: run_lictor(wd, a))
        kept_total: list[str] = []
        rejected_total: list[str] = []
        for sample in range(args.samples):
            candidates = sample_facts(provider, args.model, material)
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
                teacher_model=args.model,
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
