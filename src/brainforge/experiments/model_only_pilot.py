"""Model-only pilot: the LLM must solve the CTF challenge from source
code alone, without augure's technique selection or lictor's exploit
assembly/execution layer.

The model receives the challenge source code and must identify the
vulnerability class. It answers with the same fact format the full arm
(model + augure + lictor) uses: FACT_PROMPT in, ``{"facts": [...]}``
completion out. The model-only evaluator then reads the ``vuln`` fact
value as the predicted vulnerability class and compares it to ground
truth.

This is the *hybrid* format: a single prompt + completion consumed by
both arms of the bench. The full arm feeds the facts to augure/lictor;
the model-only arm evaluates the ``vuln`` fact directly. No technique
is measured here - technique selection is augure's job and only exists
in the full arm.

Division of labor:
- The model reads the source and emits the source-derived facts.
- augure is NOT consulted (no technique selection, no fact validation).
- lictor is NOT executed (no payload assembly, no exploit run).
- The model's response is evaluated against the ``vuln`` fact ground
  truth derived from the verified facts datasets.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any

from brainforge.experiments.pilot import FACT_PROMPT, MAX_SUGGESTED_FACTS

#: repo root, computed from this file's location (src/brainforge/experiments/)
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Minimal fallback for verified eval binaries that have no facts record
#: in any dataset. Explicit and tiny: only uaf_heap, a verified UAF target
#: the facts pipeline simply never recorded. Everything else derives from
#: datasets. ret2csu has no ground truth and stays unscored (None).
_GROUND_TRUTH_FALLBACK: dict[str, str | None] = {
    "uaf_heap": "uaf",
    "ret2csu": None,
}

#: verified facts datasets are discovered by directory name (any
#: datasets/*facts*) rather than a hard-coded glob, so the ground-truth
#: map cannot go stale across MR cycles like the old hand-written
#: _binary_to_vuln_type did.


def _facts_dataset_paths() -> list[Path]:
    """Locate every verified-facts train.jsonl under datasets/."""
    datasets = REPO_ROOT / "datasets"
    paths: list[Path] = []
    if not datasets.is_dir():
        return paths
    for child in sorted(datasets.iterdir()):
        if not child.is_dir() or "facts" not in child.name:
            continue
        train = child / "train.jsonl"
        if train.is_file():
            paths.append(train)
    return paths


_ground_truth: dict[str, str] | None = None


def _load_ground_truth(paths: list[Path] | None = None) -> dict[str, str]:
    """Build binary -> vuln class from the verified facts datasets.

    Each facts record carries ``metadata.source.path`` = ``lab/targets/<binary>``
    and a completion ``{"facts": [{"name": "vuln", "value": "sof"}]}``. The
    ``vuln`` fact value is the ground-truth class for that binary.

    ``paths`` overrides the auto-discovered dataset files (used by tests
    and by callers that stage a specific facts snapshot).
    """
    global _ground_truth
    if _ground_truth is not None and paths is None:
        return _ground_truth
    truth: dict[str, str] = {}
    scan = paths if paths is not None else _facts_dataset_paths()
    for path in scan:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = (record.get("metadata") or {}).get("source") or {}
            src_path = source.get("path") or ""
            binary = src_path.rsplit("/", 1)[-1] if src_path else ""
            if not binary:
                continue
            messages = record.get("messages") or []
            completion: str | None = None
            if len(messages) > 1:
                second = messages[1]
                if isinstance(second, dict):
                    completion = second.get("content")
                elif isinstance(second, str):
                    completion = second
            if not isinstance(completion, str):
                continue
            try:
                obj = json.loads(completion)
            except json.JSONDecodeError:
                continue
            for fact in obj.get("facts", []) or []:
                if fact.get("name") == "vuln" and fact.get("value"):
                    truth.setdefault(binary, fact["value"])
    if paths is None:
        _ground_truth = truth
    return truth


def expected_vuln(binary: str, paths: list[Path] | None = None) -> str | None:
    """Ground-truth vuln class for ``binary``, or None if unrecorded.

    Consults the facts datasets first, then the tiny documented fallback.
    ``paths`` overrides the auto-discovered dataset files (tests).
    """
    truth = _load_ground_truth(paths)
    if binary in truth:
        return truth[binary]
    return _GROUND_TRUTH_FALLBACK.get(binary)


#: Prolog fact lines, tolerant of the shapes the models actually emit
#: (run-20261108-mr8-model-v3 audit): quoted or bare atom values, the
#: final period optional, trailing whitespace tolerated by stripping.
#: The model-only metric reads only the ``vuln`` fact, so value-domain
#: checks stay in the evaluator - a wrong value is a scored failure,
#: not parser noise to drop.
_PROLOG_RE = re.compile(r"^([a-z_][a-z0-9_]*)\(\s*(.*?)\s*\)\.?$")

#: GRPO lab-loop wrapper heads (run-20261030 lesson: semantics correct,
#: shape is a wrapper of the fact grammar). Both ``predicate(...)`` and
#: ``value(...)`` heads appear in live probes; first argument is the
#: fact name, second the value. The trailing Prolog period is tolerated
#: (``predicate("win_function").``) so the wrapper branch fires before
#: _PROLOG_RE and a single-argument wrapper stays a dropped degenerate
#: marker instead of leaking a {predicate, ...} fact.
_WRAPPER_HEAD_RE = re.compile(r"^(?:predicate|value)\((.*)\)\.?$")


def _extract_facts(text: str) -> list[dict[str, str]]:
    """Parse facts from the model's response.

    Accepts either the JSON facts shape the student was trained on
    (``{"facts": [{"name": ..., "value": ...}]}``) or Prolog lines in
    every shape live models emit: FACT_PROMPT's canonical ``vuln("sof").``,
    its punctation-free variant ``vuln(sof)`` (trained-mr8 skips the final
    period), and the GRPO wrapper ``value("vuln", "uaf")`` / ``predicate(...)``
    wrapping two arguments (base emits both variants).

    Returns deduplicated dicts capped at MAX_SUGGESTED_FACTS with keys
    ``name`` and ``value``; empty values are dropped (the prompt bans them).
    """
    facts: list[dict[str, str]] = []
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
        for fact in obj["facts"]:
            if isinstance(fact, dict) and fact.get("name") and fact.get("value") is not None:
                facts.append({"name": str(fact["name"]), "value": str(fact["value"])})
        return facts

    seen: set[tuple[str, str]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        wrapper_match = _WRAPPER_HEAD_RE.match(line)
        prolog_match = _PROLOG_RE.match(line)
        if wrapper_match:
            # Two-argument wrapper: predicate("vuln", "uaf") or
            # value(function_pointer_on_heap, true). csv.reader splits
            # the comma while respecting quoted values in every
            # combination the probes show (both args quoted, only one
            # quoted, neither quoted).
            try:
                cols = next(csv.reader([wrapper_match.group(1)]))
            except (csv.Error, StopIteration):
                continue
            if len(cols) != 2:
                continue
            name, value = cols[0].strip(), cols[1].strip()
        elif prolog_match:
            name = prolog_match.group(1)
            value = prolog_match.group(2).strip()
        else:
            continue
        name, value = name.strip('"'), value.strip('"')
        if not name or not value:
            continue
        if (name, value) in seen:
            continue
        seen.add((name, value))
        facts.append({"name": name, "value": value})
    return facts[:MAX_SUGGESTED_FACTS]


def _evaluate_response(
    llm_text: str,
    target_binary: str,
    paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Evaluate the model's response against ground truth.

    Returns a dict with:
      - valid: whether the response parsed as facts
      - correct_vuln_type: whether the vuln fact matches ground truth
      - vuln_type: the predicted vuln class (or None)
      - expected_vuln: the ground-truth class (or None)
      - facts: the parsed facts
    """
    facts = _extract_facts(llm_text)
    valid = bool(facts)

    predicted: str | None = None
    for fact in facts:
        if fact.get("name") == "vuln":
            predicted = fact.get("value")
            break

    expected = expected_vuln(target_binary, paths=paths)
    # No ground truth recorded => cannot verify, count as a pass so the
    # binary is not a hard failure. A recorded class is a real check:
    # the model must emit the matching vuln fact.
    correct_vuln = expected is None or predicted == expected

    return {
        "valid": valid,
        "correct_vuln_type": correct_vuln,
        "vuln_type": predicted,
        "expected_vuln": expected,
        "facts": facts,
        "success": correct_vuln,
    }


def run_model_only(
    targets: list[tuple],
    workdir: Path,
    provider: Any,
    model: str,
    seed: str = "7",
) -> list[dict]:
    """Run one target: the model analyzes the source and emits facts.
    No augure, no lictor - just the model and the source code.

    Each target entry: (challenge_id, host:port, binary, material,
    expected_flag)

    Returns a list of records with the model's analysis and evaluation.
    """
    records: list[dict] = []

    for entry in targets:
        challenge_id = entry[0]
        binary = entry[2]
        material = entry[3]

        # Ask the model to emit the source-derived facts (FACT_PROMPT,
        # the same prompt the full arm uses - hybrid format).
        prompt = FACT_PROMPT.format(material=material)

        from brainforge.providers.base import ChatMessage, ChatRequest

        response = provider.complete(
            ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.2,
            ),
            model,
        )

        llm_text = response.content or ""

        # Evaluate the response against ground truth
        eval_result = _evaluate_response(llm_text, binary)

        record = {
            "id": challenge_id,
            "binary": binary,
            "model": model,
            "valid": eval_result["valid"],
            "success": eval_result["success"],
            "vuln_type": eval_result.get("vuln_type"),
            "expected_vuln": eval_result.get("expected_vuln"),
            "correct_vuln_type": eval_result.get("correct_vuln_type"),
            "facts": eval_result.get("facts"),
            "reason": "" if eval_result["valid"] else "model did not return valid facts",
            "raw_response": llm_text[:500],
        }
        records.append(record)

    return records


__all__ = [
    "FACT_PROMPT",
    "_evaluate_response",
    "_extract_facts",
    "expected_vuln",
    "run_model_only",
]
