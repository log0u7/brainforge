"""Tests for the MR6 target-list corrections.

Three measured facts drive this fix:

1. ``ret2csu`` is an Emporium multi-file target (the binary needs
   ``libret2csu.so``); the pilot's single-binary acquire 404s on it
   and it produced ZERO records in every measured run (the 17-ch
   run dir is empty). It is not a trainable target in the current
   loop: excluded, with the reason documented.
2. ``CORRECT_TECHNIQUE`` labels must be the routes the measured
   WINNING runs used (trails with ``success: true``), not guessed
   names: fmtstr winners ran ``fmtstr_leak``, ret2plt winners ran
   ``ret2plt``, not the ``_write``/``_leak`` guesses.
3. The verified-facts baseline must be a FRESH harvest (acquire +
   harvest against the live lab), not a stale run-dir facts file
   (the on-disk file is the last rep's state and diverges from the
   state that produced the winning plan).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ret2csu_excluded_from_every_training_surface() -> None:
    rollouts = _load("run_rollouts.py")
    miner = _load("build_sft_trails.py")
    facts = _load("build_verified_facts.py")
    assert "ret2csu" not in rollouts.TRAINING_ELIGIBLE_BINARIES
    assert "ret2csu" not in miner.TRAINING_ELIGIBLE
    assert "ret2csu" not in facts.TRAINING_TARGETS
    # the exclusion is documented, not silent: the source files carry
    # the Emporium rationale next to the allowlist
    for path in (
        REPO_ROOT / "scripts" / "run_rollouts.py",
        REPO_ROOT / "scripts" / "build_sft_trails.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "Emporium" in text and "ret2csu" in text, path


def test_mr8_counterexample_twins_verified_against_measured_routes() -> None:
    """The MR8 counter-examples train through the decision-verified
    gate. Each twin's baseline route is ret2libc (measured 2026-11-07:
    fresh acquire + harvest + plan on the live lab, no LLM facts -
    srop_train is applicable [ret2libc, ret2vdso] and selects
    ret2libc). Their material shows printf/snprintf with NO format
    string bug, so a vuln("fmtstr") candidate poisons the route and
    MUST be rejected: the completion teaches emitting nothing on
    printf-family material. ret2csu stays excluded (the Emporium
    rationale stands: multi-file target, the pilot's acquire 404s)."""
    facts = _load("build_verified_facts.py")
    for twin, route in (
        ("ret2libc_train", "ret2libc"),
        ("srop_train", "ret2libc"),
        ("callme_train", "ret2libc"),
    ):
        assert twin in facts.TRAINING_TARGETS
        assert facts.CORRECT_TECHNIQUE[twin] == route
    assert "ret2csu" not in facts.TRAINING_TARGETS
    assert set(facts.CORRECT_TECHNIQUE) == set(facts.TRAINING_TARGETS)


def test_mr8_twin_record_ids_carry_no_eval_marker() -> None:
    """The contamination gate drops any record whose ID carries an eval
    marker (assemble_sft_mr6.EVAL_ID_MARKERS): the counter-examples
    only reach the dataset if their generated IDs collide with
    nothing. The builders mint IDs like ``sft-facts-ret2libc_train-001``
    and ``sft-verified-facts-srop_train``."""
    twins = ("ret2libc_train", "srop_train", "callme_train")
    eval_markers = _load("assemble_sft_mr6.py").EVAL_ID_MARKERS
    for prefix in ("sft-facts", "sft-verified-facts", "sft-trails", "lab"):
        for twin in twins:
            rid = f"{prefix}-{twin}-001"
            assert not any(marker in rid for marker in eval_markers), rid


def test_correct_technique_matches_measured_winners() -> None:
    """The labels come from the successful trails (2026-11-03 runs):
    fmtstr -> fmtstr_leak, orw -> orw, ret2func -> ret2func,
    ret2plt -> ret2plt, shellcode -> shellcode."""
    facts = _load("build_verified_facts.py")
    assert facts.CORRECT_TECHNIQUE["fmtstr"] == "fmtstr_leak"
    assert facts.CORRECT_TECHNIQUE["orw"] == "orw"
    assert facts.CORRECT_TECHNIQUE["ret2func"] == "ret2func"
    assert facts.CORRECT_TECHNIQUE["ret2plt"] == "ret2plt"
    assert facts.CORRECT_TECHNIQUE["shellcode"] == "shellcode"
    assert set(facts.CORRECT_TECHNIQUE) == set(facts.TRAINING_TARGETS)


def test_verified_facts_harvests_fresh_baseline() -> None:
    """The baseline comes from acquire + harvest against the live lab
    (the same flow the pilot runs), never from a stale run-dir file."""
    facts = _load("build_verified_facts.py")
    assert hasattr(facts, "fresh_baseline")
    target = "fmtstr"
    workdir = Path("/tmp/nonexistent-probe-wd")
    calls: list[tuple] = []

    def runner(wd, args):
        calls.append((str(wd), list(args)))
        if args[0] == "harvest":
            (wd / "target.facts").write_text('nx("false").\n', encoding="utf-8")
        return {"status": "ok", "stdout": ""}

    baseline = facts.fresh_baseline(target, workdir, runner)
    kinds = [args[0] for _, args in calls]
    assert kinds[0] == "acquire"
    assert kinds[1] == "harvest"
    assert baseline == 'nx("false").\n'
