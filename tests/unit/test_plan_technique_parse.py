"""Regression tests for the plan-technique parse bug.

Measured: the verified-facts generator kept 0/160 candidates even
after the label fix, because ``lictor plan`` emits TEXT (``selected:
fmtstr_leak``), not a JSON dict - so ``plan.get("technique")`` was
always None and every candidate was rejected as a poison. The pilot
parses the same line with a regex; the verification flow must use the
same parse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_verified_facts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_verified_facts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN_TEXT = """mode:      handoff (prints a recipe; executes nothing)
applicable: fmtstr_write, fmtstr_leak
ranking:    fmtstr_leak > fmtstr_write
selected:   fmtstr_leak
recipe:     ronin-exploits run ... # mixin
"""


def test_plan_technique_parsed_from_stdout_text() -> None:
    """A text-mode plan output must yield its ``selected:`` line as the
    technique; a bare dict-get returned None and rejected everything."""
    module = _load_script()
    outcome = {"status": "done", "stdout": PLAN_TEXT}
    assert module.plan_technique(outcome) == "fmtstr_leak"


def test_plan_technique_none_when_selection_missing() -> None:
    module = _load_script()
    assert module.plan_technique({"status": "done", "stdout": "no plan"}) is None
    assert module.plan_technique({"status": "error", "stderr": "x"}) is None


def test_verify_candidates_keeps_a_confirming_candidate() -> None:
    """With the parse fixed, a candidate that keeps the route on the
    correct technique is KEPT (the 0/160 rejection was the bug)."""
    module = _load_script()
    workdir = Path("/tmp/nonexistent-verify-probe")

    def runner(wd, args):
        return {"status": "done", "stdout": PLAN_TEXT}

    kept, rejected = module.verify_candidates(
        baseline_facts='nx("false").\n',
        correct_technique="fmtstr_leak",
        candidates=['vuln("fmtstr").', 'fmtstr_read("true").'],
        workdir=workdir,
        runner=runner,
    )
    assert kept == ['vuln("fmtstr").', 'fmtstr_read("true").']
    assert rejected == []


def test_verify_candidates_rejects_a_poisoning_candidate() -> None:
    module = _load_script()
    workdir = Path("/tmp/nonexistent-verify-probe")
    poison_text = PLAN_TEXT.replace("fmtstr_leak", "ret2libc")

    def runner(wd, args):
        return {"status": "done", "stdout": poison_text}

    kept, rejected = module.verify_candidates(
        baseline_facts='nx("false").\n',
        correct_technique="fmtstr_leak",
        candidates=['vuln("sof").'],
        workdir=workdir,
        runner=runner,
    )
    assert kept == []
    assert rejected == ['vuln("sof"). -> ret2libc']
