"""Tests for the decision-verified facts generator.

``scripts/build_verified_facts.py`` is not part of the ``brainforge``
package: the tests load it directly. The contract under test is the
Etape 0 lesson: a fact can be domain-valid AND still poison the
routing, so the ground truth is the augure DECISION - a candidate fact
is kept only when the merged fact base still selects the correct
technique (or moves a wrong baseline toward it).
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


def _fake_runner_factory(plans: dict[str, str]):
    """A fake lictor runner: ``suggest`` accepts anything, ``plan``
    returns the technique keyed by the merged facts file content."""

    def fake_runner(workdir: Path, args: list[str]) -> dict:
        if args[0] == "suggest":
            (workdir / "target.facts").write_text(args[2], encoding="utf-8")
            return {"status": "done"}
        if args[0] == "plan":
            # the real ``lictor plan`` emits TEXT: ``selected:`` carries
            # the technique (the JSON-dict shape never existed)
            facts = (workdir / "target.facts").read_text(encoding="utf-8")
            selected = plans.get(facts, "wrong_technique")
            return {"status": "done", "stdout": f"selected:   {selected}\n"}
        return {"status": "done"}

    return fake_runner


def test_candidate_poisoning_the_route_is_rejected(tmp_path: Path) -> None:
    """The Etape 0 measured failure: vuln("fmtstr") on a ret2libc target
    redirects the routing. The generator must reject it even though the
    domain would accept the predicate."""
    module = _load_script()
    plans = {
        'vuln("sof").\n': "ret2libc",  # baseline: correct
        'vuln("sof").\nvuln("fmtstr").\n': "fmtstr_write",  # poisoned
    }
    runner = _fake_runner_factory(plans)
    kept, rejected = module.verify_candidates(
        baseline_facts='vuln("sof").',
        correct_technique="ret2libc",
        candidates=['vuln("fmtstr").', 'seccomp("true").'],
        workdir=tmp_path,
        runner=runner,
    )
    assert kept == []
    assert rejected and "vuln" in rejected[0]


def test_candidate_confirming_the_route_is_kept(tmp_path: Path) -> None:
    module = _load_script()
    plans = {
        'vuln("sof").\n': "ret2libc",
        'vuln("sof").\nallocator("tcache").\n': "ret2libc",
    }
    runner = _fake_runner_factory(plans)
    kept, rejected = module.verify_candidates(
        baseline_facts='vuln("sof").',
        correct_technique="ret2libc",
        candidates=['allocator("tcache").'],
        workdir=tmp_path,
        runner=runner,
    )
    assert kept == ['allocator("tcache").']
    assert rejected == []


def test_wrong_baseline_candidate_must_move_toward_correct(tmp_path: Path) -> None:
    """A candidate that moves a WRONG baseline selection to the correct
    technique is the most valuable record there is: keep it."""
    module = _load_script()
    plans = {
        'vuln("sof").\n': "ret2vdso",  # baseline: wrong
        'vuln("sof").\nfmtstr_read("true").\n': "fmtstr_write",  # fixed
    }
    runner = _fake_runner_factory(plans)
    kept, _rejected = module.verify_candidates(
        baseline_facts='vuln("sof").',
        correct_technique="fmtstr_write",
        candidates=['fmtstr_read("true").'],
        workdir=tmp_path,
        runner=runner,
    )
    assert kept == ['fmtstr_read("true").']


def test_eval_targets_are_excluded() -> None:
    """uaf_heap's material is the eval challenge: never a training
    target, whatever the caller passes."""
    module = _load_script()
    assert "uaf_heap" not in module.TRAINING_TARGETS
    assert "uaf_heap" not in module.CORRECT_TECHNIQUE


def test_record_shape_carries_provenance() -> None:
    """Dataset-facing changes preserve provenance tagging (AGENTS.md):
    every record states how the completion was verified."""
    module = _load_script()
    record = module.build_record(
        target="fmtstr",
        material="int main(){}",
        completion='vuln("fmtstr").',
        verified_by="decision",
    )
    assert record["metadata"]["verified_by"] == "decision"
    assert record["metadata"]["training_target"] == "fmtstr"
    assert record["messages"][0]["role"] == "user"
