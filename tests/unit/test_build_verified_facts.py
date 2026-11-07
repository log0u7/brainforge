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


def test_gate_plans_with_the_production_packs_config(tmp_path: Path) -> None:
    """The 2026-11-07 mr8 build: the gate's ``lictor plan`` trials ran
    without a lictor.yml, so the augure packs never loaded and pack-only
    techniques (orw) were unreachable - every orw candidate was rejected
    and the orw record stayed empty for a config reason, not a poison
    reason. The gate must plan against the same rule universe as
    production."""
    module = _load_script()
    runner = _fake_runner_factory({'seccomp("true").\n': "orw"})
    module.verify_candidates(
        baseline_facts="",
        correct_technique="orw",
        candidates=['seccomp("true").'],
        workdir=tmp_path,
        runner=runner,
    )
    yml = (tmp_path / "lictor.yml").read_text(encoding="utf-8")
    assert f"packs: {module.PACKS_DIR}" in yml


def test_route_preserving_contradiction_is_rejected(tmp_path: Path) -> None:
    """The mr8 fmtstr completion carried fmtstr_read("true") AND
    fmtstr_read("false"): the gate kept a candidate that contradicts the
    merged fact base as long as the route survived, so the student would
    learn to emit both values. A contradiction that preserves an already
    correct route is a poison, not a suggestion."""
    module = _load_script()
    plans = {
        'vuln("fmtstr").\nfmtstr_read("true").\n': "fmtstr_leak",
        'vuln("fmtstr").\nfmtstr_read("true").\nfmtstr_read("false").\n': "fmtstr_leak",
    }
    runner = _fake_runner_factory(plans)
    kept, rejected = module.verify_candidates(
        baseline_facts='vuln("fmtstr").\nfmtstr_read("true").',
        correct_technique="fmtstr_leak",
        candidates=['fmtstr_read("false").'],
        workdir=tmp_path,
        runner=runner,
    )
    assert kept == []
    assert rejected and rejected[0].startswith('fmtstr_read("false")')


def test_conflicting_repair_of_a_wrong_baseline_is_kept(tmp_path: Path) -> None:
    """A candidate contradicting the merged facts is still the most
    valuable record there is when it MOVES a wrong baseline onto the
    correct technique."""
    module = _load_script()
    plans = {
        'seccomp("false").\n': "ret2vdso",  # baseline: wrong
        'seccomp("false").\nseccomp("true").\n': "orw",  # repaired
    }
    runner = _fake_runner_factory(plans)
    kept, _rejected = module.verify_candidates(
        baseline_facts='seccomp("false").',
        correct_technique="orw",
        candidates=['seccomp("true").'],
        workdir=tmp_path,
        runner=runner,
    )
    assert kept == ['seccomp("true").']


def test_eval_targets_are_excluded() -> None:
    """uaf_heap's material is the eval challenge: never a training
    target, whatever the caller passes."""
    module = _load_script()
    assert "uaf_heap" not in module.TRAINING_TARGETS
    assert "uaf_heap" not in module.CORRECT_TECHNIQUE


def test_record_carries_the_real_teacher_provenance() -> None:
    """The builder knows its teacher (the --model that proposed the
    candidates) and must stamp it: the assembler's fallback teacher
    (deepseek-v4-pro, hardcoded) is provenance-guessing for any run
    that used another model. MR8 runs the teacher z-ai/glm-5.3-flash;
    the dataset must say so."""
    module = _load_script()
    record = module.build_record(
        target="fmtstr",
        material="int main(){}",
        completion='vuln("fmtstr").',
        verified_by="decision",
        teacher_model="z-ai/glm-5.3-flash",
    )
    assert record["metadata"]["teachers"] == [
        {
            "role": "fact_teacher",
            "kind": "teacher",
            "provider": "openrouter",
            "model": "z-ai/glm-5.3-flash",
            "family": "glm",
        }
    ]


def test_record_without_teacher_keeps_the_legacy_shape() -> None:
    """No teacher_model argument: no teachers key written. The assembler
    then applies its own provenance rule (records without teachers get
    the legacy fallback) - the builder must not guess."""
    module = _load_script()
    record = module.build_record(
        target="fmtstr",
        material="int main(){}",
        completion='vuln("fmtstr").',
        verified_by="decision",
    )
    assert "teachers" not in record["metadata"]


def test_teacher_failure_skips_the_sample() -> None:
    """The 2026-11-07 crash: one hung teacher call (provider wall-clock
    deadline) killed the whole build at target 2/8. A sample is a hint,
    not ground truth - a failed teacher call must skip the sample and
    let the build continue."""
    module = _load_script()
    from brainforge.providers.base import ChatRequest

    class DeadTeacher:
        def complete(self, request: ChatRequest, model: str):
            raise TimeoutError("provider call exceeded the wall-clock deadline")

    provider = DeadTeacher()
    assert module.sample_facts(provider, "z-ai/glm-5.3-flash", "int main(){}") == []


def test_provider_from_args_carries_the_timeout_knobs() -> None:
    """--timeout / --max-retries reach the ProviderConfig: a slow
    reasoning teacher on OpenRouter needs more than the 120s default,
    and the retry policy is the operator's choice."""
    module = _load_script()
    args = module.argparse.Namespace(
        provider_url="https://openrouter.ai/api/v1",
        provider_api_key_env="NO_SUCH_KEY",
        timeout=300.0,
        max_retries=2,
    )
    provider = module.provider_from_args(args)
    assert provider.config.timeout == 300.0
    assert provider.config.max_retries == 2


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
