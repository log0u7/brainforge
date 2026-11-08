"""Regression tests for the model-only pilot metric fix (hybrid format).

Covers the six metric bugs from the model-only bench audit:

1. srop-001 was impossible to pass (expected ``srop`` but the prompt's
   vocabulary excluded it) -> now derived from facts datasets as ``sof``.
2. callme-002 was judged against ``ret2func`` but its vuln class is
   ``sof`` -> dataset ground truth.
3. uaf_train_a / uaf_train_b were free passes (absent from the mapping) ->
   now real checks (``uaf``) and flagged as contaminated training targets.
4. uaf_heap-001 is the one real UAF check -> preserved via fallback.
5. ``expected_technique`` was a no-op -> technique is no longer measured
   in model-only mode (only the ``vuln`` fact).
6. The ground-truth map is derived from the verified facts datasets
   instead of a hand-maintained mapping that goes stale each MR cycle.

The model-only arm now shares the FACT_PROMPT + facts-JSON completion
format with the full arm (model + augure + lictor): the hybrid format.
"""

import json
from pathlib import Path

from brainforge.experiments import model_only_pilot as m


def _write_facts_record(path: Path, binary: str, vuln: str) -> None:
    """Append one facts-dataset record (the shape the fact pipeline writes)."""
    record = {
        "id": f"sft-mr8-facts-{binary}",
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": "FACT_PROMPT ... Challenge material:\nint vuln(){}"},
            {
                "role": "assistant",
                "content": json.dumps({"facts": [{"name": "vuln", "value": vuln}]}),
            },
        ],
        "metadata": {
            "pipeline": "verified-facts",
            "training_target": binary,
            "source": {
                "type": "lictor-lab",
                "path": f"lab/targets/{binary}",
            },
        },
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _stage_datasets(tmp_path: Path, binaries: dict[str, str]) -> list[Path]:
    """Write a tmp sft-mr8-facts train.jsonl and return its path."""
    datasets = tmp_path / "datasets" / "sft-mr8-facts"
    datasets.mkdir(parents=True)
    train = datasets / "train.jsonl"
    for binary, vuln in binaries.items():
        _write_facts_record(train, binary, vuln)
    return [train]


class TestGroundTruthDerivation:
    def test_derives_vuln_class_from_facts_datasets(self, tmp_path: Path) -> None:
        paths = _stage_datasets(
            tmp_path,
            {
                "callme-002": "sof",
                "srop-001": "sof",
                "fmtstr-002": "fmtstr",
                "uaf_train_a": "uaf",
                "uaf_train_b": "uaf",
                "uaf_heap-002": "uaf",
            },
        )
        assert m.expected_vuln("callme-002", paths=paths) == "sof"
        assert m.expected_vuln("srop-001", paths=paths) == "sof"
        assert m.expected_vuln("fmtstr-002", paths=paths) == "fmtstr"
        assert m.expected_vuln("uaf_train_a", paths=paths) == "uaf"
        assert m.expected_vuln("uaf_train_b", paths=paths) == "uaf"
        assert m.expected_vuln("uaf_heap-002", paths=paths) == "uaf"

    def test_unknown_binary_has_no_ground_truth(self, tmp_path: Path) -> None:
        paths = _stage_datasets(tmp_path, {"callme-002": "sof"})
        assert m.expected_vuln("does-not-exist", paths=paths) is None

    def test_uaf_heap_preserved_by_document_fallback(self) -> None:
        # uaf_heap has no facts record in any dataset; the fallback keeps
        # it a real UAF check instead of a free pass.
        assert m.expected_vuln("uaf_heap") == "uaf"

    def test_ret2csu_is_unscored_when_no_record_exists(self) -> None:
        # ret2csu has no facts record and no fallback -> None (honest,
        # not a fabricated class).
        assert m.expected_vuln("ret2csu") is None


class TestFactExtraction:
    def test_parses_json_facts_completion(self) -> None:
        facts = m._extract_facts(
            '{"facts": [{"name": "vuln", "value": "uaf"}, {"name": "seccomp", "value": "false"}]}'
        )
        assert facts == [
            {"name": "vuln", "value": "uaf"},
            {"name": "seccomp", "value": "false"},
        ]

    def test_parses_prolog_lines_fallback(self) -> None:
        facts = m._extract_facts('vuln("sof").\nseccomp("false").\n')
        assert {"name": "vuln", "value": "sof"} in facts
        assert {"name": "seccomp", "value": "false"} in facts

    def test_returns_empty_for_unparseable(self) -> None:
        assert m._extract_facts("no facts here") == []
        assert m._extract_facts("") == []


class TestFactExtractionTolerance:
    """Regression tests for the run-20261108-mr8-model-v3-10rep audit.

    The probe of the live models (base-qwen3-1.7b on 9932, trained-mr8
    on 9933) proved the models emit the facts in several shapes the old
    parser rejected for punctuation, so the metric scored every
    well-formed answer as a miss:

    - trained-mr8 emits bare atoms without the final Prolog period:
      ``vuln(sof)`` (the trained model's answer is CORRECT and the
      metric scored 1/120).
    - trained-mr8 also emits the trailing-two-space chat template
      artifact: ``vuln(sof)  \nfmtstr_read(true)  \\n...``.
    - base emits the quoted shape without the period too:
      ``vuln("fmtstr")``.
    - base emits the GRPO lab-loop wrapper under the ``value`` name
      (the lesson from run-20261030 calls this shape ``predicate``):
      ``value("vuln", "uaf")``.
    - the trained model's refusal prose (``% The source code does not
      show...``) must stay empty: no facts, no accidental match.

    The metric measures the classification, not the punctuation: the
    Prolog fallback must accept a missing period, a bare atom value and
    the wrapper shapes, while a wrong vuln value (``vuln(true)``,
    ``vuln("fmtstr")`` on a uaf target) still counts as a failure.
    """

    def test_parses_prolog_line_without_period_quoted(self) -> None:
        facts = m._extract_facts('vuln("sof")')
        assert {"name": "vuln", "value": "sof"} in facts

    def test_parses_prolog_line_without_period_bare_atom(self) -> None:
        # trained-mr8's exact shape: vuln(sof)
        facts = m._extract_facts("vuln(sof)")
        assert {"name": "vuln", "value": "sof"} in facts

    def test_parses_prolog_line_with_period_still_works(self) -> None:
        # the canonical grammar keeps working after the tolerance change
        facts = m._extract_facts('vuln("sof").')
        assert {"name": "vuln", "value": "sof"} in facts

    def test_parses_multi_line_block_with_trailing_spaces(self) -> None:
        # trained-mr8's exact shape: two trailing spaces per line
        facts = m._extract_facts(
            "vuln(sof)  \nfmtstr_read(true)  \nseccomp(false)  \nfunction_pointer_on_heap(true)"
        )
        assert {"name": "vuln", "value": "sof"} in facts
        assert {"name": "fmtstr_read", "value": "true"} in facts
        assert {"name": "seccomp", "value": "false"} in facts
        assert {"name": "function_pointer_on_heap", "value": "true"} in facts

    def test_parses_value_wrapper_shape_from_base(self) -> None:
        # base emits value("vuln", "uaf") - the GRPO lab-loop wrapper
        # under the ``value`` name, the vuln fact is the second arg
        facts = m._extract_facts(
            'value("function_pointer_on_heap", true)\n'
            'value("allocator", "tcache")\n'
            'value("glibc_minor", 36)\n'
            'value("vuln", "uaf")'
        )
        assert {"name": "vuln", "value": "uaf"} in facts

    def test_parses_predicate_wrapper_shape_without_period(self) -> None:
        facts = m._extract_facts('predicate("vuln", "uaf")')
        assert {"name": "vuln", "value": "uaf"} in facts

    def test_degenerate_single_arg_wrapper_is_dropped_without_period(self) -> None:
        # predicate(win_function): the argument itself is the fact
        # name slot, not a wrapped tuple - a degenerate marker line,
        # dropped silently, never parsed as a {predicate, ...} fact.
        assert m._extract_facts("predicate(win_function)") == []

    def test_degenerate_single_arg_wrapper_is_dropped_with_period(self) -> None:
        # predicate("win_function").: same degenerate marker, but the
        # trailing Prolog period must not push it into the Prolog
        # branch as a {name: predicate, value: win_function} fact.
        assert m._extract_facts('predicate("win_function").') == []

    def test_refusal_prose_yields_no_facts(self) -> None:
        # trained-mr8's refusal on callme-002: the Prolog comment line
        # must not be parsed as facts
        prose = (
            "% The source code does not show any of the allowed predicates "
            "(vuln, fmtstr_read, seccomp, allocator, glibc_minor, "
            "function_pointer_on_heap) as per the challenge material. "
            "No facts are emitted."
        )
        assert m._extract_facts(prose) == []

    def test_wrong_vuln_value_is_a_failure_not_a_parse_error(self) -> None:
        # vuln(true): the value is outside the vuln domain, but the
        # metric must count it as a wrong answer (a failure), not as
        # "model did not return valid facts" - the attempt is real
        facts = m._extract_facts("vuln(true)")
        assert facts == [{"name": "vuln", "value": "true"}]
        result = m._evaluate_response("vuln(true)", "uaf_heap")
        assert result["valid"] is True
        assert result["success"] is False

    def test_wrong_vuln_class_is_a_failure(self) -> None:
        # the poisoned uaf_heap answer: vuln("fmtstr") on a uaf target
        result = m._evaluate_response('vuln("fmtstr")', "uaf_heap")
        assert result["valid"] is True
        assert result["vuln_type"] == "fmtstr"
        assert result["expected_vuln"] == "uaf"
        assert result["success"] is False


class TestEvaluation:
    def test_correct_vuln_is_a_success(self, tmp_path: Path) -> None:
        paths = _stage_datasets(tmp_path, {"callme-002": "sof"})
        result = m._evaluate_response(
            '{"facts": [{"name": "vuln", "value": "sof"}]}', "callme-002", paths=paths
        )
        assert result["valid"] is True
        assert result["correct_vuln_type"] is True
        assert result["vuln_type"] == "sof"
        assert result["expected_vuln"] == "sof"
        assert result["success"] is True

    def test_wrong_vuln_is_a_failure(self, tmp_path: Path) -> None:
        paths = _stage_datasets(tmp_path, {"callme-002": "sof"})
        result = m._evaluate_response(
            '{"facts": [{"name": "vuln", "value": "fmtstr"}]}', "callme-002", paths=paths
        )
        assert result["valid"] is True
        assert result["correct_vuln_type"] is False
        assert result["success"] is False

    def test_srop_no_longer_impossible(self, tmp_path: Path) -> None:
        # Bug 1: srop-001 expected "srop" which the old prompt forbade.
        # Ground truth is now "sof"; a model answering "sof" succeeds.
        paths = _stage_datasets(tmp_path, {"srop-001": "sof"})
        result = m._evaluate_response(
            '{"facts": [{"name": "vuln", "value": "sof"}]}', "srop-001", paths=paths
        )
        assert result["success"] is True
        assert result["expected_vuln"] == "sof"

    def test_uaf_train_not_a_free_pass(self, tmp_path: Path) -> None:
        # Bug 3: uaf_train_a/b were always correct (absent from map).
        # Now they are real checks: the model must say "uaf".
        paths = _stage_datasets(tmp_path, {"uaf_train_a": "uaf", "uaf_train_b": "uaf"})
        right = m._evaluate_response(
            '{"facts": [{"name": "vuln", "value": "uaf"}]}', "uaf_train_a", paths=paths
        )
        wrong = m._evaluate_response(
            '{"facts": [{"name": "vuln", "value": "sof"}]}', "uaf_train_b", paths=paths
        )
        assert right["success"] is True
        assert wrong["success"] is False

    def test_no_facts_is_invalid(self, tmp_path: Path) -> None:
        paths = _stage_datasets(tmp_path, {"callme-002": "sof"})
        result = m._evaluate_response("this is not facts", "callme-002", paths=paths)
        assert result["valid"] is False
        assert result["success"] is False

    def test_no_ground_truth_is_a_pass_not_a_failure(self, tmp_path: Path) -> None:
        # Binary without a recorded class can't be verified -> pass if the
        # model emitted valid facts (honest, not a fabricated expectation).
        paths = _stage_datasets(tmp_path, {})
        result = m._evaluate_response(
            '{"facts": [{"name": "vuln", "value": "sof"}]}', "ret2csu", paths=paths
        )
        assert result["expected_vuln"] is None
        assert result["success"] is True


class TestHybridFormat:
    def test_uses_fact_prompt_not_the_old_model_only_prompt(self) -> None:
        # The prompt is FACT_PROMPT (shared with the full arm): it asks for
        # fact predicates, not the old free-form JSON with vuln_type enum.
        out = m.FACT_PROMPT.format(material="int vuln(){}")
        assert "fact predicates" in out
        assert "vuln" in out
        assert "vuln_type" not in out
        assert "sof|fmtstr|uaf|ret2libc|ret2plt|ret2func|shellcode|orw|other" not in out

    def test_run_model_only_signature_unchanged(self) -> None:
        import inspect

        sig = inspect.signature(m.run_model_only)
        params = list(sig.parameters.keys())
        assert params == ["targets", "workdir", "provider", "model", "seed"]

    def test_pilot_is_importable_and_callable(self) -> None:
        assert callable(m.run_model_only)
