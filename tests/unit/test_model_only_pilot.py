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
