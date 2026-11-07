"""Tests for the MR8 fact-classification SFT assembler.

MR8 extends the mr7-58 recipe (single-turn, JSON facts) with the three
printf-family training twins as counter-examples:

- ret2libc_train, srop_train, callme_train: printf/snprintf in the
  material, vuln=sof, bounded-copy sink. Measured baseline routes
  (2026-11-07 fresh acquire+harvest+plan): all three plan ret2libc
  (srop_train: applicable [ret2libc, ret2vdso], selects ret2libc).

The twins carry no case file in examples/cases; their material comes
from the lictor lab (service.c) via build_sft_facts.material_for.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "assemble_sft_mr8.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("assemble_sft_mr8", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mr8_twins_in_vuln_truth() -> None:
    """The three printf-family twins are in VULN_TRUTH with vuln=sof."""
    module = _load_script()
    for twin in ("ret2libc_train", "srop_train", "callme_train"):
        assert twin in module.VULN_TRUTH
        assert module.VULN_TRUTH[twin] == "sof"


def test_mr8_twins_have_seccomp_false_override() -> None:
    """The twins call system/execve: no seccomp filter."""
    module = _load_script()
    for twin in ("ret2libc_train", "srop_train", "callme_train"):
        assert twin in module.FACT_OVERRIDES
        assert module.FACT_OVERRIDES[twin] == {"seccomp": "false"}


def test_mr8_records_use_json_facts_envelope() -> None:
    """Each record's assistant completion is a JSON object with a
    ``facts`` array of {name, value} dicts - the format the bench
    ask_facts requires. Records carry provenance (id, domain, metadata)
    so the trainer's validation gate accepts them."""
    module = _load_script()
    votes = {target: {} for target in module.VULN_TRUTH}
    records, skipped = module.build_records(votes, Path("examples/cases"))
    assert skipped == [], skipped
    assert len(records) == len(module.VULN_TRUTH)
    for record in records:
        assert record["id"].startswith("sft-mr8-facts-")
        assert record["domain"] == "agentic"
        metadata = record["metadata"]
        assert metadata["verified_by"] == "decision"
        assert metadata["training_target"] == record["id"][len("sft-mr8-facts-") :]
        assert metadata["quality"]["passed"] is True
        assert metadata["recitation_risk"] is False
        assert metadata["teachers"][0]["model"] == "z-ai/glm-5.3-flash"
        assert metadata["source"]["path"] == f"lab/targets/{metadata['training_target']}"
        completion = json.loads(record["messages"][1]["content"])
        assert "facts" in completion
        assert isinstance(completion["facts"], list)
        for fact in completion["facts"]:
            assert set(fact.keys()) == {"name", "value"}


def test_mr8_twin_record_has_sof_and_seccomp_false() -> None:
    """The twin completion carries vuln=sof (ground truth) and
    seccomp=false (override from the source), nothing else."""
    module = _load_script()
    votes = {target: {} for target in module.VULN_TRUTH}
    records, _ = module.build_records(votes, Path("examples/cases"))
    for record in records:
        material = record["messages"][0]["content"]
        if "ret2libc_train" in material:
            completion = json.loads(record["messages"][1]["content"])
            facts = {f["name"]: f["value"] for f in completion["facts"]}
            assert facts["vuln"] == "sof", facts
            assert facts["seccomp"] == "false", facts
            return
    raise AssertionError("no ret2libc_train record found")
