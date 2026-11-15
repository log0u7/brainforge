"""Tests for the SFT fact-suggestion dataset builder.

``scripts/build_sft_facts.py`` is not part of the ``brainforge``
package: the tests load it directly from the repository layout so the
contract (ground truth per target, no uaf_heap contamination) is
verified without installing the script.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_sft_facts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_sft_facts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_loads_and_defines_contract() -> None:
    module = _load_script()
    assert hasattr(module, "GROUND_TRUTH")
    assert hasattr(module, "TRAINING_TARGETS")
    assert set(module.TRAINING_TARGETS) == set(module.GROUND_TRUTH.keys())


def test_uaf_heap_is_excluded_from_training_targets() -> None:
    """The lab original ``uaf_heap`` material is the eval challenge
    ``ctf-lictor-uaf_heap-001``: it must never appear as a training
    target (contamination contract)."""
    module = _load_script()
    assert "uaf_heap" not in module.GROUND_TRUTH
    assert "uaf_heap" not in module.TRAINING_TARGETS


def test_mr8_counterexample_twins_have_empty_ground_truth() -> None:
    """The MR8 counter-examples: the three printf-family training twins
    carry a format-string-SHAPED PLT (printf/snprintf) with NO format
    string bug - vuln=sof is the whole truth and the LLM must emit
    NOTHING there. The mr7 bench measured the failure mode: the fact
    emitter asserted vuln("fmtstr") on a strcpy target whose PLT shows
    printf/snprintf (shellcode-002), and the suggested-only poison
    flipped the route (3 lost reps). These twins teach the exact
    discrimination: printf-family in the PLT is not a format string."""
    module = _load_script()
    for twin in ("ret2libc_train", "srop_train", "callme_train"):
        assert twin in module.GROUND_TRUTH
        assert module.GROUND_TRUTH[twin] == []
        assert twin in module.TRAINING_TARGETS


def test_fmtstr_ground_truth_is_intact() -> None:
    module = _load_script()
    assert module.GROUND_TRUTH["fmtstr"] == [
        'vuln("fmtstr").',
        'fmtstr_read("true").',
    ]


def test_builder_produces_jsonl_with_provenance(tmp_path: Path) -> None:
    """Running the script must write valid JSONL records that carry
    provenance metadata (pipeline, target, n_facts). The lab dir is
    a tmp fixture: the CI runner has no lictor checkout."""
    module = _load_script()
    lab_dir = tmp_path / "lab" / "targets"
    for target in module.TRAINING_TARGETS:
        (lab_dir / target).mkdir(parents=True, exist_ok=True)
        (lab_dir / target / "service.c").write_text(f"/* {target} fixture */\n", encoding="utf-8")
    out = tmp_path / "train.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--out",
            str(out),
            "--lab-dir",
            str(lab_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in result.stdout
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(module.TRAINING_TARGETS)
    for line in lines:
        record = json.loads(line)
        assert record["domain"] == "agentic"
        metadata = record["metadata"]
        assert metadata["pipeline"] == "sft-facts"
        assert metadata["target"] in module.GROUND_TRUTH
        assert metadata["n_facts"] == len(module.GROUND_TRUTH[metadata["target"]])
        messages = record["messages"]
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "\n".join(module.GROUND_TRUTH[metadata["target"]])
