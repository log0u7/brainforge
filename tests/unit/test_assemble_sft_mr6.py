"""Tests for the MR6 SFT dataset assembler.

``scripts/assemble_sft_mr6.py`` merges the mined trails, the fresh
rollout trajectories, the decision-verified fact records and the
replay corpora into one training JSONL. Contracts under test: the
contamination gate refuses any record whose ID translates into the
clean-protocol eval namespace (nonzero exit), near-duplicate rollout
records are capped per target, and the empty-completion share is
capped so the dataset cannot silently re-create the MR5 failure (a
model trained to say nothing).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "assemble_sft_mr6.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("assemble_sft_mr6", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(rid: str, content: str = 'vuln("fmtstr").') -> dict:
    return {
        "id": rid,
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": "challenge"},
            {"role": "assistant", "content": content},
        ],
        "metadata": {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def test_assemble_merges_all_sources(tmp_path: Path) -> None:
    module = _load_script()
    trails = _write_jsonl(tmp_path / "trails.jsonl", [_record("lab-fmtstr-001")])
    rollouts = _write_jsonl(
        tmp_path / "rollouts.jsonl",
        [_record("lab-fmtstr-001", f"seed-{i} facts") for i in range(5)],
    )
    facts = _write_jsonl(tmp_path / "facts.jsonl", [_record("sft-verified-facts-orw")])
    replay = _write_jsonl(tmp_path / "replay.jsonl", [_record(f"replay-{i}") for i in range(3)])
    out = tmp_path / "train.jsonl"
    summary = module.assemble(
        sources=[trails, rollouts, facts, replay],
        out=out,
        per_target_cap=10,
        empty_completion_share=0.3,
    )
    assert summary["records"] == 10
    assert summary["contaminating"] == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10


def test_assemble_caps_empty_completions(tmp_path: Path) -> None:
    """4/6 empty completions is what killed trained-sft-facts: the
    assembler caps the empty share instead of letting it dominate."""
    module = _load_script()
    records = [_record(f"r-{i}", content="" if i < 8 else 'vuln("sof").') for i in range(10)]
    src = _write_jsonl(tmp_path / "src.jsonl", records)
    out = tmp_path / "train.jsonl"
    summary = module.assemble(
        sources=[src], out=out, per_target_cap=100, empty_completion_share=0.2
    )
    # 2 non-empty kept in full; the empty share is capped against the
    # kept total, so almost every empty record is dropped
    assert summary["records"] == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    empties = sum(1 for line in lines if json.loads(line)["messages"][-1]["content"] == "")
    assert empties <= 0.2 * summary["records"] + 1
    assert summary["empties_dropped"] == 8


def test_assemble_refuses_eval_contamination(tmp_path: Path) -> None:
    """An eval-derived record must never reach the training file: the
    assembler drops it AND reports it; a run where contamination was
    found exits nonzero at the CLI layer."""
    module = _load_script()
    bad = _record("ctf-lictor-uaf_heap-002-001")
    src = _write_jsonl(tmp_path / "bad.jsonl", [bad, _record("lab-fmtstr-001")])
    out = tmp_path / "train.jsonl"
    summary = module.assemble(sources=[src], out=out, per_target_cap=10, empty_completion_share=1.0)
    assert summary["contaminating"] == 1
    text = out.read_text(encoding="utf-8")
    assert "uaf_heap" not in text
    assert "lab-fmtstr-001" in text


def test_rollout_repetitions_capped_per_target(tmp_path: Path) -> None:
    """40 seeds x 6 targets = 240 rollout records: useful variety, but
    one target must not dominate the mix (the cap keeps the dataset
    balanced across the technique classes)."""
    module = _load_script()
    records = [_record("lab-fmtstr-001", f"variant {i}") for i in range(30)] + [
        _record("lab-orw-001", f"variant {i}") for i in range(3)
    ]
    src = _write_jsonl(tmp_path / "src.jsonl", records)
    out = tmp_path / "train.jsonl"
    summary = module.assemble(sources=[src], out=out, per_target_cap=10, empty_completion_share=1.0)
    assert summary["records"] == 13
    counts: dict[str, int] = {}
    for line in out.read_text(encoding="utf-8").strip().splitlines():
        counts[json.loads(line)["id"]] = counts.get(json.loads(line)["id"], 0) + 1
    assert counts["lab-fmtstr-001"] == 10
    assert counts["lab-orw-001"] == 3
