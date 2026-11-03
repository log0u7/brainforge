"""Tests for the assembler's provenance enrichment and quality gate.

The training pipeline validates records against the provenance schema
(``source``, ``teachers``, ``quality.passed``, ``recitation_risk``) and
the MR5 lesson says: SFT must not clone FAILED trajectories (the
failed rollouts stay on disk for the GRPO zero-reward sidecar). The
assembler therefore:

1. drops rollout records whose verdict carries no flag (unless
   ``--keep-failed``), counting them;
2. fills the missing provenance keys on every emitted record without
   overwriting the replay records' own provenance (their
   ``recitation_risk: true`` stays).
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


def _rollout_record(rid: str, flag: str | None) -> dict:
    verdict = {"verdict": "solved" if flag else "failed", "flag": flag}
    return {
        "id": rid,
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": "challenge"},
            {"role": "assistant", "content": json.dumps(verdict)},
        ],
        "metadata": {"pilot": {"model": "base-qwen3-1.7b"}},
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def test_failed_rollouts_dropped_for_sft(tmp_path: Path) -> None:
    """A failed trajectory is not SFT material: cloning it teaches the
    model to fail. The successes stay, the failures are counted."""
    module = _load_script()
    records = [
        _rollout_record("lab-fmtstr-001", "FLAG{a}"),
        _rollout_record("lab-fmtstr-001", None),
        _rollout_record("lab-orw-001", "FLAG{b}"),
    ]
    src = _write_jsonl(tmp_path / "src.jsonl", records)
    out = tmp_path / "train.jsonl"
    summary = module.assemble(sources=[src], out=out, per_target_cap=10, empty_completion_share=0.2)
    assert summary["records"] == 2
    assert summary["failed_dropped"] == 1


def test_provenance_keys_filled_without_overwriting_replay(tmp_path: Path) -> None:
    """Every emitted record carries source/teachers/quality/
    recitation_risk; the replay record's own provenance (including its
    pre-cutoff recitation_risk=true) is preserved."""
    module = _load_script()
    replay = {
        "id": "replay-1",
        "domain": "ctf",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": json.dumps({"flag": "FLAG{r}"})},
        ],
        "metadata": {
            "source": {"type": "rop-emporium", "date": "2020-01-01"},
            "teachers": [{"role": "ctf_teacher", "kind": "teacher"}],
            "recitation_risk": True,
        },
    }
    src = _write_jsonl(
        tmp_path / "src.jsonl", [_rollout_record("lab-fmtstr-001", "FLAG{a}"), replay]
    )
    out = tmp_path / "train.jsonl"
    module.assemble(sources=[src], out=out, per_target_cap=10, empty_completion_share=0.2)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").strip().splitlines()]
    by_id = {r["id"]: r for r in records}
    rollout_meta = by_id["lab-fmtstr-001"]["metadata"]
    for key in ("source", "teachers", "quality", "recitation_risk"):
        assert key in rollout_meta, key
    assert rollout_meta["quality"]["passed"] is True
    assert rollout_meta["recitation_risk"] is False
    assert rollout_meta["teachers"][0]["model"] == "base-qwen3-1.7b"
    replay_meta = by_id["replay-1"]["metadata"]
    assert replay_meta["recitation_risk"] is True
    assert replay_meta["source"]["type"] == "rop-emporium"
    # the quality gate must be present on the replay record too, but
    # its own verdict's flag was verified just now
    assert replay_meta["quality"]["passed"] is True


def test_enriched_records_pass_the_trainer_validation(tmp_path: Path) -> None:
    """The end-to-end contract: every record the assembler emits passes
    ``brainforge.dataset.validation.validate_record``."""
    from brainforge.dataset.validation import validate_record

    module = _load_script()
    src = _write_jsonl(
        tmp_path / "src.jsonl",
        [_rollout_record("lab-fmtstr-001", "FLAG{a}")],
    )
    out = tmp_path / "train.jsonl"
    module.assemble(sources=[src], out=out, per_target_cap=10, empty_completion_share=0.2)
    for line in out.read_text(encoding="utf-8").strip().splitlines():
        errors = validate_record(json.loads(line))
        assert errors == [], errors


def test_empty_tool_observations_get_a_truthful_marker(tmp_path: Path) -> None:
    """A successful no-facts rollout (the best records: route from
    profiler facts alone) has an EMPTY facts-step tool observation; the
    validator forbids empty content, so the assembler stamps a truthful
    marker instead of dropping the record."""
    module = _load_script()
    record = _rollout_record("lab-fmtstr-001", "FLAG{a}")
    record["messages"] = [
        {"role": "user", "content": "challenge"},
        {"role": "assistant", "content": json.dumps({"action": "facts"})},
        {"role": "tool", "content": ""},
        {"role": "assistant", "content": json.dumps({"verdict": "solved", "flag": "FLAG{a}"})},
    ]
    src = _write_jsonl(tmp_path / "src.jsonl", [record])
    out = tmp_path / "train.jsonl"
    module.assemble(sources=[src], out=out, per_target_cap=10, empty_completion_share=0.2)
    from brainforge.dataset.validation import validate_record

    for line in out.read_text(encoding="utf-8").strip().splitlines():
        parsed = json.loads(line)
        errors = validate_record(parsed)
        assert errors == [], errors
        assert all((m["content"] or "").strip() for m in parsed["messages"])


def test_keep_failed_flag_retains_failed_rollouts(tmp_path: Path) -> None:
    module = _load_script()
    records = [_rollout_record("lab-fmtstr-001", None)]
    src = _write_jsonl(tmp_path / "src.jsonl", records)
    out = tmp_path / "train.jsonl"
    summary = module.assemble(
        sources=[src],
        out=out,
        per_target_cap=10,
        empty_completion_share=0.2,
        keep_failed=True,
    )
    assert summary["records"] == 1
    assert summary["failed_dropped"] == 0
