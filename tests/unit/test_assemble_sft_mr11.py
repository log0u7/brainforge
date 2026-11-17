"""Tests for the MR11 SFT dataset assembler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import assemble_sft_mr11 as asm


def facts_call() -> str:
    return json.dumps(
        {
            "thought": "read the challenge material, emit the source-derived facts",
            "action": "facts",
            "action_input": {"predicates": ['nx("true").']},
        }
    )


def augure_call() -> str:
    return json.dumps(
        {
            "thought": "guess the technique from the filename",
            "action": "augure",
            "action_input": {"technique": "ret2libc"},
        }
    )


def record(
    rid: str, *, first: str = "facts", flagged: bool = False, source: str | None = None
) -> dict:
    call = facts_call() if first == "facts" else augure_call()
    record = {
        "id": rid,
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": "Solve this CTF challenge."},
            {"role": "assistant", "content": call},
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": json.dumps({"flag": "FLAG{x}" if flagged else None}),
            },
        ],
    }
    if source is not None:
        record["metadata"] = {"flag_verified": flagged}
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_first_action_is_facts_true_and_false():
    assert asm.first_action_is_facts(record("a"))
    assert not asm.first_action_is_facts(record("a", first="augure"))
    # a non-JSON first assistant message is not a facts call
    assert not asm.first_action_is_facts(
        {"messages": [{"role": "assistant", "content": "let me think..."}]}
    )
    # no assistant messages at all
    assert not asm.first_action_is_facts(
        {"messages": [{"role": "user", "content": "hi"}]}
    )


def test_assembly_filters_and_dedup(tmp_path: Path):
    traj = tmp_path / "trajectories"
    traj.mkdir()
    write_jsonl(
        traj / "trajectories-seed-100.jsonl",
        [
            record("lab-jop-001", flagged=True, source="mining"),  # kept
            record("lab-jop-001", flagged=True, source="mining"),  # dup id, dropped
            record(
                "lab-brop-001", flagged=False, source="mining"
            ),  # unflagged, dropped
            record(
                "lab-fmtstr-001", flagged=True, first="augure", source="mining"
            ),  # facts-skip, dropped
        ],
    )
    base = tmp_path / "base.jsonl"
    write_jsonl(
        base,
        [
            record(
                "lab-fmtstr-001", source="mr10-base"
            ),  # facts-first but id dup, dropped
            record("lab-ret2libc-001", source="mr10-base"),  # kept
            record(
                "lab-uaf-001", first="augure", source="mr10-base"
            ),  # facts-skip, dropped
        ],
    )
    out = tmp_path / "mr11" / "train.jsonl"
    # run main with patched argv
    orig_argv = sys.argv

    orig_argv = sys.argv
    sys.argv = [
        "assemble_sft_mr11.py",
        "--trajectories",
        str(traj),
        "--base",
        str(base),
        "--out",
        str(out),
    ]
    try:
        assert asm.main() == 0
    finally:
        sys.argv = orig_argv

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    # the invalid mining fmtstr record (facts-skip) never claims its id,
    # so the base's valid fmtstr row survives
    assert [r["id"] for r in rows] == [
        "lab-jop-001",
        "lab-fmtstr-001",
        "lab-ret2libc-001",
    ]
    assert rows[0]["metadata"]["mr11_source"] == "teacher-mining"
    assert rows[1]["metadata"]["mr11_source"] == "mr10-base"
    assert rows[2]["metadata"]["mr11_source"] == "mr10-base"
