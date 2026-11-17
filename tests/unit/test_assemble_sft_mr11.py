"""Tests for the MR11 SFT dataset assembler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import assemble_sft_mr11 as asm


def facts_call() -> dict:
    return {
        "thought": "read the challenge material, emit the source-derived facts",
        "action": "facts",
        "action_input": {"predicates": ['nx("true").', 'pie("false").']},
    }


def augure_call() -> dict:
    return {
        "thought": "guess the technique from the filename",
        "action": "augure",
        "action_input": {"technique": "ret2libc"},
    }


def record(
    rid: str,
    *,
    first: str = "facts",
    flagged: bool = False,
    metadata: dict | None = None,
) -> dict:
    call = facts_call() if first == "facts" else augure_call()
    record = {
        "id": rid,
        "domain": "agentic",
        "messages": [
            {
                "role": "user",
                "content": "Solve this CTF challenge against the local lab target.",
            },
            {"role": "assistant", "content": json.dumps(call)},
            {"role": "tool", "content": 'nx("true").\npie("false").'},
            {
                "role": "assistant",
                "content": json.dumps({"flag": "FLAG{x}" if flagged else None}),
            },
        ],
    }
    if metadata is not None:
        record["metadata"] = metadata
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_first_action_is_facts():
    assert asm.first_action_is_facts(record("a"))
    assert not asm.first_action_is_facts(record("a", first="augure"))
    # a non-JSON first assistant message is not a facts call
    assert not asm.first_action_is_facts(
        {"messages": [{"role": "assistant", "content": "let me look..."}]}
    )
    assert not asm.first_action_is_facts({"messages": []})


def test_uaf_class_gate():
    # the uaf-class targets may carry the class fact
    assert asm.uaf_class_legal(record("ctf-lictor-uaf_train_a-001"))
    # any other target claiming vuln("uaf") is the dataset-v1 poison
    poison = record("ctf-lictor-fmtstr-001")
    poison["messages"][1]["content"] = json.dumps(
        {
            "thought": "t",
            "action": "facts",
            "action_input": {"predicates": ['vuln("uaf").']},
        }
    )
    assert not asm.uaf_class_legal(poison)
    # the class fact must not hide in plain-text assistant content either
    plain = record("ctf-lictor-fmtstr-001")
    plain["messages"][1]["content"] = json.dumps(facts_call()) + '\nvuln("uaf").'
    assert not asm.uaf_class_legal(plain)


def test_assembly_filters_provenance_and_dedup(tmp_path: Path):
    traj = tmp_path / "trajectories"
    traj.mkdir()
    flagged_meta = {"flag_verified": True, "pilot": {"model": "teacher-qwen3-4b"}}
    base = tmp_path / "base.jsonl"
    write_jsonl(
        traj / "trajectories-seed-100.jsonl",
        [
            record("ctf-lictor-fmtstr-001", flagged=True, metadata=dict(flagged_meta)),
            record(
                "ctf-lictor-orw-001", flagged=True, metadata=dict(flagged_meta)
            ),  # kept
            record(
                "ctf-lictor-orw-001", flagged=True, metadata=dict(flagged_meta)
            ),  # identical, dropped
            record(
                "ctf-lictor-shellcode-001", flagged=False, metadata={}
            ),  # unflagged, dropped
            record(
                "ctf-lictor-fmtstr-001", first="augure", metadata=dict(flagged_meta)
            ),  # facts-skip, dropped
            record(
                "ctf-lictor-fmtstr-001", flagged=True, metadata=dict(flagged_meta)
            ),  # identical content: dropped
        ],
    )
    write_jsonl(
        base,
        [
            record("ctf-lictor-ret2plt-001", metadata={}),  # facts-first, kept
            record(
                "ctf-lictor-uaf_train_a-001", first="augure", metadata={}
            ),  # facts-skip, dropped
        ],
    )
    out = tmp_path / "mr11" / "train.jsonl"
    argv = sys.argv
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
        sys.argv = argv

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ids = [r["id"] for r in rows]
    # only identical content collapses (the same-id distinct runs both
    # survive: the class balance relies on them)
    assert ids == [
        "ctf-lictor-fmtstr-001",
        "ctf-lictor-orw-001",
        "ctf-lictor-ret2plt-001",
    ]
    mining = [r for r in rows if r["metadata"].get("mr11_source") == "teacher-mining"]
    base_rows = [r for r in rows if r["metadata"].get("mr11_source") == "mr10-base"]
    assert len(mining) == 2
    assert len(base_rows) == 1
    # the trainer's validation gate requires the provenance keys
    for row in mining:
        for key in ("source", "teachers", "quality", "recitation_risk"):
            assert key in row["metadata"], key
