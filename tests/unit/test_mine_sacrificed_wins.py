"""Tests for the sacrificed-eval win miner.

``scripts/build_sft_trails.py`` gains ``mine_sacrificed_wins``: it
mines ONE record per EXECUTED WIN from a target that was moved from
the eval set into training (the sacrifice ledger, e.g. uaf_heap-002).
The unit under test is the facts reconstruction: the final on-disk
``target.facts`` is a later fresh harvest that may LACK the facts the
winning plan actually used (they were suggested mid-run and later
overwritten), so the miner must recover them from the trail's
``suggest`` events, and must verify the winning plan's provenance
evidence is present in the reconstruction before keeping the record.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_sft_trails.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_sft_trails", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _win_trail(target: str, wins: list[str], *, evidence: list[tuple[str, str]]) -> str:
    """A trail for one target: acquire -> suggest (uaf facts) -> plan
    (selected uaf_tcache, provenance evidence) -> outcome. One segment
    per winning lictor run; a trailing losing outcome adds noise."""
    lines: list[str] = []
    lines.append(json.dumps({"run": "acq-1", "type": "acquire", "target": "127.0.0.1:9987"}))
    for i, run in enumerate(wins):
        lines.append(
            json.dumps(
                {
                    "run": f"sug-{i}",
                    "type": "suggest",
                    "suggested": [["vuln", ["uaf"]], ["allocator", ["tcache"]]],
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "run": f"plan-{i}",
                    "type": "plan",
                    "mode": "handoff (prints a recipe; executes nothing)",
                    "selected": "uaf_tcache",
                    "applicable": ["uaf_tcache"],
                    "explain": {
                        "provenance": {
                            "uaf_tcache": [
                                {
                                    "rule": "app_uaf_tcache",
                                    "evidence": [list(pair) for pair in evidence],
                                }
                            ]
                        }
                    },
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "run": run,
                    "type": "outcome",
                    "technique": "uaf_tcache",
                    "success": True,
                    "executed": True,
                }
            )
        )
    lines.append(
        json.dumps(
            {
                "run": "lose-1",
                "type": "outcome",
                "technique": "fmtstr_write",
                "success": False,
                "executed": True,
            }
        )
    )
    return "".join(line + "\n" for line in lines)


def _sacrifice_run_dir(root: Path, *, evidence: list[tuple[str, str]]) -> Path:
    """One run dir holding a sacrificed eval target (uaf_heap-002) whose
    final on-disk facts LACK the suggested uaf facts (later harvest)."""
    run = root / "run-20261104-test"
    d = run / "uaf_heap-002"
    d.mkdir(parents=True)
    (d / "trail-uaf_heap-002.jsonl").write_text(
        _win_trail("uaf_heap-002", ["win-a", "win-b"], evidence=evidence),
        encoding="utf-8",
    )
    # final harvest state: NO vuln("uaf"), NO allocator("tcache")
    (d / "target.facts").write_text(
        'nx("true").\nwin_symbol("win", 4198902).\n',
        encoding="utf-8",
    )
    (d / "run.json").write_text(
        json.dumps(
            {
                "status": "flagged",
                "attempts": [
                    {
                        "technique": "uaf_tcache",
                        "executed": True,
                        "success": True,
                        "flag": "FLAG{uaf_002}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_mine_sacrificed_wins_extracts_one_record_per_win(tmp_path: Path) -> None:
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "uaf"), ("allocator", "tcache")])
    out = tmp_path / "sacrificed.jsonl"
    summary = module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    assert summary["records"] == 2
    records = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert all(r["id"] == "ctf-lictor-uaf_heap-002-001" for r in records)
    verdicts = [json.loads(r["messages"][-1]["content"]) for r in records]
    assert all(v["verdict"] == "solved" and v["flag"] == "FLAG{uaf_002}" for v in verdicts)


def test_reconstruction_recovers_suggested_class_facts(tmp_path: Path) -> None:
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "uaf"), ("allocator", "tcache")])
    out = tmp_path / "sacrificed.jsonl"
    module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    record = json.loads(out.read_text().strip().splitlines()[0])
    # step 1 is the facts step: its predicates carry the suggested facts
    # even though the final on-disk facts lost them
    predicates = json.loads(record["messages"][1]["content"])["action_input"]["predicates"]
    assert 'vuln("uaf").' in predicates
    assert 'allocator("tcache").' in predicates
    assert 'nx("true").' in predicates  # the final facts survive too


def test_records_carry_executed_success_provenance(tmp_path: Path) -> None:
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "uaf"), ("allocator", "tcache")])
    out = tmp_path / "sacrificed.jsonl"
    module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    records = [json.loads(line) for line in out.read_text().strip().splitlines()]
    provs = {r["metadata"]["provenance"] for r in records}
    assert provs == {"executed-success-win-a", "executed-success-win-b"}
    assert all(r["metadata"]["flag_verified"] for r in records)
    # the internal evidence key is bookkeeping, not training content: it
    # must never leak into the serialized record
    assert not any("_provenance" in json.dumps(r) for r in records)


def test_win_without_selected_plan_is_skipped(tmp_path: Path) -> None:
    module = _load_script()
    run = tmp_path / "run-no-plan"
    d = run / "uaf_heap-002"
    d.mkdir(parents=True)
    # winning outcome with NO preceding plan event carrying a selection
    lines = [
        json.dumps({"run": "acq", "type": "acquire"}),
        json.dumps(
            {
                "run": "win",
                "type": "outcome",
                "technique": "uaf_tcache",
                "success": True,
                "executed": True,
            }
        ),
    ]
    (d / "trail-uaf_heap-002.jsonl").write_text("".join(line + "\n" for line in lines))
    (d / "target.facts").write_text('nx("true").\n')
    (d / "run.json").write_text(json.dumps({"status": "flagged", "attempts": []}))
    out = tmp_path / "sacrificed.jsonl"
    summary = module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    assert summary["records"] == 0
    assert summary["skipped_no_plan"] == 1
    assert not out.exists() or not out.read_text().strip()


def test_evidence_missing_from_reconstruction_skips_win(tmp_path: Path) -> None:
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "never_suggested")])
    out = tmp_path / "sacrificed.jsonl"
    summary = module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    assert summary["records"] == 0
    assert summary["skipped_evidence_missing"] == 2


def test_mine_sacrificed_wins_dedups_duplicate_facts(tmp_path: Path) -> None:
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "uaf"), ("allocator", "tcache")])
    out = tmp_path / "sacrificed.jsonl"
    module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    record = json.loads(out.read_text().strip().splitlines()[0])
    predicates = json.loads(record["messages"][1]["content"])["action_input"]["predicates"]
    assert len(predicates) == len(set(predicates))
    assert 'vuln("uaf").' in predicates and predicates.count('vuln("uaf").') == 1


def test_expected_flag_oracle_fills_the_verdict(tmp_path: Path) -> None:
    """The bench trail holds executed wins but run.json may carry no
    successful attempt (the pilot budget ran out while the lictor
    repetitions won): the oracle flag from the sacrificed case file
    fills the verdict so the record reads solved, like every other
    SFT record."""
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "uaf"), ("allocator", "tcache")])
    # strip the winning attempt from run.json: the trail wins are
    # lictor repetitions, not pilot attempts
    run_json = json.loads((run / "uaf_heap-002" / "run.json").read_text())
    run_json["attempts"] = []
    (run / "uaf_heap-002" / "run.json").write_text(json.dumps(run_json))
    out = tmp_path / "sacrificed.jsonl"
    summary = module.mine_sacrificed_wins(
        [run],
        out,
        target="uaf_heap-002",
        challenge_id="ctf-lictor-uaf_heap-002-001",
        expected_flag="FLAG{uaf_002}",
    )
    assert summary["records"] == 2
    records = [json.loads(line) for line in out.read_text().strip().splitlines()]
    verdicts = [json.loads(r["messages"][-1]["content"]) for r in records]
    assert all(v["verdict"] == "solved" and v["flag"] == "FLAG{uaf_002}" for v in verdicts)
    assert all(r["metadata"]["flag_verified"] for r in records)


def test_no_oracle_flag_keeps_the_unflagged_outcome(tmp_path: Path) -> None:
    module = _load_script()
    run = _sacrifice_run_dir(tmp_path, evidence=[("vuln", "uaf"), ("allocator", "tcache")])
    run_json = json.loads((run / "uaf_heap-002" / "run.json").read_text())
    run_json["attempts"] = []
    (run / "uaf_heap-002" / "run.json").write_text(json.dumps(run_json))
    out = tmp_path / "sacrificed.jsonl"
    module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    records = [json.loads(line) for line in out.read_text().strip().splitlines()]
    verdicts = [json.loads(r["messages"][-1]["content"]) for r in records]
    assert all(v["flag"] is None for v in verdicts)
    assert all(not r["metadata"]["flag_verified"] for r in records)


def test_mine_sacrificed_wins_nonexistent_target_is_a_noop(tmp_path: Path) -> None:
    module = _load_script()
    run = tmp_path / "run-empty"
    run.mkdir()
    out = tmp_path / "sacrificed.jsonl"
    summary = module.mine_sacrificed_wins(
        [run], out, target="uaf_heap-002", challenge_id="ctf-lictor-uaf_heap-002-001"
    )
    assert summary["records"] == 0
    assert not out.exists()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(
        (n, f)
        for n, f in vars(sys.modules[__name__]).items()
        if n.startswith("test_") and callable(f)
    ):
        try:
            fn(Path("/tmp") / f"test-mine-{name}")
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
