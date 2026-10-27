import json

import pytest
from typer.testing import CliRunner

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.cli import app
from brainforge.dataset.writer import DatasetRecord
from brainforge.pipeline.engine import CaseResult, run_pipeline
from brainforge.providers.base import ChatMessage

runner = CliRunner()


def _record(cid: str) -> DatasetRecord:
    return DatasetRecord(
        id=cid,
        domain="security",
        messages=[
            ChatMessage(role="user", content="analyze"),
            ChatMessage(role="assistant", content="{}"),
        ],
        metadata={},
    )


class FakePipeline:
    def __init__(self, concurrency: int = 1):
        self.concurrency = concurrency


class FakeEngine:
    """Minimal engine double: run_pipeline only needs run_case, pipeline,
    pipeline_name (write_rejected is only reached with rejected_dir set).
    Results are keyed by case id so parallel execution stays deterministic."""

    def __init__(self, results: list[CaseResult], state_path=None, concurrency: int = 1):
        self.pipeline = FakePipeline(concurrency)
        self.pipeline_name = "test_pipeline"
        self.results_by_id = {result.case_id: result for result in results}
        self.state_path = state_path
        self.snapshots: dict[str, dict] = {}

    def run_case(self, case: Case) -> CaseResult:
        if self.state_path is not None:
            self.snapshots[case.id] = json.loads(self.state_path.read_text(encoding="utf-8"))
        return self.results_by_id[case.id]


def _case(cid: str) -> Case:
    return Case(
        id=cid,
        source=CaseSource(type="manual"),
        input=CaseInput(code="x = 1", description="desc"),
        metadata={"language": "python"},
    )


def _accepted(cid: str) -> CaseResult:
    return CaseResult(case_id=cid, accepted=True, record=_record(cid))


def _rejected(cid: str, reasons: list[str]) -> CaseResult:
    return CaseResult(case_id=cid, accepted=False, gate_reasons=reasons)


def _error(cid: str, reason: str = "boom") -> CaseResult:
    return CaseResult(
        case_id=cid,
        accepted=False,
        gate_reasons=[f"provider_error: judge: {reason}"],
        provider_error=reason,
    )


def test_state_file_tracks_final_counts_and_statuses(tmp_path):
    output = tmp_path / "out.jsonl"
    state_path = tmp_path / "out.state.json"
    engine = FakeEngine(
        [
            _accepted("c1"),
            _rejected("c2", ["low_confidence", "unresolved_disagreement"]),
            _error("c3"),
        ],
        state_path=state_path,
    )
    stats = run_pipeline(
        engine, [_case("c1"), _case("c2"), _case("c3")], output, state_path=state_path
    )
    assert stats["accepted"] == 1
    assert stats["errors"] == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "done"
    assert state["total_cases"] == 3
    assert state["done"] == 3
    assert state["accepted"] == 1
    assert state["rejected"] == 1
    assert state["errors"] == 1
    assert state["current_case"] is None
    assert state["started_at"]
    assert state["updated_at"]
    assert [(c["id"], c["status"]) for c in state["cases"]] == [
        ("c1", "accepted"),
        ("c2", "rejected"),
        ("c3", "error"),
    ]
    assert state["cases"][1]["gate_reasons"] == 2
    assert all(isinstance(c["duration_ms"], int) for c in state["cases"])


def test_state_marks_cases_running_before_each_case(tmp_path):
    output = tmp_path / "out.jsonl"
    state_path = tmp_path / "out.state.json"
    engine = FakeEngine(
        [_accepted("c1"), _rejected("c2", ["low_confidence"])], state_path=state_path
    )
    run_pipeline(engine, [_case("c1"), _case("c2")], output, state_path=state_path)
    first = engine.snapshots["c1"]
    assert first["current_case"] == "c1"
    entry = next(c for c in first["cases"] if c["id"] == "c1")
    assert entry["status"] == "running"
    second = engine.snapshots["c2"]
    assert second["done"] == 1
    assert second["current_case"] == "c2"
    assert next(c for c in second["cases"] if c["id"] == "c1")["status"] == "accepted"


def test_state_file_defaults_next_to_output_and_leaves_no_tmp(tmp_path):
    output = tmp_path / "datasets" / "sub" / "out.jsonl"
    run_pipeline(FakeEngine([_accepted("c1")]), [_case("c1")], output)
    state_path = tmp_path / "datasets" / "sub" / "out.jsonl.state.json"
    assert state_path.is_file()
    leftovers = [p.name for p in tmp_path.rglob("*.tmp")]
    assert leftovers == [], leftovers


def test_state_file_stays_consistent_in_parallel_mode(tmp_path):
    output = tmp_path / "out.jsonl"
    state_path = tmp_path / "out.state.json"
    engine = FakeEngine(
        [_accepted("c1"), _accepted("c2"), _error("c3")],
        state_path=state_path,
        concurrency=3,
    )
    stats = run_pipeline(
        engine, [_case("c1"), _case("c2"), _case("c3")], output, state_path=state_path
    )
    assert stats["accepted"] == 2
    assert stats["errors"] == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "done"
    assert state["done"] == 3
    assert state["accepted"] == 2
    assert state["errors"] == 1
    assert state["current_case"] is None
    assert {c["id"]: c["status"] for c in state["cases"]} == {
        "c1": "accepted",
        "c2": "accepted",
        "c3": "error",
    }
    # every case was visible as running the moment its execution started
    for cid, snapshot in engine.snapshots.items():
        assert next(c for c in snapshot["cases"] if c["id"] == cid)["status"] == "running"
    assert [p.name for p in tmp_path.rglob("*.tmp")] == []


def test_state_records_error_when_a_case_crashes(tmp_path):
    output = tmp_path / "out.jsonl"
    state_path = tmp_path / "out.state.json"

    class ExplodingEngine(FakeEngine):
        def run_case(self, case: Case) -> CaseResult:
            if case.id == "c2":
                raise RuntimeError("boom")
            return super().run_case(case)

    engine = ExplodingEngine([_accepted("c1"), _accepted("c2")], state_path=state_path)
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(engine, [_case("c1"), _case("c2")], output, state_path=state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "running"  # crash: final write never happened
    assert next(c for c in state["cases"] if c["id"] == "c2")["status"] == "error"


def _done_state(cid: str = "c1") -> dict:
    return {
        "started_at": "2026-10-27T10:00:00+00:00",
        "updated_at": "2026-10-27T10:00:01+00:00",
        "status": "done",
        "total_cases": 1,
        "done": 1,
        "accepted": 1,
        "rejected": 0,
        "errors": 0,
        "current_case": None,
        "cases": [{"id": cid, "status": "accepted", "duration_ms": 12, "gate_reasons": 0}],
    }


def test_cli_status_prints_state_from_dataset_path(tmp_path):
    (tmp_path / "out.jsonl.state.json").write_text(json.dumps(_done_state()), encoding="utf-8")
    result = runner.invoke(app, ["pipeline", "status", str(tmp_path / "out.jsonl")])
    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert "1/1" in result.output
    assert "c1" in result.output


def test_cli_status_missing_state_file_exits_cleanly(tmp_path):
    result = runner.invoke(app, ["pipeline", "status", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 1
    combined = result.output
    if not combined:
        combined = result.stderr
    assert "no state file" in combined
