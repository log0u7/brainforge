"""Unit tests for the final benchmark harness (ADR-0021).

The harness reuses the pilot lictor lab loop (verification reads the live
service output through the per-target oracle regex), measures the success
rate over repetitions and audits contamination: training challenge IDs must
never appear in the eval set.
"""

import json
from pathlib import Path

import pytest

from brainforge.errors import BrainforgeError
from brainforge.experiments.bench_lab import run_bench_lab


def _target(challenge_id: str, binary: str, expected_flag: str) -> tuple:
    return (challenge_id, "127.0.0.1:9000", binary, "material", expected_flag)


def _record(challenge_id: str, model: str, flag: str | None, status: str = "done") -> dict:
    outcome: dict = {"status": status}
    if flag is not None:
        outcome["flag"] = flag
    return {
        "id": challenge_id,
        "challenge_id": challenge_id,
        "model": model,
        "outcome": outcome,
    }


def _fake_pilot(records_per_call: list[list[dict]]):
    """Injectable run_pilot double: returns the next canned records list."""
    calls: list[dict] = []

    def fake_run_pilot(targets, workdir, provider, model, **kwargs):
        calls.append(
            {
                "targets": list(targets),
                "workdir": workdir,
                "model": model,
            }
        )
        return records_per_call[len(calls) - 1]

    return fake_run_pilot, calls


def _challenges() -> list[tuple]:
    return [
        _target("chal-unseen-1", "chall1", "FLAG{secreta}"),
        _target("chal-unseen-2", "chall2", "FLAG{secretb}"),
    ]


class TestRunBenchLabOrchestration:
    def test_runs_base_and_trained_over_same_challenges_one_rep_each(self, tmp_path: Path) -> None:
        pilot, calls = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base-snapshot",
            trained_model="trained-snapshot",
            repetitions=1,
            training_ids=set(),
            pilot_fn=pilot,
        )
        assert [c["model"] for c in calls] == [
            "base-snapshot",
            "base-snapshot",
            "trained-snapshot",
            "trained-snapshot",
        ]
        # both models see the exact same unseen challenge list (one target per call)
        base_targets = [t[0] for c in calls if c["model"] == "base-snapshot" for t in c["targets"]]
        trained_targets = [
            t[0] for c in calls if c["model"] == "trained-snapshot" for t in c["targets"]
        ]
        assert base_targets == ["chal-unseen-1", "chal-unseen-2"]
        assert trained_targets == ["chal-unseen-1", "chal-unseen-2"]
        assert result["repetitions"] == 1
        assert result["challenges"] == ["chal-unseen-1", "chal-unseen-2"]

    def test_repetitions_run_the_challenge_list_n_times_per_model(self, tmp_path: Path) -> None:
        # 2 challenges x 3 reps = 6 calls per model, base first then trained
        pilot, calls = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=3,
            training_ids=set(),
            pilot_fn=pilot,
        )
        assert len(calls) == 12
        assert result["repetitions"] == 3
        # each challenge was attempted 3 times per model
        assert result["base"]["attempts"] == 6
        assert result["trained"]["attempts"] == 6


class TestSuccessRateOverRepetitions:
    def test_success_rate_counts_verified_flags_over_all_attempts(self, tmp_path: Path) -> None:
        # trained model: flag found on 2 of 3 attempts for chal-unseen-1,
        # never for chal-unseen-2 -> 2/6
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
                [_record("chal-unseen-1", "trained", "FLAG{wrong}")],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=3,
            training_ids=set(),
            pilot_fn=pilot,
        )
        assert result["base"]["successes"] == 0
        assert result["base"]["success_rate"] == 0.0
        assert result["trained"]["successes"] == 2
        assert result["trained"]["success_rate"] == pytest.approx(2 / 6)

    def test_success_is_verdict_not_raw_flag_string_match(self, tmp_path: Path) -> None:
        # a model reciting a training flag (wrong challenge) is NOT a success:
        # the record flag must equal the challenge's expected flag
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", "FLAG{from-training}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids=set(),
            pilot_fn=pilot,
        )
        assert result["base"]["successes"] == 0
        assert result["base"]["success_rate"] == 0.0

    def test_success_counts_per_challenge_over_repetitions(self, tmp_path: Path) -> None:
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", "FLAG{secretb}")],
                [_record("chal-unseen-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
                [_record("chal-unseen-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
                [_record("chal-unseen-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=3,
            training_ids=set(),
            pilot_fn=pilot,
        )
        per_challenge = result["base"]["per_challenge"]
        assert per_challenge["chal-unseen-1"] == {"attempts": 3, "successes": 2}
        assert per_challenge["chal-unseen-2"] == {"attempts": 3, "successes": 1}


class TestContaminationAudit:
    def test_training_ids_in_eval_set_fail_the_audit(self, tmp_path: Path) -> None:
        challenges = [
            _target("chal-train-1", "chall1", "FLAG{secreta}"),
            _target("chal-unseen-2", "chall2", "FLAG{secretb}"),
        ]
        pilot, _ = _fake_pilot(
            [
                [_record("chal-train-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-train-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=challenges,
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids={"chal-train-1", "chal-train-2"},
            pilot_fn=pilot,
        )
        assert result["contamination"]["clean"] is False
        assert result["contamination"]["training_ids_in_eval"] == ["chal-train-1"]

    def test_clean_eval_set_passes_the_audit(self, tmp_path: Path) -> None:
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids={"chal-train-1"},
            pilot_fn=pilot,
        )
        assert result["contamination"]["clean"] is True
        assert result["contamination"]["training_ids_in_eval"] == []

    def test_missing_training_ids_fails_fast(self, tmp_path: Path) -> None:
        pilot, calls = _fake_pilot([])
        with pytest.raises(BrainforgeError, match="training_ids"):
            run_bench_lab(
                challenges=_challenges(),
                workdir=tmp_path,
                provider=None,
                base_model="base",
                trained_model="trained",
                repetitions=1,
                pilot_fn=pilot,
            )
        assert calls == []


class TestOutputs:
    def test_writes_results_json_and_md(self, tmp_path: Path) -> None:
        output = tmp_path / "experiments" / "bench_lab"
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids=set(),
            output=output,
            pilot_fn=pilot,
        )
        payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
        assert payload["repetitions"] == 1
        assert payload["base"]["successes"] == 1
        assert payload["trained"]["successes"] == 1
        assert (output / "results.md").exists()
        md = (output / "results.md").read_text(encoding="utf-8")
        assert "| model |" in md
        assert "trained" in md

    def test_no_output_dir_skips_writes(self, tmp_path: Path) -> None:
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", None)],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", None)],
                [_record("chal-unseen-2", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids=set(),
            pilot_fn=pilot,
        )
        assert result["base"]["attempts"] == 2
        assert not (tmp_path / "results.json").exists()
