"""Unit tests for the final benchmark harness (ADR-0021).

The harness reuses the pilot lictor lab loop (verification reads the live
service output through the per-target oracle regex), measures the success
rate over repetitions and audits contamination: training challenge IDs must
never appear in the eval set.
"""

import json
import threading
from pathlib import Path

import pytest

from brainforge.errors import BrainforgeError
from brainforge.experiments.bench_lab import run_bench_lab


def _target(challenge_id: str, binary: str, expected_flag: str) -> tuple:
    return (challenge_id, "127.0.0.1:9000", binary, "material", expected_flag)


def _record(challenge_id: str, model: str, flag: str | None, status: str = "done") -> dict:
    """Alias on the real build_trajectory record shape: one contract
    between the tests and the pilot, no synthetic ``outcome`` dict."""
    return _pilot_record(challenge_id, model, flag, status)


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


def _pilot_record(challenge_id: str, model: str, flag: str | None, status: str = "done") -> dict:
    """The real build_trajectory record shape (pilot.py): the challenge id
    lives in ``id`` (there is no ``challenge_id`` key) and the flag lives
    in the final assistant verdict message, not in an ``outcome`` dict."""
    verdict: dict = {
        "verdict": "solved" if flag else status,
        "technique": "ret2plt",
        "flag": flag,
        "source": "lictor_run",
    }
    messages = [
        {"role": "user", "content": "Solve this CTF challenge against the local lab target."},
        {"role": "assistant", "content": json.dumps(verdict, ensure_ascii=False)},
    ]
    return {
        "id": challenge_id,
        "domain": "agentic",
        "messages": messages,
        "metadata": {
            "pipeline": "pilot",
            "domain": "agentic",
            "pilot": {"model": model, "decision": {"technique": "ret2plt"}},
            "flag_verified": bool(flag),
        },
    }


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

    def test_lab_namespace_training_ids_fail_the_audit(self, tmp_path: Path) -> None:
        # GRPO lab-loop training records carry ``lab-<technique>-001`` ids
        # while the eval manifest names the same lictor lab target
        # ``ctf-lictor-<technique>-001``: a raw string intersection always
        # comes back empty, certifying a contaminated eval as clean. The
        # audit must compare in the eval namespace.
        challenges = [_target("ctf-lictor-fmtstr-001", "fmtstr", "FLAG{a}")]
        pilot, _ = _fake_pilot(
            [
                [_record("ctf-lictor-fmtstr-001", "base", None)],
                [_record("ctf-lictor-fmtstr-001", "trained", None)],
            ]
        )
        result = run_bench_lab(
            challenges=challenges,
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids={"lab-fmtstr-001"},
            pilot_fn=pilot,
        )
        assert result["contamination"]["clean"] is False
        assert result["contamination"]["training_ids_in_eval"] == ["ctf-lictor-fmtstr-001"]

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


class TestRealPilotRecordShape:
    """Regression: the harness must read the real build_trajectory record
    shape - the challenge id in ``id`` and the flag in the final assistant
    verdict message - not the synthetic shape of the old doubles."""

    def test_real_records_key_per_challenge_and_count_successes(self, tmp_path: Path) -> None:
        pilot, _ = _fake_pilot(
            [
                [_pilot_record("chal-unseen-1", "base", None)],
                [_pilot_record("chal-unseen-2", "base", None)],
                [_pilot_record("chal-unseen-1", "trained", "FLAG{secreta}")],
                [_pilot_record("chal-unseen-2", "trained", None)],
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
        # the per-challenge slots must carry the real challenge ids, not null
        per_challenge = result["trained"]["per_challenge"]
        assert set(per_challenge) == {"chal-unseen-1", "chal-unseen-2"}
        assert per_challenge["chal-unseen-1"]["successes"] == 1
        assert per_challenge["chal-unseen-2"]["successes"] == 0
        assert result["trained"]["successes"] == 1
        assert result["trained"]["success_rate"] == pytest.approx(1 / 2)

    def test_real_records_wrong_flag_on_challenge_is_not_a_success(self, tmp_path: Path) -> None:
        # the verdict flag must match the per-target oracle flag: a recited
        # training flag on the wrong challenge is contamination, not a win
        pilot, _ = _fake_pilot(
            [
                [_pilot_record("chal-unseen-1", "trained", "FLAG{from-training}")],
                [_pilot_record("chal-unseen-2", "trained", None)],
                [_pilot_record("chal-unseen-1", "base", None)],
                [_pilot_record("chal-unseen-2", "base", None)],
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
        assert result["trained"]["successes"] == 0
        assert result["trained"]["per_challenge"]["chal-unseen-1"]["successes"] == 0
        assert result["trained"]["per_challenge"]["chal-unseen-1"]["attempts"] == 1


class TestBenchProgressLine:
    """The MR6-v2 hang measured the gap: a hung challenge is invisible
    while the bench runs (the protocol pipes the output through
    ``tail -45``). One progress line per (model, rep, challenge) launch
    and one per outcome - flushed - makes the oldest in-flight run
    visible in the log instead of the trails."""

    def test_prints_a_progress_line_per_launch_and_outcome(self, tmp_path: Path, capsys) -> None:
        pilot, _ = _fake_pilot(
            [
                [_record("chal-unseen-1", "base", "FLAG{secreta}")],
                [_record("chal-unseen-2", "base", None)],
                [_record("chal-unseen-1", "trained", None)],
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
            pilot_fn=pilot,
        )
        out = capsys.readouterr().out
        # every launch and every outcome is announced, flushed
        assert "[bench base rep 1/1] chal-unseen-1" in out
        assert "[bench base rep 1/1] chal-unseen-2" in out
        assert "[bench trained rep 1/1] chal-unseen-1" in out
        assert "[bench trained rep 1/1] chal-unseen-2" in out
        # the outcome lines carry the elapsed time (the hang detector)
        assert "elapsed" in out
        # the flags are visible on the outcome line
        assert "FLAG" in out


class TestBenchRepetitionSeed:
    """The bench never passed a seed to the pilot, so every repetition ran
    run_pilot's default seed 7 and lictor replayed the same bandit draws:
    N repetitions were N copies of one trajectory, not N independent
    samples. Each repetition must carry its own seed."""

    def test_each_repetition_gets_its_own_seed(self, tmp_path: Path) -> None:
        seen: list[tuple[str, str | None]] = []

        def pilot(targets, workdir, provider, model, **kwargs):
            seen.append((model, kwargs.get("seed")))
            return [_pilot_record(targets[0][0], model, None)]

        run_bench_lab(
            challenges=[_target("chal-unseen-1", "chall1", "FLAG{secreta}")],
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=3,
            training_ids=set(),
            pilot_fn=pilot,
        )
        # same rep index -> same seed across models (paired comparison),
        # consecutive reps -> consecutive seeds (independent draws)
        assert seen == [
            ("base", "7"),
            ("base", "8"),
            ("base", "9"),
            ("trained", "7"),
            ("trained", "8"),
            ("trained", "9"),
        ]

    def test_seed_base_is_configurable(self, tmp_path: Path) -> None:
        seen: list[str | None] = []

        def pilot(targets, workdir, provider, model, **kwargs):
            seen.append(kwargs.get("seed"))
            return [_pilot_record(targets[0][0], model, None)]

        run_bench_lab(
            challenges=[_target("chal-unseen-1", "chall1", "FLAG{secreta}")],
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=2,
            training_ids=set(),
            pilot_fn=pilot,
            seed_base=100,
        )
        assert seen == ["100", "101", "100", "101"]


class TestBenchAttemptWatchdog:
    """The MR6-v2 re-bench froze for 67 minutes: a hung attempt blocked
    the whole bench (the provider guard never fired, nothing bounded the
    attempt). Every attempt runs under a wall-clock cap: a blocked
    attempt becomes an explicit TIMEOUT miss and the bench moves on to
    the next challenge."""

    def test_blocked_attempt_times_out_and_bench_continues(self, tmp_path: Path, capsys):
        release = threading.Event()
        calls: list[str] = []

        def hung_pilot(targets, workdir, provider, model, **kwargs):
            challenge_id = targets[0][0]
            calls.append(challenge_id)
            if challenge_id == "chal-unseen-1":
                release.wait()  # simulates the gpu3080 freeze
            return [_pilot_record(challenge_id, model, None)]

        run_bench_lab(
            challenges=_challenges(),
            workdir=tmp_path,
            provider=None,
            base_model="base",
            trained_model="trained",
            repetitions=1,
            training_ids=set(),
            pilot_fn=hung_pilot,
            attempt_cap=0.5,
        )
        out = capsys.readouterr().out
        # the bench survived the blocked attempt and ran every next one
        assert calls == ["chal-unseen-1", "chal-unseen-2", "chal-unseen-1", "chal-unseen-2"]
        # the blocked attempt is announced as a TIMEOUT miss, never silent
        assert "[bench base rep 1/1] chal-unseen-1: TIMEOUT" in out
