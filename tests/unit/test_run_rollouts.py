"""Tests for the MR6 rollout runner.

``scripts/run_rollouts.py`` drives the SFT dataset expansion: it maps
the oracle-verified manifest onto the TRAINING-ELIGIBLE lab targets
only (the 6 leaked originals; eval challenges never become training
rollouts) and runs the pilot once per seed so the bandit draws differ
across repetitions.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_rollouts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_rollouts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_and_cases(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    lines = [
        {
            "source": "lictor-lab",
            "id": "ctf-lictor-fmtstr-001",
            "tier": "verified",
            "expected_flag": True,
        },
        {
            "source": "lictor-lab",
            "id": "ctf-lictor-uaf_heap-001",
            "tier": "verified",
            "expected_flag": True,
        },
        {
            "source": "lictor-lab",
            "id": "ctf-lictor-orw-001",
            "tier": "verified",
            "expected_flag": True,
        },
    ]
    manifest.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    cases = tmp_path / "cases"
    cases.mkdir()
    for case_id, binary in (
        ("ctf-lictor-fmtstr-001", "fmtstr"),
        ("ctf-lictor-uaf_heap-001", "uaf_heap"),
        ("ctf-lictor-orw-001", "orw"),
    ):
        (cases / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "source": {"path": f"/opt/target/{binary}"},
                    "input": {"code": "int main(){}"},
                    "metadata": {"expected_flag": f"FLAG{{{binary}}}"},
                }
            ),
            encoding="utf-8",
        )
    return manifest, cases


def test_rollout_targets_exclude_eval_challenges(tmp_path: Path) -> None:
    """uaf_heap is a clean-protocol eval challenge: its case never
    becomes a training rollout, whatever the manifest carries."""
    module = _load_script()
    manifest, cases = _manifest_and_cases(tmp_path)
    targets = module.build_rollout_targets(manifest, cases)
    binaries = {t[2] for t in targets}
    assert binaries == {"fmtstr", "orw"}
    assert all("uaf_heap" not in t[0] for t in targets)


def test_rollout_sweep_runs_one_seed_per_repetition(tmp_path: Path) -> None:
    module = _load_script()
    manifest, cases = _manifest_and_cases(tmp_path)
    targets = module.build_rollout_targets(manifest, cases)

    seen: list[tuple[str, str | None]] = []

    def fake_pilot(targets_list, workdir, provider, model, output=None, seed="7"):
        seen.append((seed, output.name if output else None))
        return [{"id": "x", "messages": [], "metadata": {}}]

    out = tmp_path / "rollouts"
    module.run_rollouts(
        targets=targets,
        workdir=tmp_path / "work",
        provider=object(),
        model="base-qwen3-1.7b",
        repetitions=3,
        seed_start=10,
        pilot_fn=fake_pilot,
        out_dir=out,
    )
    assert [s for s, _ in seen] == ["10", "11", "12"]
    assert all(name and name.startswith("trajectories-seed-") for _, name in seen)
