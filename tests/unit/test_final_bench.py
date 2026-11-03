"""Unit tests for the MR5 final benchmark runner (ADR-0021).

The runner assembles the real benchmark inputs from repo artifacts:
the oracle-verified challenge cases (manifest + case files) mapped onto
the live lictor lab targets, and the training challenge IDs that feed
the contamination audit. The provider split routes each model to its
own llama-server (llama.cpp serves one model per instance).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from brainforge.experiments.final_bench import (
    LICTOR_LAB_PORTS,
    SplitProvider,
    load_training_ids,
    load_verified_challenges,
)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _write_case(cases_dir: Path, case_id: str, expected_flag: str) -> None:
    case = {
        "id": case_id,
        "source": {"type": "lictor-lab", "path": f"lab/targets/{case_id.split('-')[-2]}"},
        "input": {"code": f"source code of {case_id}"},
        "metadata": {"tier": "verified", "expected_flag": expected_flag},
    }
    (cases_dir / f"{case_id}.json").write_text(json.dumps(case), encoding="utf-8")


def _manifest_row(case_id: str, source: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "source": source,
        "tier": "verified",
        "expected_flag": True,
    }


class TestLoadVerifiedChallenges:
    def test_maps_live_lab_targets_to_tuples(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(
            manifest,
            [
                _manifest_row("ctf-lictor-fmtstr-001", "lictor-lab"),
                _manifest_row("ctf-rop-ret2win-001", "rop-emporium"),
            ],
        )
        cases = tmp_path / "cases"
        cases.mkdir()
        _write_case(cases, "ctf-lictor-fmtstr-001", "FLAG{fmt}")

        challenges, skipped = load_verified_challenges(manifest, cases)

        assert challenges == [
            (
                "ctf-lictor-fmtstr-001",
                f"127.0.0.1:{LICTOR_LAB_PORTS['fmtstr']}",
                "fmtstr",
                "source code of ctf-lictor-fmtstr-001",
                "FLAG{fmt}",
            )
        ]
        assert skipped == ["ctf-rop-ret2win-001"]

    def test_skips_unverified_tier(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(
            manifest,
            [
                _manifest_row("ctf-lictor-fmtstr-001", "lictor-lab"),
                {
                    "id": "ctf-ccb-somecase-001",
                    "source": "ctf-archives",
                    "tier": "tier-b",
                    "expected_flag": False,
                },
            ],
        )
        cases = tmp_path / "cases"
        cases.mkdir()
        _write_case(cases, "ctf-lictor-fmtstr-001", "FLAG{fmt}")

        challenges, skipped = load_verified_challenges(manifest, cases)

        assert len(challenges) == 1
        assert skipped == ["ctf-ccb-somecase-001"]

    def test_missing_case_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(
            manifest,
            [
                _manifest_row("ctf-lictor-fmtstr-001", "lictor-lab"),
                _manifest_row("ctf-lictor-orw-001", "lictor-lab"),
            ],
        )
        cases = tmp_path / "cases"
        cases.mkdir()
        _write_case(cases, "ctf-lictor-fmtstr-001", "FLAG{fmt}")

        challenges, skipped = load_verified_challenges(manifest, cases)

        assert len(challenges) == 1
        assert skipped == ["ctf-lictor-orw-001"]

    def test_ret2csu_maps_to_lab_port_9992(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [_manifest_row("ctf-lictor-ret2csu-001", "lictor-lab")])
        cases = tmp_path / "cases"
        cases.mkdir()
        case = {
            "id": "ctf-lictor-ret2csu-001",
            "source": {"type": "lictor-lab", "path": "lab/targets/ret2csu"},
            "input": {"code": "recon material"},
            "metadata": {"tier": "verified", "expected_flag": "ROPE{x}"},
        }
        (cases / "ctf-lictor-ret2csu-001.json").write_text(json.dumps(case), encoding="utf-8")

        challenges, skipped = load_verified_challenges(manifest, cases)

        assert challenges == [
            (
                "ctf-lictor-ret2csu-001",
                "127.0.0.1:9992",
                "ret2csu",
                "recon material",
                "ROPE{x}",
            )
        ]
        assert skipped == []

    @pytest.mark.parametrize(
        ("binary", "port"),
        [
            ("ret2libc-001", 9982),
            ("ret2libc-002", 9983),
            ("fmtstr-002", 9984),
            ("ret2func-002", 9985),
            ("shellcode-002", 9986),
            ("uaf_heap-002", 9987),
            ("orw-002", 9988),
            ("callme-002", 9989),
            ("ret2plt-002", 9990),
            ("srop-001", 9991),
        ],
    )
    def test_mr5_scaleup_targets_map_to_lab_ports(
        self, tmp_path: Path, binary: str, port: int
    ) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [_manifest_row(f"ctf-lictor-{binary}-001", "lictor-lab")])
        cases = tmp_path / "cases"
        cases.mkdir()
        case = {
            "id": f"ctf-lictor-{binary}-001",
            "source": {"type": "lictor-lab", "path": f"lab/targets/{binary}"},
            "input": {"code": "source material"},
            "metadata": {"tier": "verified", "expected_flag": "FLAG{x}"},
        }
        (cases / f"ctf-lictor-{binary}-001.json").write_text(json.dumps(case), encoding="utf-8")

        challenges, skipped = load_verified_challenges(manifest, cases)

        assert challenges == [
            (f"ctf-lictor-{binary}-001", f"127.0.0.1:{port}", binary, "source material", "FLAG{x}")
        ]
        assert skipped == []

    def test_no_live_target_for_binary_is_skipped(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(
            manifest,
            [_manifest_row("ctf-lictor-unknownbin-001", "lictor-lab")],
        )
        cases = tmp_path / "cases"
        cases.mkdir()
        # the case file names a binary that the live lab does not serve
        case = {
            "id": "ctf-lictor-unknownbin-001",
            "source": {"type": "lictor-lab", "path": "lab/targets/unknownbin"},
            "input": {"code": "material"},
            "metadata": {"tier": "verified", "expected_flag": "FLAG{x}"},
        }
        (cases / "ctf-lictor-unknownbin-001.json").write_text(json.dumps(case), encoding="utf-8")

        challenges, skipped = load_verified_challenges(manifest, cases)

        assert challenges == []
        assert skipped == ["ctf-lictor-unknownbin-001"]


class TestCommittedWiring:
    def test_committed_ret2csu_challenge_is_runnable(self) -> None:
        """The 7th lictor lab target is wired: the committed manifest line
        plus case file map onto the live ret2csu listener (9992) instead
        of landing in the skipped list."""
        repo = Path(__file__).resolve().parents[2]
        challenges, skipped = load_verified_challenges(
            repo / "experiments" / "ctf_ingest" / "manifest.jsonl",
            repo / "examples" / "cases",
        )
        by_id = {c[0]: c for c in challenges}
        assert "ctf-lictor-ret2csu-001" in by_id
        entry = by_id["ctf-lictor-ret2csu-001"]
        assert entry[1] == "127.0.0.1:9992"
        assert entry[2] == "ret2csu"
        assert entry[4]
        assert "ctf-lictor-ret2csu-001" not in skipped

    @pytest.mark.parametrize(
        ("binary", "port"),
        [
            ("ret2libc-001", 9982),
            ("ret2libc-002", 9983),
            ("fmtstr-002", 9984),
            ("ret2func-002", 9985),
            ("shellcode-002", 9986),
            ("uaf_heap-002", 9987),
            ("orw-002", 9988),
            ("callme-002", 9989),
            ("ret2plt-002", 9990),
            ("srop-001", 9991),
        ],
    )
    def test_committed_mr5_scaleup_challenges_are_runnable(self, binary: str, port: int) -> None:
        """The MR5 scale-up targets are wired: each committed manifest
        line plus case file maps onto its live lab listener instead of
        landing in the skipped list."""
        repo = Path(__file__).resolve().parents[2]
        challenges, skipped = load_verified_challenges(
            repo / "experiments" / "ctf_ingest" / "manifest.jsonl",
            repo / "examples" / "cases",
        )
        by_id = {c[0]: c for c in challenges}
        case_id = f"ctf-lictor-{binary}-001"
        assert case_id in by_id
        entry = by_id[case_id]
        assert entry[1] == f"127.0.0.1:{port}"
        assert entry[2] == binary
        assert entry[4]
        assert case_id not in skipped

    def test_committed_eval_set_has_seventeen_live_targets(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        challenges, _skipped = load_verified_challenges(
            repo / "experiments" / "ctf_ingest" / "manifest.jsonl",
            repo / "examples" / "cases",
        )
        lictor_ids = {c[0] for c in challenges if c[0].startswith("ctf-lictor-")}
        assert lictor_ids == {
            "ctf-lictor-callme-002-001",
            "ctf-lictor-fmtstr-001",
            "ctf-lictor-fmtstr-002-001",
            "ctf-lictor-orw-001",
            "ctf-lictor-orw-002-001",
            "ctf-lictor-ret2csu-001",
            "ctf-lictor-ret2func-001",
            "ctf-lictor-ret2func-002-001",
            "ctf-lictor-ret2libc-001-001",
            "ctf-lictor-ret2libc-002-001",
            "ctf-lictor-ret2plt-001",
            "ctf-lictor-ret2plt-002-001",
            "ctf-lictor-shellcode-001",
            "ctf-lictor-shellcode-002-001",
            "ctf-lictor-srop-001-001",
            "ctf-lictor-uaf_heap-001",
            "ctf-lictor-uaf_heap-002-001",
        }


class TestLoadTrainingIds:
    def test_collects_ids_from_jsonl_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text('{"id": "x"}\n{"id": "y"}\n', encoding="utf-8")
        b.write_text('{"id": "y"}\n{"id": "z"}\n', encoding="utf-8")

        assert load_training_ids(a, b) == {"x", "y", "z"}

    def test_empty_file_yields_no_ids(self, tmp_path: Path) -> None:
        a = tmp_path / "empty.jsonl"
        a.write_text("", encoding="utf-8")

        assert load_training_ids(a) == set()


class TestSplitProvider:
    def test_routes_by_model_name(self) -> None:
        class _FakeProvider:
            def __init__(self, tag: str) -> None:
                self.tag = tag
                self.calls: list[str] = []

            def complete(self, request: Any, model: str) -> str:
                self.calls.append(model)
                return self.tag

        base = _FakeProvider("base")
        trained = _FakeProvider("trained")
        provider = SplitProvider(providers={"base-snapshot": base, "trained-snapshot": trained})

        assert provider.complete(object(), "base-snapshot") == "base"
        assert provider.complete(object(), "trained-snapshot") == "trained"
        assert base.calls == ["base-snapshot"]
        assert trained.calls == ["trained-snapshot"]

    def test_unknown_model_fails_loudly(self) -> None:
        class _FakeProvider:
            def complete(self, request: Any, model: str) -> str:
                return "ok"

        provider = SplitProvider(providers={"base": _FakeProvider()})
        with pytest.raises(KeyError):
            provider.complete(object(), "not-a-known-model")
