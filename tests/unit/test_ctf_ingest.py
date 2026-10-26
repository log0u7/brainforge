"""Ingest tests: content-root resolution, dedup, statement matching, ids.

experiments/ctf_ingest is a script, not a package module: the tests load
it by file path. The fixture corpus exercises the three extraction
layouts seen in the wild (wrapper dirs, nested zips, flat copies).
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ctf_ingest", Path(__file__).resolve().parents[2] / "experiments" / "ctf_ingest" / "ingest.py"
)
ingest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _make_case_file(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def corpus(tmp_path: Path):
    """Three events; each challenge appears in the layouts seen in the wild."""
    base = tmp_path / "ctf-archives"
    event = base / "2026-Test-Event"
    attachments = event / "attachments"
    attachments.mkdir(parents=True)

    # symlinked wrapper + real content dir: must dedup to one case
    _make_case_file(attachments / "Mosaic", {"attachments/chall.txt": "x" * 40})
    (attachments / "ROM.Mosaic").symlink_to(attachments / "Mosaic", target_is_directory=True)

    # hash-suffixed dir name vs README heading "### Cold Forge"
    _make_case_file(
        attachments / "cold_forge_4b1bc2963947358f4ce48f769581a096",
        {"attachments/readme.txt": "pwn me"},
    )
    (event / "README.md").write_text(
        "### Cold Forge\n\n> A statement about the cold forge challenge.\nflag格式TEST{abc}\n",
        encoding="utf-8",
    )
    # the ingest appends "ctf-archives" itself: corpus = its parent
    return tmp_path


def test_content_root_dedup_merges_layouts(corpus, tmp_path):
    cases = ingest.ctf_archives_cases(corpus, events={"2026-Test-Event": ("test", "2026-10-01")})
    mosaic = [c for c, _ in cases if "mosaic" in c["id"]]
    assert len(mosaic) == 1
    assert mosaic[0]["id"] == "ctf-test-rom-mosaic-001"  # longest name wins


def test_statement_hash_suffix_stripped(corpus, tmp_path):
    cases = ingest.ctf_archives_cases(corpus, events={"2026-Test-Event": ("test", "2026-10-01")})
    forge = [c for c, _ in cases if "cold_forge" in c["id"]]
    assert len(forge) == 1
    record = forge[0]
    assert "cold forge challenge" in record["input"]["description"]
    assert record["metadata"]["flag_format"] == "TEST{abc}"


def test_content_root_terminates_on_cycle(tmp_path):
    attachments = tmp_path / "attachments"
    loop_a = attachments / "loop_a"
    loop_b = attachments / "loop_b"
    loop_a.mkdir(parents=True)
    loop_b.mkdir(parents=True)
    (loop_a / "inner").symlink_to(loop_b)
    (loop_b / "inner").symlink_to(loop_a)
    root = ingest._content_root(loop_a)
    assert root.is_dir()


def test_case_ids_unique_and_valid(corpus, tmp_path):
    cases = [case for case, _ in ingest.ctf_archives_cases(corpus)]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for cid in ids:
        assert _ID_RE.match(cid), cid


def test_statementless_challenge_falls_back_to_canonical_name(tmp_path):
    base = tmp_path / "ctf-archives"
    event = base / "2026-Test-Event"
    attachments = event / "attachments"
    (attachments / "mystery_box" / "attachments").mkdir(parents=True)
    (attachments / "mystery_box" / "attachments" / "evidence.bin").write_text("z" * 30)
    # no README at all: the statement falls back to the directory name
    cases = ingest.ctf_archives_cases(tmp_path, events={"2026-Test-Event": ("test", "2026-10-01")})
    assert len(cases) == 1
    record, _ = cases[0]
    assert "mystery_box" in record["input"]["description"]
    assert record["metadata"]["flag_format"] is None


def test_committed_manifest_matches_committed_cases():
    """Regression guard: every committed ctf-*.json must have exactly one
    manifest line and vice versa (the orphan class found by review)."""
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        json.loads(line)["id"]
        for line in (repo / "experiments" / "ctf_ingest" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    disk = {p.stem for p in (repo / "examples" / "cases").glob("ctf-*.json")}
    assert manifest == disk, sorted(manifest ^ disk)
