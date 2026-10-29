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
    # the three hand-written pack examples pre-date the corpus and stay
    # outside the manifest; everything else must be manifest-covered
    legacy = {"ctf-crypto-xor-001", "ctf-pwn-gets-001", "ctf-web-sqli-001"}
    assert manifest == disk - legacy, sorted(manifest ^ (disk - legacy))
    assert not (manifest - disk), sorted(manifest - disk)


def test_archive_case_includes_recon_for_elf_binaries(corpus, tmp_path):
    """A tier-B archive case whose attachment is an ELF binary must carry the
    recon output (file type, protections, disasm head) instead of only the
    'not excerpted' placeholder: the KD-GLM run trained on disasm-style
    prompts (lictor/rop cases use run_recon) and the holdout prompts were
    recon-style too, so the model learned to answer that distribution and
    fell back to markdown on prompt shapes it never saw."""
    from unittest import mock

    event = corpus / "ctf-archives" / "2026-Test-Event"
    attachments = event / "attachments"
    challenge = attachments / "elf_bin_0123456789abcdef0123456789abcdef"
    challenge.mkdir(parents=True)
    (challenge / "chall.bin").write_bytes(b"\x7fELF" + b"\x00" * 8)
    (event / "README.md").write_text("### Elf Bin\n\n> A statement.\n", encoding="utf-8")
    fake_recon = (
        "$ file -b chall.bin\nELF 64-bit LSB executable\n$ objdump -d (head)\n  400566: cmp"
    )
    events = {"2026-Test-Event": ("test", "2026-10-01")}
    with mock.patch.object(ingest, "run_recon", return_value=fake_recon):
        cases = ingest.ctf_archives_cases(corpus, events=events)
    case = next(
        c for c in cases if c[0]["id"].startswith("ctf-test-elf_bin_0123456789abcdef0123456789")
    )
    code = case[0]["input"]["code"]
    assert "$ file -b" in code
    assert "objdump -d" in code
    assert "not excerpted" not in code or code.count("not excerpted") < code.count("$ file -b") + 2


def test_archive_case_keeps_placeholder_for_non_elf_files(corpus, tmp_path):
    """Non-ELF attachments (raw dumps, images) keep the placeholder: recon
    tools cannot make sense of them."""
    from unittest import mock

    event = corpus / "ctf-archives" / "2026-Test-Event"
    attachments = event / "attachments"
    challenge = attachments / "raw_dump_0123456789abcdef0123456789abcdef"
    challenge.mkdir(parents=True)
    (challenge / "dump.raw").write_bytes(b"\x00\x01\x02\x03")
    (event / "README.md").write_text("### Raw Dump\n\n> A statement.\n", encoding="utf-8")
    events = {"2026-Test-Event": ("test", "2026-10-01")}
    with mock.patch.object(ingest, "run_recon") as recon:
        cases = ingest.ctf_archives_cases(corpus, events=events)
    case = next(
        c for c in cases if c[0]["id"].startswith("ctf-test-raw_dump_0123456789abcdef0123456789")
    )
    assert "not excerpted" in case[0]["input"]["code"]
    recon.assert_not_called()
