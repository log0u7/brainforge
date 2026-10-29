#!/usr/bin/env python3
"""Build BrainForge CTF cases from the local CTF corpus.

Corpus layout (default ``~/projets/datasets/ctf-corpus``):

* ``lictor-lab/<target>/``: ``service.c`` plus ``flag.txt`` (tier A:
  expected flag known, material is ours and recent).
* ``rop-emporium/rop_emporium_all_challenges/<name>/``: ELF binary plus
  ``flag.txt`` (tier A). Challenges shipping an ``encrypted_flag.dat``
  are skipped: their plaintext flag is not locally verifiable.
* ``ctf-archives/<event>/<challenge>/``: public CTF dumps (tier B: no
  expected flag, the statement and its flag format stay in metadata).

Tier A cases carry ``metadata.expected_flag`` for the flag gate; tier B
cases carry ``metadata.flag_format`` only. Writeups are verification
references, never teacher inputs: none of the generated cases embeds a
writeup.

Outputs:

* ``examples/cases/<id>.json`` case files (committed).
* ``experiments/ctf_ingest/manifest.jsonl`` one line per case with its
  verification status (committed).

Usage: ``python experiments/ctf_ingest/ingest.py [--corpus DIR] [--repo DIR]``
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

CORPUS_DEFAULT = Path.home() / "projets" / "datasets" / "ctf-corpus"
REPO_DEFAULT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".pem",
    ".csv",
    ".md",
    ".c",
    ".py",
    ".conf",
}
EXCERPT_CHARS = 1500
DISASM_LINES = 60

# ROP Emporium launched its challenge set before every teacher cutoff used
# in BrainForge configs; the date is deliberately early so the
# recitation-risk tagging sees the material as pre-cutoff.
ROP_EMPORIUM_DATE = "2020-01-01"
ZEKR_RELEASE_DATE = "2026-09-19"
LICTOR_LAB_DATE = "2026-10-01"


def run_recon(binary: Path) -> str:
    """Bounded static recon text: file, ELF header, protections, disasm head."""
    parts = [f"$ file -b {binary.name}", _out(["file", "-b", str(binary)])]
    header = _out(["readelf", "-hW", str(binary)])
    for key in ("Class:", "Machine:", "Type:"):
        for line in header.splitlines():
            if key in line:
                parts.append(line.strip())
    stack = _out(["readelf", "-lW", str(binary)])
    nx = "GNU_STACK" in stack and "RWE" in [
        seg.split()[6] for seg in stack.splitlines() if "GNU_STACK" in seg and len(seg.split()) > 6
    ]
    parts.append(f"NX: {'disabled' if nx else 'enabled'}")
    dyn = _out(["readelf", "-dW", str(binary)])
    parts.append(f"BIND_NOW: {'yes' if 'BIND_NOW' in dyn else 'no'}")
    type_line = next((line for line in header.splitlines() if "Type:" in line), "")
    pie = "DYN" in type_line
    parts.append(f"PIE: {'yes' if pie else 'no'}")
    symbols = _out(["nm", str(binary)]) if binary.exists() else ""
    interesting = [
        line
        for line in symbols.splitlines()
        if any(token in line for token in ("win", "system", "backdoor", "shell"))
    ]
    for line in interesting[:6]:
        parts.append(line)
    disasm = _out(["objdump", "-d", str(binary)])
    parts.append("$ objdump -d (head)")
    parts.extend(disasm.splitlines()[:DISASM_LINES])
    return "\n".join(parts)[:4000]


def _out(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def read_text(path: Path, limit: int = EXCERPT_CHARS) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def case_json(case: dict) -> str:
    return json.dumps(case, ensure_ascii=False, indent=2) + "\n"


def lictor_cases(corpus: Path) -> list[tuple[dict, dict]]:
    cases = []
    for target in sorted((corpus / "lictor-lab").iterdir()):
        source = target / "service.c"
        flag = target / "flag.txt"
        if not source.is_file() or not flag.is_file():
            continue
        flag_value = read_text(flag, 200).strip()
        case_id = f"ctf-lictor-{target.name}-001"
        case = {
            "id": case_id,
            "source": {
                "type": "lictor-lab",
                "repository": "log0u7/lictor lab",
                "path": f"lab/targets/{target.name}",
                "date": LICTOR_LAB_DATE,
            },
            "input": {
                "code": read_text(source),
                "description": (
                    f"Binary exploitation lab target '{target.name}' (local service, "
                    "source provided, compiled with stack protections per its header "
                    "comment). A flag file is read by the service when the challenge "
                    "is solved. Solve this CTF challenge."
                ),
                "question": "Solve this CTF challenge.",
            },
            "metadata": {
                "language": "c",
                "category": "pwn",
                "tier": "verified",
                "expected_flag": flag_value,
                "expected_technique": target.name,
            },
        }
        manifest = {
            "id": case_id,
            "tier": "verified",
            "expected_flag": bool(flag_value),
            "expected_technique": True,
            "url": "https://github.com/log0u7/lictor",
        }
        cases.append((case, manifest))
    return cases


def rop_emporium_cases(corpus: Path) -> list[tuple[dict, dict]]:
    base = corpus / "rop-emporium" / "rop_emporium_all_challenges"
    cases = []
    for challenge in sorted(base.iterdir()) if base.is_dir() else []:
        binary = challenge / challenge.name
        flag = challenge / "flag.txt"
        if not binary.is_file() or not flag.is_file():
            continue
        flag_value = read_text(flag, 200).strip()
        if not flag_value:
            continue
        case_id = f"ctf-rop-{challenge.name}-001"
        case = {
            "id": case_id,
            "source": {
                "type": "rop-emporium",
                "repository": "ropemporium.com",
                "path": f"challenges/{challenge.name}",
                "date": ROP_EMPORIUM_DATE,
                "url": f"https://ropemporium.com/challenge/{challenge.name}.html",
            },
            "input": {
                "code": run_recon(binary),
                "description": (
                    f"ROP Emporium '{challenge.name}' challenge binary (x86_64 Linux). "
                    "Recon output below. Solve this CTF challenge."
                ),
                "question": "Solve this CTF challenge.",
            },
            "metadata": {
                "language": "c",
                "category": "pwn",
                "tier": "verified",
                "expected_flag": flag_value,
                "expected_technique": challenge.name,
            },
        }
        manifest = {
            "id": case_id,
            "tier": "verified",
            "expected_flag": bool(flag_value),
            "expected_technique": challenge.name,
            "url": f"https://ropemporium.com/challenge/{challenge.name}.html",
        }
        cases.append((case, manifest))
    return cases


CTF_ARCHIVE_EVENTS = {
    "2026-Zeekr-Cybersecurity-Quals": ("zeekr", ZEKR_RELEASE_DATE),
    "2026-WQB-Quals": ("wqb", "2026-09-04"),
    "CCB2026-Quals": ("ccb", "2026-09-13"),
}


def _content_root(challenge: Path) -> Path:
    """Descend single-child wrapper dirs until real content appears.

    Symlink cycles terminate: revisited (resolved) paths stop the descent.
    """
    current = challenge
    visited = {current.resolve()}
    while True:
        children = [c for c in current.iterdir()]
        dirs = [c for c in children if c.is_dir()]
        files = [c for c in children if c.is_file()]
        if files or len(dirs) != 1:
            return current
        current = dirs[0]
        resolved = current.resolve()
        if resolved in visited:
            return current
        visited.add(resolved)


def ctf_archives_cases(
    corpus: Path, events: dict[str, tuple[str, str]] | None = None
) -> list[tuple[dict, dict]]:
    """Build cases for every dumped event in ``ctf-archives/``.

    Extraction layouts vary per zip (wrappers, nesting): challenges are
    resolved to their content root and deduplicated by that root, with
    the longest directory name as the canonical challenge name. Flags
    stay format-only (tier B): the dumps carry no verifiable ground
    truth. ``events`` defaults to the known CTF-Archives dumps.
    """
    base = corpus / "ctf-archives"
    cases = []
    for event_dir, (prefix, event_date) in sorted((events or CTF_ARCHIVE_EVENTS).items()):
        event = base / event_dir
        readme = read_text(event / "README.md", 40000) if (event / "README.md").is_file() else ""
        attachments = event / "attachments"
        roots: dict[Path, list[str]] = {}
        for challenge in sorted(attachments.iterdir()) if attachments.is_dir() else []:
            if not challenge.is_dir():
                continue
            root = _content_root(challenge)
            if not any(f.is_file() for f in root.rglob("*")):
                continue
            # symlinked wrappers resolve to the same inode: dedup on resolve()
            roots.setdefault(root.resolve(), []).append(challenge.name)
        for root, names in roots.items():
            canonical = max(names, key=len)
            slug = re.sub(r"[^a-z0-9_-]", "", canonical.lower().replace(".", "-"))[:40]
            case_id = f"ctf-{prefix}-{slug}-001"
            statement = _statement_for(readme, canonical)
            excerpts = _archive_material(root)
            case = {
                "id": case_id,
                "source": {
                    "type": "ctfd",
                    "repository": f"CTF-Archives/{event_dir}",
                    "path": canonical,
                    "date": event_date,
                    "url": (
                        f"https://github.com/CTF-Archives/{event_dir}/releases/tag/attachments"
                    ),
                },
                "input": {
                    "code": excerpts,
                    "description": (
                        "CTF challenge statement (Chinese), "
                        f"{event_dir} (CTF-Archives public dump). Statement:\n\n"
                        + (statement or canonical)
                    ),
                    "question": "Solve this CTF challenge.",
                },
                "metadata": {
                    "language": "zh",
                    "category": "forensics",
                    "tier": "unverified",
                    "flag_format": _flag_format(statement),
                },
            }
            manifest = {
                "id": case_id,
                "tier": "unverified",
                "expected_flag": False,
                "url": case["source"]["url"],
            }
            cases.append((case, manifest))
    return cases


def _statement_for(readme: str, name: str) -> str:
    """Extract the statement block under the ``### <title>`` heading.

    Challenge directories use dots (``Ghost.Fleet``) while README headings
    use spaces (``### Ghost Fleet``); both sides are normalised to
    alphanumeric-only lowercase before matching. The block runs until the
    next heading.
    """
    normalized = "".join(char for char in name.lower() if char.isalnum())
    # release dirs carry a hex-hash suffix ("cold_forge_4b1bc..."): strip it
    # or the normalized name never matches the README heading
    normalized = re.sub(r"[0-9a-f]{8,}$", "", normalized)
    lines = readme.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("###") and normalized in "".join(
            char for char in line.lower() if char.isalnum()
        ):
            start = index + 1
            break
    if start is None:
        return ""
    block = []
    for line in lines[start:]:
        if line.startswith("###"):
            break
        block.append(line)
    text = "\n".join(block).strip()
    cleaned = "\n".join(
        line[2:] if line.startswith("> ") else ("" if line == ">" else line)
        for line in text.splitlines()
    )
    cleaned = "\n".join(part for part in cleaned.splitlines() if part.strip()).strip()
    return cleaned[:2000] if cleaned else ""


def _flag_format(statement: str) -> str | None:
    """First flag template in the statement, e.g. ``GEELY{<sha256_hex>}``."""
    match = re.search(r"[A-Z][A-Za-z0-9_]{1,20}\{[^}]{0,120}\}", statement)
    return match.group(0) if match else None


def _attachment_excerpts(challenge: Path) -> str:
    parts = []
    for path in sorted(challenge.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(challenge)
        if path.suffix.lower() in TEXT_SUFFIXES:
            parts.append(f"--- {rel} ---\n{read_text(path)}")
        else:
            parts.append(f"--- {rel} (binary/image, not excerpted) ---")
    return "\n".join(parts)[:8000]


def _is_elf(path: Path) -> bool:
    try:
        return path.read_bytes()[:4] == b"\x7fELF"
    except OSError:
        return False


def _archive_material(challenge: Path) -> str:
    """Text excerpts plus recon output for ELF binaries.

    Binary attachments get the same bounded static recon the tier-A
    lictor/rop cases carry: file type, ELF header, protections, symbol
    hits and a disasm head. Without it a pwn challenge is unanswerable
    ('binary/image, not excerpted' tells the model nothing), and the
    prompt distribution diverges from the recon-style prompts the model
    is trained on.
    """
    parts = []
    for path in sorted(challenge.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(challenge)
        if _is_elf(path):
            parts.append(f"--- {rel} (recon) ---\n{run_recon(path)}")
        elif path.suffix.lower() in TEXT_SUFFIXES:
            parts.append(f"--- {rel} ---\n{read_text(path)}")
        else:
            parts.append(f"--- {rel} (binary/image, not excerpted) ---")
    return "\n".join(parts)[:8000]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()

    builders = (
        ("lictor-lab", lictor_cases),
        ("rop-emporium", rop_emporium_cases),
        ("ctf-archives", ctf_archives_cases),
    )
    written = 0
    manifest_lines = []
    seen_ids: set[str] = set()
    id_re = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    written_paths: list[Path] = []
    for source_name, builder in builders:
        for case, manifest in builder(args.corpus):
            assert id_re.match(case["id"]), f"bad case id: {case['id']}"
            assert case["id"] not in seen_ids, f"duplicate case id: {case['id']}"
            seen_ids.add(case["id"])
            target = args.repo / "examples" / "cases" / f"{case['id']}.json"
            written_paths.append(target)
            target.write_text(case_json(case), encoding="utf-8")
            manifest_lines.append(
                json.dumps({"source": source_name, **manifest}, ensure_ascii=False)
            )
            written += 1
    manifest_path = Path(__file__).parent / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    case_ids = {json.loads(line)["id"] for line in manifest_lines}
    written_ids = {p.stem for p in written_paths}
    assert case_ids == written_ids, f"manifest/case mismatch: {sorted(case_ids ^ written_ids)}"
    print(f"wrote {written} cases -> {args.repo / 'examples' / 'cases'}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
