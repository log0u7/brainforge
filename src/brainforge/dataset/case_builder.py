import hashlib
import json
import subprocess
from pathlib import Path

from brainforge.case import Case, CaseInput, CaseSource

_CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".sh": "shell",
    ".bash": "shell",
}
_MAX_FILE_BYTES = 512_000


def case_id_for(payload: str) -> str:
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"case-{digest}"


def build_case_from_dict(data: dict) -> Case:
    source = CaseSource(**(data.get("source") or {"type": "manual"}))
    case_input = CaseInput(**(data.get("input") or {}))
    metadata = data.get("metadata") or {}
    case_id = data.get("id") or case_id_for(
        json.dumps([source.model_dump(), case_input.model_dump()], sort_keys=True)
    )
    return Case(id=case_id, source=source, input=case_input, metadata=metadata)


def _git_commit_date(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _language_for(path: Path) -> str | None:
    return _CODE_EXTENSIONS.get(path.suffix.lower())


def build_case_from_file(path: Path, root: Path) -> Case | None:
    if path.stat().st_size > _MAX_FILE_BYTES:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None
    relative = path.relative_to(root)
    commit_date = _git_commit_date(path)
    source = CaseSource(
        type="file",
        path=str(relative),
        date=commit_date,
    )
    case_input = CaseInput(code=text)
    metadata = {"language": _language_for(path)}
    case_id = case_id_for(
        json.dumps([source.model_dump(), case_input.model_dump()], sort_keys=True)
    )
    return Case(id=case_id, source=source, input=case_input, metadata=metadata)


def build_cases_from_path(path: Path | str) -> list[Case]:
    root = Path(path)
    if root.is_file():
        files = [root]
        root = root.parent
    else:
        files = sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and not item.name.startswith(".") and "__pycache__" not in item.parts
        )
    cases: list[Case] = []
    for item in files:
        if item.suffix.lower() == ".json":
            try:
                data = json.loads(item.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "input" in data:
                cases.append(build_case_from_dict(data))
        elif item.suffix.lower() in _CODE_EXTENSIONS:
            case = build_case_from_file(item, root)
            if case is not None:
                cases.append(case)
    return cases
