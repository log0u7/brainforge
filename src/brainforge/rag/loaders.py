import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

_TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".jsonl",
    ".csv",
    ".log",
    ".diff",
    ".patch",
    ".py",
    ".js",
    ".ts",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".sh",
    ".bash",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".html",
    ".css",
}
_MAX_FILE_BYTES = 2_000_000


class Document(BaseModel):
    doc_id: str
    source: str
    text: str
    metadata: dict = Field(default_factory=dict)


def _doc_id_for(path: Path) -> str:
    return "doc-" + hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def _json_to_text(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(data, indent=2, ensure_ascii=False)


def load_documents(path: Path | str) -> list[Document]:
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
    documents: list[Document] = []
    for item in files:
        if item.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        if item.stat().st_size > _MAX_FILE_BYTES:
            continue
        try:
            raw = item.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not raw.strip():
            continue
        text = _json_to_text(raw) if item.suffix.lower() == ".json" else raw
        documents.append(
            Document(
                doc_id=_doc_id_for(item),
                source=str(item.relative_to(root)) if item.is_relative_to(root) else str(item),
                text=text,
                metadata={"extension": item.suffix.lower()},
            )
        )
    return documents
