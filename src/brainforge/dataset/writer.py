import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

from brainforge.case import Case
from brainforge.providers.base import ChatMessage


class DatasetRecord(BaseModel):
    id: str
    domain: str
    messages: list[ChatMessage]
    metadata: dict


def build_record(
    case: Case,
    domain: str,
    student_input: str,
    canonical: dict,
    metadata: dict,
    raw_assistant: str | None = None,
) -> DatasetRecord:
    # raw_assistant: the teacher's exact output (KD mode), so the recorded
    # top-k logprobs align with the student's gold tokens. When absent, the
    # canonical dict is re-serialized with stable formatting.
    assistant_content = (
        raw_assistant
        if raw_assistant is not None
        else json.dumps(canonical, ensure_ascii=False, indent=2)
    )
    return DatasetRecord(
        id=case.id,
        domain=domain,
        messages=[
            ChatMessage(role="user", content=student_input),
            ChatMessage(role="assistant", content=assistant_content),
        ],
        metadata=metadata,
    )


def build_fallback_record(
    case: Case,
    domain: str,
    student_input: str,
    reasons: list[str],
    metadata: dict,
) -> DatasetRecord:
    """Abstention record teaching the student to fall back instead of guessing.

    Emitted for cases the pipeline judged unreliable (gate rejection, teacher
    disagreement, low confidence): the assistant target is a canonical
    ``insufficient_information`` verdict, a value every domain pack accepts.
    """
    canonical = {
        "verdict": "insufficient_information",
        "reasoning": "; ".join(reasons),
        "confidence": 0.0,
    }
    return DatasetRecord(
        id=case.id,
        domain=domain,
        messages=[
            ChatMessage(role="user", content=student_input),
            ChatMessage(
                role="assistant",
                content=json.dumps(canonical, ensure_ascii=False, indent=2),
            ),
        ],
        metadata=metadata,
    )


def append_record(path: Path | str, record: DatasetRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def write_jsonl(path: Path | str, records: list[DatasetRecord]) -> None:
    """Replace the file with exactly these records, atomically.

    Writes a sibling temp file then renames, so a crash mid-write can never
    leave a truncated dataset behind.
    """
    import os

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
    os.replace(tmp, target)


def read_jsonl(path: Path | str) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    """Stream JSONL records one at a time (bounded RAM for large datasets)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def file_digest(path: Path | str) -> str:
    """Streaming sha256 hex digest of a file.

    Used as a Dataset.from_generator cache key component: the fingerprint
    must change whenever the file content changes, or a regenerated dataset
    silently trains on a stale cached arrow table.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rejected(
    rejected_dir: Path | str,
    pipeline_name: str,
    case: Case,
    reasons: list[str],
    judge_result: dict | None,
    metadata: dict,
) -> Path:
    target = Path(rejected_dir) / pipeline_name / f"{case.id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case.model_dump(),
        "gate_reasons": reasons,
        "judge": judge_result,
        "metadata": metadata,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
