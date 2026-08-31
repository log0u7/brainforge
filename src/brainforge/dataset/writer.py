import json
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
    case: Case, domain: str, student_input: str, canonical: dict, metadata: dict
) -> DatasetRecord:
    assistant_content = json.dumps(canonical, ensure_ascii=False, indent=2)
    return DatasetRecord(
        id=case.id,
        domain=domain,
        messages=[
            ChatMessage(role="user", content=student_input),
            ChatMessage(role="assistant", content=assistant_content),
        ],
        metadata=metadata,
    )


def append_record(path: Path | str, record: DatasetRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def write_jsonl(path: Path | str, records: list[DatasetRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
