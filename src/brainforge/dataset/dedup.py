import re

from brainforge.dataset.writer import DatasetRecord

_WORD_RE = re.compile(r"\w+")
_SHINGLE_SIZE = 5


def _normalization_text(record: DatasetRecord) -> str:
    user = record.messages[0].content if record.messages else ""
    assistant = record.messages[-1].content if record.messages else ""
    combined = f"{user}\n{assistant}".lower()
    return " ".join(_WORD_RE.findall(combined))


def shingles(text: str, size: int = _SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def jaccard(first: set, second: set) -> float:
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def exact_hash(record: DatasetRecord) -> str:
    import hashlib

    return hashlib.sha256(_normalization_text(record).encode("utf-8")).hexdigest()


def find_duplicates(records: list[DatasetRecord], threshold: float = 0.85) -> dict[int, int]:
    hashes: dict[str, int] = {}
    shingle_sets: list[set] = []
    duplicates: dict[int, int] = {}
    for index, record in enumerate(records):
        digest = exact_hash(record)
        if digest in hashes:
            duplicates[index] = hashes[digest]
            shingle_sets.append(set())
            continue
        hashes[digest] = index
        current = shingles(_normalization_text(record))
        duplicate_of = None
        for prior_index, prior in enumerate(shingle_sets):
            if not prior:
                continue
            if jaccard(current, prior) >= threshold:
                duplicate_of = prior_index
                break
        if duplicate_of is not None:
            duplicates[index] = duplicate_of
        shingle_sets.append(current)
    return duplicates


def deduplicate(
    records: list[DatasetRecord], threshold: float = 0.85
) -> tuple[list[DatasetRecord], dict[int, int]]:
    duplicates = find_duplicates(records, threshold)
    kept = [record for index, record in enumerate(records) if index not in duplicates]
    return kept, duplicates
