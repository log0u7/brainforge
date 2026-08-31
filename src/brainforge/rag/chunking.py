import hashlib

from pydantic import BaseModel

from brainforge.rag.loaders import Document

_MAX_CHARS = 1200
_OVERLAP_CHARS = 150


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    start: int
    end: int
    hash: str


def _sections(text: str, is_markdown: bool) -> list[str]:
    if is_markdown:
        blocks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                if current:
                    blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks
    return text.split("\n\n")


def _blocks_to_spans(blocks: list[str]) -> list[tuple[int, int]]:
    spans = []
    offset = 0
    for block in blocks:
        spans.append((offset, offset + len(block)))
        offset += len(block) + 2
    return spans


def chunk_text(
    text: str, max_chars: int = _MAX_CHARS, overlap: int = _OVERLAP_CHARS
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            newline = text.rfind("\n", start + max_chars // 2, end)
            if newline > start:
                end = newline
        spans.append((start, end))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_document(document: Document, max_chars: int = _MAX_CHARS) -> list[Chunk]:
    is_markdown = document.metadata.get("extension") == ".md"
    blocks = _sections(document.text, is_markdown)
    spans = _blocks_to_spans(blocks)
    chunks: list[Chunk] = []
    sequence = 0
    for block_text, (block_start, block_end) in zip(blocks, spans, strict=False):
        block = (
            document.text[block_start:block_end] if block_end > len(document.text) else block_text
        )
        if len(block) <= max_chars:
            candidates = [(block_start, block_start + len(block))]
        else:
            candidates = chunk_text(block, max_chars)
        for span_start, span_end in candidates:
            text = block[span_start - block_start : span_end - block_start]
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{sequence:04d}",
                    doc_id=document.doc_id,
                    source=document.source,
                    text=text,
                    start=block_start + (span_start - block_start),
                    end=block_start + (span_end - block_start),
                    hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
            sequence += 1
    return chunks


def chunk_documents(documents: list[Document], max_chars: int = _MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars))
    return chunks
