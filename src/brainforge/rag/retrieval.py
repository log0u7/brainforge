from pathlib import Path

from pydantic import BaseModel

from brainforge.rag.chunking import chunk_documents
from brainforge.rag.embeddings import EmbeddingBackend
from brainforge.rag.loaders import load_documents
from brainforge.rag.store import VectorStore


class RetrievedChunk(BaseModel):
    text: str
    source: str
    doc_id: str
    chunk_id: str
    hash: str
    score: float


class Retriever:
    def __init__(self, index_dir: Path | str, backend: EmbeddingBackend):
        self.store = VectorStore(index_dir)
        self.backend = backend

    def index(self, path: Path | str) -> dict:
        documents = load_documents(path)
        chunks = chunk_documents(documents)
        self.store.clear()
        added = 0
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.backend.embed([chunk.text for chunk in batch])
            added += self.store.add(batch, vectors)
        return {"documents": len(documents), "chunks": added}

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        vectors = self.backend.embed([query])
        if vectors.shape[0] == 0:
            return []
        results = self.store.search(vectors[0], k=k)
        return [
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                hash=chunk.hash,
                score=round(score, 4),
            )
            for chunk, score in results
        ]
