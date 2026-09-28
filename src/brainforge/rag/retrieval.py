from pathlib import Path

import numpy as np
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
        # ponytail: embed in batches of 64, persist once; per-batch np.save would
        # make total I/O quadratic during indexing
        batch_size = 64
        vector_batches = [
            self.backend.embed([chunk.text for chunk in chunks[start : start + batch_size]])
            for start in range(0, len(chunks), batch_size)
        ]
        if vector_batches:
            self.store.add(chunks, np.vstack(vector_batches))
        return {"documents": len(documents), "chunks": len(chunks)}

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
