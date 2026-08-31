import sqlite3
from pathlib import Path

import numpy as np

from brainforge.rag.chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE,
    doc_id TEXT,
    source TEXT,
    text TEXT,
    hash TEXT,
    start INTEGER,
    end INTEGER
);
"""


class VectorStore:
    def __init__(self, index_dir: Path | str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "index.db"
        self.vectors_path = self.index_dir / "vectors.npy"
        self._matrix: np.ndarray | None = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_SCHEMA)
        return conn

    def _load_matrix(self, dim: int) -> np.ndarray:
        if self._matrix is None:
            if self.vectors_path.exists():
                self._matrix = np.load(self.vectors_path)
            else:
                self._matrix = np.zeros((0, dim), dtype=np.float32)
        return self._matrix

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        matrix = self._load_matrix(vectors.shape[1] if len(vectors) else self._dim_hint())
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    "INSERT OR REPLACE INTO chunks"
                    " (chunk_id, doc_id, source, text, hash, start, end)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.source,
                        chunk.text,
                        chunk.hash,
                        chunk.start,
                        chunk.end,
                    ),
                )
        matrix = np.vstack([matrix, vectors]) if len(vectors) else matrix
        self._matrix = matrix
        np.save(self.vectors_path, matrix)
        return len(chunks)

    def _dim_hint(self) -> int:
        if self.vectors_path.exists():
            return int(np.load(self.vectors_path).shape[1])
        return 1

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, source, text, hash, start, end FROM chunks ORDER BY id"
            ).fetchall()
        return [
            Chunk(
                chunk_id=r[0], doc_id=r[1], source=r[2], text=r[3], hash=r[4], start=r[5], end=r[6]
            )
            for r in rows
        ]

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        chunks = self.all_chunks()
        matrix = self._load_matrix(self._dim_hint())
        if not chunks or matrix.shape[0] == 0:
            return []
        count = min(len(chunks), matrix.shape[0])
        chunks = chunks[:count]
        scores = matrix[:count] @ query_vector
        order = np.argsort(-scores)[:k]
        return [(chunks[int(index)], float(scores[int(index)])) for index in order]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
        self._matrix = None
        if self.vectors_path.exists():
            self.vectors_path.unlink()
