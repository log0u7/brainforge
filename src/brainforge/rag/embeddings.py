from typing import Any, Protocol

import numpy as np

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float32)


class HashingBackend:
    def __init__(self, dim: int = 256):
        self.name = "hashing"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = int.from_bytes(
                    __import__("hashlib").sha1(token.encode("utf-8")).digest()[:4], "little"
                )
                matrix[row, digest % self.dim] += 1.0
        return _normalize(matrix)


class FastembedBackend:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the fastembed embedding backend; "
                "install it or use the 'hashing' backend"
            ) from exc
        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dim probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return _normalize(matrix)


def get_backend(name: str, **kwargs: Any) -> EmbeddingBackend:
    if name == "hashing":
        return HashingBackend(**kwargs)
    if name == "fastembed":
        return FastembedBackend(**kwargs)
    raise ValueError(f"unknown embedding backend '{name}' (expected: hashing, fastembed)")
