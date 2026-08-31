import hashlib
import json
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel

from brainforge.providers.base import ChatRequest, ChatResponse, Provider


class CacheProvider(Provider):
    def __init__(self, inner: Provider, cache_path: Path, enabled: bool = True):
        super().__init__(inner.name, inner.config)
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        if not self.enabled:
            return self.inner.complete(request, model)
        key = self._key(request, model)
        hit = self._load(key)
        if hit is not None:
            hit.cached = True
            return hit
        response = self.inner.complete(request, model)
        self._store(key, response)
        return response

    def structured(self, request: ChatRequest, model: str, schema: type[BaseModel]) -> ChatResponse:
        return self.inner.structured(request, model, schema)

    def _key(self, request: ChatRequest, model: str) -> str:
        payload = json.dumps(
            {
                "provider": self.inner.name,
                "model": model,
                "messages": [m.model_dump() for m in request.messages],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "cache_extra": request.cache_extra,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path)

    def _load(self, key: str) -> ChatResponse | None:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return ChatResponse.model_validate(json.loads(row[0]))

    def _store(self, key: str, response: ChatResponse) -> None:
        payload = response.model_copy(update={"cached": False}).model_dump_json()
        try:
            with self._connect() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache "
                    "(key TEXT PRIMARY KEY, created_at REAL, payload TEXT)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, created_at, payload) VALUES (?, ?, ?)",
                    (key, time.time(), payload),
                )
        except sqlite3.Error:
            pass
