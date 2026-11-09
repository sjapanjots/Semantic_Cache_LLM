"""Storage back-ends for cache entries and their embeddings."""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CacheEntry", "BaseStore", "InMemoryStore"]


@dataclass
class CacheEntry:
    """One cached prompt and the answer the LLM gave for it."""

    prompt: str
    response: Any
    embedding: np.ndarray
    scope: str = "default"
    normalized_prompt: str = ""
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    last_hit_at: float | None = None
    hits: int = 0

    @property
    def last_used_at(self) -> float:
        return self.last_hit_at if self.last_hit_at is not None else self.created_at


class BaseStore(ABC):
    """Interface every store implements. Embeddings are assumed to be L2-normalised."""

    @abstractmethod
    def add(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    def get(self, entry_id: str) -> CacheEntry | None: ...

    @abstractmethod
    def delete(self, entry_id: str) -> bool: ...

    @abstractmethod
    def clear(self, scope: str | None = None) -> int: ...

    @abstractmethod
    def search(self, vector: np.ndarray, scope: str, k: int = 1) -> list[tuple[CacheEntry, float]]:
        """Return up to ``k`` entries in ``scope`` ranked by cosine similarity."""

    @abstractmethod
    def find_exact(self, normalized_prompt: str, scope: str) -> CacheEntry | None: ...

    @abstractmethod
    def record_hit(self, entry_id: str, when: float) -> None: ...

    @abstractmethod
    def entries(self, scope: str | None = None) -> Iterator[CacheEntry]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources. No-op by default."""


class _ScopeIndex:
    """Vectors of one scope, stacked into a matrix that is rebuilt lazily after writes."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.exact: dict[str, str] = {}
        self._matrix: np.ndarray | None = None

    def add(self, entry: CacheEntry) -> None:
        self.ids.append(entry.id)
        self.vectors.append(np.asarray(entry.embedding, dtype=np.float32).ravel())
        if entry.normalized_prompt:
            self.exact[entry.normalized_prompt] = entry.id
        self._matrix = None

    def remove(self, entry: CacheEntry) -> None:
        try:
            i = self.ids.index(entry.id)
        except ValueError:
            return
        del self.ids[i]
        del self.vectors[i]
        if self.exact.get(entry.normalized_prompt) == entry.id:
            del self.exact[entry.normalized_prompt]
        self._matrix = None

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.vstack(self.vectors) if self.vectors else np.empty((0, 0))
        return self._matrix


class InMemoryStore(BaseStore):
    """Keeps everything in process memory. Fast, thread-safe, lost on exit."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._scopes: dict[str, _ScopeIndex] = {}
        self._lock = threading.RLock()

    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            if entry.id in self._entries:
                self.delete(entry.id)
            self._entries[entry.id] = entry
            self._scopes.setdefault(entry.scope, _ScopeIndex()).add(entry)

    def get(self, entry_id: str) -> CacheEntry | None:
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._entries.pop(entry_id, None)
            if entry is None:
                return False
            index = self._scopes.get(entry.scope)
            if index is not None:
                index.remove(entry)
                if not index.ids:
                    del self._scopes[entry.scope]
            return True

    def clear(self, scope: str | None = None) -> int:
        with self._lock:
            if scope is None:
                n = len(self._entries)
                self._entries.clear()
                self._scopes.clear()
                return n
            index = self._scopes.pop(scope, None)
            if index is None:
                return 0
            for entry_id in index.ids:
                self._entries.pop(entry_id, None)
            return len(index.ids)

    def search(self, vector: np.ndarray, scope: str, k: int = 1) -> list[tuple[CacheEntry, float]]:
        with self._lock:
            index = self._scopes.get(scope)
            if index is None or not index.ids:
                return []
            matrix = index.matrix()
            query = np.asarray(vector, dtype=np.float32).ravel()
            if matrix.shape[1] != query.shape[0]:
                raise ValueError(
                    f"embedding dimension mismatch: store has {matrix.shape[1]}, "
                    f"query has {query.shape[0]}"
                )
            scores = matrix @ query
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._entries[index.ids[i]], float(scores[i])) for i in top]

    def find_exact(self, normalized_prompt: str, scope: str) -> CacheEntry | None:
        with self._lock:
            index = self._scopes.get(scope)
            if index is None:
                return None
            entry_id = index.exact.get(normalized_prompt)
            return self._entries.get(entry_id) if entry_id else None

    def record_hit(self, entry_id: str, when: float) -> None:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                entry.hits += 1
                entry.last_hit_at = when

    def entries(self, scope: str | None = None) -> Iterator[CacheEntry]:
        with self._lock:
            items = list(self._entries.values())
        return iter(e for e in items if scope is None or e.scope == scope)

    def __len__(self) -> int:
        return len(self._entries)
