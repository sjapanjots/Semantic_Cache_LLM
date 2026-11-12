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

from .index import IndexSpec, VectorIndex, add_many, make_index

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
    cost: float | None = None  # explicit USD cost of the original call; overrides pricing
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=lambda: time.time())
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

    # Optional hooks; in-memory stores ignore them, persistent stores override them.

    def bind_embedder(self, name: str) -> None:  # noqa: B027
        """Record which embedder produced the stored vectors (and reject a different one)."""

    def load_stats(self) -> dict[str, Any] | None:
        return None

    def save_stats(self, data: dict[str, Any]) -> None:  # noqa: B027
        pass

    def close(self) -> None:  # noqa: B027
        """Release resources. No-op by default."""


class _ScopeIndex:
    """One scope's entry ids, exact-match keys and vector index."""

    def __init__(self, index: IndexSpec) -> None:
        self._spec = index
        self.ids: set[str] = set()
        self.exact: dict[str, str] = {}
        self.vectors: VectorIndex | None = None  # created on the first add, once dim is known

    def add(self, entry: CacheEntry) -> None:
        vector = np.asarray(entry.embedding, dtype=np.float32).ravel()
        if self.vectors is None:
            self.vectors = make_index(self._spec, vector.shape[0])
        self.vectors.add(entry.id, vector)
        self.ids.add(entry.id)
        if entry.normalized_prompt:
            self.exact[entry.normalized_prompt] = entry.id

    def add_many(self, entries: list[CacheEntry]) -> None:
        if not entries:
            return
        vectors = np.vstack([np.asarray(e.embedding, dtype=np.float32).ravel() for e in entries])
        if self.vectors is None:
            self.vectors = make_index(self._spec, vectors.shape[1])
        add_many(self.vectors, [e.id for e in entries], vectors)
        for entry in entries:
            self.ids.add(entry.id)
            if entry.normalized_prompt:
                self.exact[entry.normalized_prompt] = entry.id

    def remove(self, entry: CacheEntry) -> None:
        if entry.id not in self.ids:
            return
        self.ids.discard(entry.id)
        if self.vectors is not None:
            self.vectors.remove(entry.id)
        if self.exact.get(entry.normalized_prompt) == entry.id:
            del self.exact[entry.normalized_prompt]


class InMemoryStore(BaseStore):
    """Keeps everything in process memory. Fast, thread-safe, lost on exit.

    Args:
        index: how each scope searches its vectors: ``"exact"`` (default), ``"hnsw"``
            for large caches, or a ``factory(dim)``. See :mod:`semcache.index`.
    """

    def __init__(self, *, index: IndexSpec = None) -> None:
        if index not in (None, "exact", "hnsw") and not callable(index):
            raise ValueError(f"unknown index {index!r}; use 'exact', 'hnsw' or a factory(dim)")
        self._index = index
        self._entries: dict[str, CacheEntry] = {}
        self._scopes: dict[str, _ScopeIndex] = {}
        self._lock = threading.RLock()

    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            if entry.id in self._entries:
                self.delete(entry.id)
            self._entries[entry.id] = entry
            self._scopes.setdefault(entry.scope, _ScopeIndex(self._index)).add(entry)

    def add_many(self, entries: list[CacheEntry]) -> None:
        """Add many entries at once; vector indexes are built in bulk (fast for HNSW)."""
        by_scope: dict[str, list[CacheEntry]] = {}
        with self._lock:
            for entry in entries:
                if entry.id in self._entries:
                    self.delete(entry.id)
                self._entries[entry.id] = entry
                by_scope.setdefault(entry.scope, []).append(entry)
            for scope, group in by_scope.items():
                self._scopes.setdefault(scope, _ScopeIndex(self._index)).add_many(group)

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
            if index is None or index.vectors is None or not index.ids:
                return []
            return [(self._entries[key], score) for key, score in index.vectors.search(vector, k)]

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
