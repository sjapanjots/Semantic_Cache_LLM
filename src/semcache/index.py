"""Vector indexes: how a store finds the most similar saved prompts.

* :class:`ExactIndex` (default) compares the query with every vector. It is exact and
  fast up to tens of thousands of entries per namespace.
* :class:`HNSWIndex` uses an HNSW graph (``pip install "semcacher[hnsw]"``). Lookups
  stay fast with millions of entries, at the cost of being approximate: very rarely
  the closest entry is missed and the next-closest is returned instead. Raise
  ``ef_search`` for better recall at some speed.

Pick one when creating a store::

    from semcache import SemanticCache, InMemoryStore, SQLiteStore
    from semcache.index import HNSWIndex

    cache = SemanticCache(store=InMemoryStore(index="hnsw"))
    cache = SemanticCache(path="cache.db", index="hnsw")   # SQLite, same choice
    cache = SemanticCache(store=SQLiteStore("cache.db", index=lambda dim: HNSWIndex(dim, M=32)))

``index`` takes ``"exact"``, ``"hnsw"``, or any callable ``factory(dim) -> index``;
each namespace and model gets its own index. Vectors must be L2-normalised (every
bundled embedder does this), so a dot product is the cosine similarity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["VectorIndex", "ExactIndex", "HNSWIndex", "IndexSpec", "make_index", "add_many"]


@runtime_checkable
class VectorIndex(Protocol):
    """What a store needs from an index. Keys are entry ids."""

    dim: int

    def add(self, key: str, vector: np.ndarray) -> None:
        """Add ``vector`` under ``key``, replacing any vector already stored there."""

    def remove(self, key: str) -> None:
        """Remove ``key``; unknown keys are ignored."""

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Up to ``k`` ``(key, cosine similarity)`` pairs, most similar first."""

    def __len__(self) -> int: ...


def add_many(index: VectorIndex, keys: list[str], vectors: np.ndarray) -> None:
    """Bulk-add, using the index's own ``add_many`` when it has a faster one."""
    bulk = getattr(index, "add_many", None)
    if bulk is not None:
        bulk(keys, vectors)
    else:
        for key, vector in zip(keys, vectors, strict=True):
            index.add(key, vector)


def _query(vector: np.ndarray, dim: int) -> np.ndarray:
    q = np.asarray(vector, dtype=np.float32).ravel()
    if q.shape[0] != dim:
        raise ValueError(f"embedding dimension mismatch: index has {dim}, query has {q.shape[0]}")
    return q


class ExactIndex:
    """Brute-force cosine search over a growable matrix. Exact; no extra packages."""

    def __init__(self, dim: int, *, initial_capacity: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._matrix = np.empty((max(1, initial_capacity), dim), dtype=np.float32)
        self._keys: list[str] = []
        self._rows: dict[str, int] = {}

    def add(self, key: str, vector: np.ndarray) -> None:
        v = _query(vector, self.dim)
        row = self._rows.get(key)
        if row is None:
            row = len(self._keys)
            if row == self._matrix.shape[0]:  # grow by doubling: amortised O(1) per add
                grown = np.empty((row * 2, self.dim), dtype=np.float32)
                grown[:row] = self._matrix
                self._matrix = grown
            self._keys.append(key)
            self._rows[key] = row
        self._matrix[row] = v

    def remove(self, key: str) -> None:
        row = self._rows.pop(key, None)
        if row is None:
            return
        last = len(self._keys) - 1
        if row != last:  # move the last vector into the hole
            moved = self._keys[last]
            self._matrix[row] = self._matrix[last]
            self._keys[row] = moved
            self._rows[moved] = row
        self._keys.pop()

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        n = len(self._keys)
        if n == 0 or k <= 0:
            return []
        scores = self._matrix[:n] @ _query(vector, self.dim)
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._keys[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._keys)


class HNSWIndex:
    """Approximate nearest-neighbour search with hnswlib.

    Args:
        dim: vector dimension.
        M: graph links per node; higher means better recall and more memory.
        ef_construction: build-time search width; higher means a better graph.
        ef_search: query-time search width; raise it for better recall. A missed
            neighbour only costs a cache miss: whatever is returned must still pass
            the similarity threshold.
        initial_capacity: starting size; the index doubles itself when full.
    """

    def __init__(
        self,
        dim: int,
        *,
        M: int = 16,  # noqa: N803 - the name used throughout HNSW literature and hnswlib
        ef_construction: int = 200,
        ef_search: int = 128,
        initial_capacity: int = 1024,
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("HNSWIndex needs hnswlib: pip install 'semcacher[hnsw]'") from exc
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ef_search = ef_search
        self._index = hnswlib.Index(space="ip", dim=dim)  # ip distance = 1 - cosine
        self._index.init_index(
            max_elements=max(1, initial_capacity),
            ef_construction=ef_construction,
            M=M,
            allow_replace_deleted=True,
        )
        self._labels: dict[str, int] = {}
        self._keys: dict[int, str] = {}
        self._next_label = 0

    def add(self, key: str, vector: np.ndarray) -> None:
        v = _query(vector, self.dim)
        self.remove(key)
        capacity = self._index.get_max_elements()
        if len(self._labels) + 1 > capacity:
            self._index.resize_index(capacity * 2)
        label = self._next_label
        self._next_label += 1
        self._index.add_items(v[None, :], [label], replace_deleted=True)
        self._labels[key] = label
        self._keys[label] = key

    def add_many(self, keys: list[str], vectors: np.ndarray) -> None:
        """Bulk insert using all CPU cores; much faster than repeated :meth:`add`."""
        if not keys:
            return
        batch = np.asarray(vectors, dtype=np.float32).reshape(len(keys), -1)
        if batch.shape[1] != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: index has {self.dim}, vectors have {batch.shape[1]}"
            )
        latest = dict(zip(keys, range(len(keys)), strict=True))  # last vector per key wins
        for key in latest:
            self.remove(key)
        needed = len(self._labels) + len(latest)
        capacity = self._index.get_max_elements()
        if needed > capacity:
            self._index.resize_index(max(needed, capacity * 2))
        labels = list(range(self._next_label, self._next_label + len(latest)))
        self._next_label += len(latest)
        self._index.add_items(batch[list(latest.values())], labels, replace_deleted=True)
        for key, label in zip(latest, labels, strict=True):
            self._labels[key] = label
            self._keys[label] = key

    def remove(self, key: str) -> None:
        label = self._labels.pop(key, None)
        if label is None:
            return
        del self._keys[label]
        self._index.mark_deleted(label)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = _query(vector, self.dim)
        k = min(k, len(self._labels))
        if k <= 0:
            return []
        ef = max(self.ef_search, k)
        while True:
            self._index.set_ef(ef)
            try:
                labels, distances = self._index.knn_query(q, k=k)
                break
            except RuntimeError:
                # hnswlib couldn't collect k live results (many deleted nodes nearby);
                # search wider, up to the whole index.
                if ef >= len(self._labels) + self._index.get_current_count():
                    raise
                ef *= 4
        return [
            (self._keys[int(label)], float(1.0 - dist))
            for label, dist in zip(labels[0], distances[0], strict=True)
        ]

    def __len__(self) -> int:
        return len(self._labels)


IndexSpec = str | Callable[[int], Any] | None


def make_index(spec: IndexSpec, dim: int) -> VectorIndex:
    """Build an index for ``dim``-dimensional vectors from an ``index=`` argument."""
    if spec is None or spec == "exact":
        return ExactIndex(dim)
    if spec == "hnsw":
        return HNSWIndex(dim)
    if callable(spec):
        return spec(dim)
    raise ValueError(f"unknown index {spec!r}; use 'exact', 'hnsw' or a factory(dim)")
