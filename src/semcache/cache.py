"""The :class:`SemanticCache`: look up answers by meaning instead of exact text."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embedders import Embedder, HashingEmbedder, normalize_text
from .pricing import PriceBook
from .sqlite_store import SQLiteStore
from .stats import CacheStats, StatsTracker
from .store import BaseStore, CacheEntry, InMemoryStore

__all__ = ["SemanticCache", "CacheHit"]


@dataclass(frozen=True)
class CacheHit:
    """Result of a successful lookup."""

    response: Any
    similarity: float
    entry: CacheEntry
    exact: bool = False
    cost_saved: float = 0.0


class SemanticCache:
    """Return a saved LLM answer when a new prompt means the same as an old one.

    Args:
        embedder: turns prompts into vectors. Defaults to :class:`HashingEmbedder`.
        store: where entries live. Defaults to :class:`InMemoryStore`.
        path: shorthand for ``store=SQLiteStore(path)``, to persist the cache to a file.
        threshold: minimum cosine similarity (0..1) for a semantic hit. Higher is stricter.
            Defaults to the embedder's ``recommended_threshold`` (0.85 if it has none).
        namespace: default namespace; entries in different namespaces never match.
        pricing: a :class:`PriceBook`, or a ``{"model": (input, output)}`` mapping of
            USD per million tokens merged over the bundled defaults.
        ttl: seconds an entry stays valid. ``None`` keeps entries forever.
        max_entries: cap on stored entries; the least recently used are evicted.
        stats_flush_every: with a persistent store, save stats every N operations
            (they are always saved on :meth:`close`).
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: BaseStore | None = None,
        *,
        path: str | Path | None = None,
        threshold: float | None = None,
        namespace: str = "default",
        pricing: PriceBook | Mapping[str, Any] | None = None,
        ttl: float | None = None,
        max_entries: int | None = None,
        stats_flush_every: int = 20,
    ) -> None:
        if store is not None and path is not None:
            raise ValueError("pass either store or path, not both")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.embedder: Embedder = embedder or HashingEmbedder()
        if threshold is None:
            threshold = float(getattr(self.embedder, "recommended_threshold", 0.85))
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if store is None:
            store = SQLiteStore(path) if path is not None else InMemoryStore()
        self.store: BaseStore = store
        self.store.bind_embedder(self.embedder.name)
        self.threshold = threshold
        self.namespace = namespace
        self.pricing = pricing if isinstance(pricing, PriceBook) else PriceBook(pricing)
        self.ttl = ttl
        self.max_entries = max_entries
        self.stats_flush_every = max(1, stats_flush_every)
        saved = self.store.load_stats()
        self._stats = StatsTracker(CacheStats.from_dict(saved) if saved else None)
        self._ops_since_flush = 0
        self._lock = threading.RLock()

    # -- helpers -----------------------------------------------------------------------

    def _scope(self, namespace: str | None, model: str | None) -> str:
        """Entries only match within the same namespace *and* model."""
        return f"{namespace or self.namespace}::{model or '*'}"

    def _embed_one(self, text: str):
        return self.embedder.embed([text])[0]

    def _expired(self, entry: CacheEntry, now: float) -> bool:
        return self.ttl is not None and now - entry.created_at > self.ttl

    def _after_op(self) -> None:
        self._ops_since_flush += 1
        if self._ops_since_flush >= self.stats_flush_every:
            self.flush_stats()

    def _evict(self) -> None:
        if self.max_entries is None or len(self.store) <= self.max_entries:
            return
        excess = len(self.store) - self.max_entries
        oldest = sorted(self.store.entries(), key=lambda e: e.last_used_at)[:excess]
        for entry in oldest:
            self.store.delete(entry.id)

    def entry_cost(self, entry: CacheEntry) -> float:
        """What the call that produced ``entry`` cost, i.e. what one hit on it saves."""
        if entry.cost is not None:
            return entry.cost
        return self.pricing.cost(entry.model, entry.input_tokens, entry.output_tokens)

    # -- public API --------------------------------------------------------------------

    def lookup(
        self,
        prompt: str,
        *,
        model: str | None = None,
        namespace: str | None = None,
        threshold: float | None = None,
    ) -> CacheHit | None:
        """Find a cached answer for ``prompt``. Returns ``None`` on a miss."""
        scope = self._scope(namespace, model)
        normalized = normalize_text(prompt)
        threshold = self.threshold if threshold is None else threshold

        now = time.time()
        with self._lock:
            entry = self.store.find_exact(normalized, scope) if normalized else None
            if entry is not None and self._expired(entry, now):
                self.store.delete(entry.id)
                entry = None
            similarity, exact = 1.0, True
            if entry is None:
                exact = False
                vector = None
                for _ in range(8):  # skip past expired neighbours, deleting them as we go
                    vector = self._embed_one(prompt) if vector is None else vector
                    results = self.store.search(vector, scope, k=1)
                    if not results or results[0][1] < threshold:
                        break
                    candidate, score = results[0]
                    if self._expired(candidate, now):
                        self.store.delete(candidate.id)
                        continue
                    entry, similarity = candidate, score
                    break

            if entry is None:
                self._stats.record_miss(model)
                self._after_op()
                return None

            saved = self.entry_cost(entry)
            self.store.record_hit(entry.id, now)
            self._stats.record_hit(
                model or entry.model, entry.input_tokens, entry.output_tokens, saved, exact
            )
            self._after_op()
            return CacheHit(entry.response, similarity, entry, exact=exact, cost_saved=saved)

    def get(self, prompt: str, default: Any = None, **kwargs: Any) -> Any:
        """Return the cached response for ``prompt``, or ``default`` on a miss."""
        hit = self.lookup(prompt, **kwargs)
        return default if hit is None else hit.response

    def set(
        self,
        prompt: str,
        response: Any,
        *,
        model: str | None = None,
        namespace: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Store ``response`` as the answer to ``prompt``.

        ``input_tokens`` and ``output_tokens`` describe the original call; they are priced
        with :attr:`pricing` and credited as savings every time this entry is served.
        Pass ``cost`` (USD) instead to skip the price lookup.
        """
        normalized = normalize_text(prompt)
        scope = self._scope(namespace, model)
        entry = CacheEntry(
            prompt=prompt,
            response=response,
            embedding=self._embed_one(prompt),
            scope=scope,
            normalized_prompt=normalized,
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost=cost,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            existing = self.store.find_exact(normalized, scope) if normalized else None
            if existing is not None:
                self.store.delete(existing.id)
            self.store.add(entry)
            self._stats.record_write(model, self.entry_cost(entry))
            self._evict()
            self._after_op()
        return entry

    @property
    def stats(self) -> CacheStats:
        """A snapshot of hit rate, tokens saved and money saved."""
        return self._stats.snapshot()

    def reset_stats(self) -> None:
        self._stats.reset()
        self.flush_stats()

    def flush_stats(self) -> None:
        """Write stats to the store now (a no-op for in-memory stores)."""
        with self._lock:
            self.store.save_stats(self._stats.snapshot().to_dict())
            self._ops_since_flush = 0

    def prune_expired(self) -> int:
        """Delete every entry older than ``ttl``. Returns how many were removed."""
        if self.ttl is None:
            return 0
        now = time.time()
        with self._lock:
            stale = [e.id for e in self.store.entries() if self._expired(e, now)]
            for entry_id in stale:
                self.store.delete(entry_id)
        return len(stale)

    def delete(self, entry_id: str) -> bool:
        return self.store.delete(entry_id)

    def clear(self, namespace: str | None = None) -> int:
        """Remove all entries, or only those in ``namespace`` (every model)."""
        if namespace is None:
            return self.store.clear()
        prefix = f"{namespace}::"
        scopes = {e.scope for e in self.store.entries() if e.scope.startswith(prefix)}
        return sum(self.store.clear(s) for s in scopes)

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, prompt: str) -> bool:
        scope = self._scope(None, None)
        return self.store.find_exact(normalize_text(prompt), scope) is not None

    def close(self) -> None:
        self.flush_stats()
        self.store.close()

    def __enter__(self) -> SemanticCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
