"""The :class:`SemanticCache`: look up answers by meaning instead of exact text."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .embedders import Embedder, HashingEmbedder, normalize_text
from .pricing import PriceBook
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
        threshold: minimum cosine similarity (0..1) for a semantic hit. Higher is stricter.
            Defaults to the embedder's ``recommended_threshold`` (0.85 if it has none).
        namespace: default namespace; entries in different namespaces never match.
        pricing: a :class:`PriceBook`, or a ``{"model": (input, output)}`` mapping of
            USD per million tokens merged over the bundled defaults.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: BaseStore | None = None,
        *,
        threshold: float | None = None,
        namespace: str = "default",
        pricing: PriceBook | Mapping[str, Any] | None = None,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        if threshold is None:
            threshold = float(getattr(self.embedder, "recommended_threshold", 0.85))
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.store: BaseStore = store if store is not None else InMemoryStore()
        self.threshold = threshold
        self.namespace = namespace
        self.pricing = pricing if isinstance(pricing, PriceBook) else PriceBook(pricing)
        self._stats = StatsTracker()
        self._lock = threading.RLock()

    # -- helpers -----------------------------------------------------------------------

    def _scope(self, namespace: str | None, model: str | None) -> str:
        """Entries only match within the same namespace *and* model."""
        return f"{namespace or self.namespace}::{model or '*'}"

    def _embed_one(self, text: str):
        return self.embedder.embed([text])[0]

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

        with self._lock:
            entry = self.store.find_exact(normalized, scope) if normalized else None
            similarity, exact = 1.0, True
            if entry is None:
                exact = False
                results = self.store.search(self._embed_one(prompt), scope, k=1)
                if results and results[0][1] >= threshold:
                    entry, similarity = results[0]

            if entry is None:
                self._stats.record_miss(model)
                return None

            saved = self.entry_cost(entry)
            self.store.record_hit(entry.id, time.time())
            self._stats.record_hit(
                model or entry.model, entry.input_tokens, entry.output_tokens, saved, exact
            )
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
        return entry

    @property
    def stats(self) -> CacheStats:
        """A snapshot of hit rate, tokens saved and money saved."""
        return self._stats.snapshot()

    def reset_stats(self) -> None:
        self._stats.reset()

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
        self.store.close()

    def __enter__(self) -> SemanticCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
