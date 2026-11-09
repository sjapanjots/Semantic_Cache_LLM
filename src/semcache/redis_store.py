"""A store kept in Redis, so several processes or servers share one cache.

::

    from semcache import SemanticCache
    from semcache.redis_store import RedisStore

    cache = SemanticCache(store=RedisStore("redis://localhost:6379/0"))

Install the client with ``pip install "semcacher[redis]"``. Any Redis-compatible
server works (Redis, Valkey, KeyDB, managed services); no modules are required.

Layout, all under ``prefix`` (default ``"semcache:"``):

* ``e:<id>``: hash with one entry (prompt, response JSON, embedding bytes, tokens, ...)
* ``s:<scope>``: set of entry ids in a scope
* ``x:<scope>``: hash mapping normalised prompt to entry id (the exact-match path)
* ``scopes``: set of scopes in use
* ``meta``: hash with the embedder name
* ``stats``: hash of counters that every process adds to

Search is an exact cosine scan done in this process. Embeddings never change once
written, so each process downloads an entry's vector once and keeps it; a search only
fetches the scope's id list and any vectors it has not seen yet. That is fast up to
tens of thousands of entries per scope.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .sqlite_store import EmbedderMismatchError
from .store import BaseStore, CacheEntry

__all__ = ["RedisStore"]

_FIELDS = (
    "scope",
    "prompt",
    "normalized_prompt",
    "response",
    "embedding",
    "model",
    "input_tokens",
    "output_tokens",
    "cost",
    "metadata",
    "created_at",
    "last_hit_at",
    "hits",
)


def _s(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _flatten_stats(data: dict[str, Any]) -> dict[str, float]:
    """Numeric stats fields, with per-model ones as ``by_model.<model>.<field>``."""
    flat: dict[str, float] = {}
    for key, value in data.items():
        if key == "by_model":
            for model, fields in (value or {}).items():
                for field, v in fields.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        flat[f"by_model.{model}.{field}"] = float(v)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = float(value)
    return flat


# Derived values that CacheStats.to_dict() adds; they must not be summed.
_DERIVED = {"hit_rate", "tokens_saved", "savings_rate"}


class RedisStore(BaseStore):
    """Stores entries, hit counts and stats in Redis.

    Args:
        url: Redis connection URL, used when ``client`` is not given.
        client: an existing ``redis.Redis`` client. It must be created with
            ``decode_responses=False`` (the default), because embeddings are bytes.
        prefix: key prefix, so several caches can share one database.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        client: Any = None,
        prefix: str = "semcache:",
    ) -> None:
        self._owns_client = client is None
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "RedisStore needs the redis package: pip install 'semcacher[redis]'"
                ) from exc
            client = redis.Redis.from_url(url)
        self.r = client
        self.prefix = prefix
        self._lock = threading.RLock()
        self._vectors: dict[str, np.ndarray] = {}  # entry id -> embedding (immutable)
        self._matrices: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self._stats_saved: dict[str, float] = {}

    # -- keys and (de)serialisation ------------------------------------------------------

    def _k(self, *parts: str) -> str:
        return self.prefix + ":".join(parts)

    def _entry_key(self, entry_id: str) -> str:
        return self._k("e", entry_id)

    @staticmethod
    def _dumps(value: Any, what: str) -> str:
        try:
            return json.dumps(value)
        except TypeError as exc:
            raise TypeError(
                f"RedisStore can only store JSON-serialisable {what}; "
                f"convert it first (e.g. store response.model_dump() or the text)."
            ) from exc

    def _to_mapping(self, entry: CacheEntry) -> dict[str, Any]:
        return {
            "scope": entry.scope,
            "prompt": entry.prompt,
            "normalized_prompt": entry.normalized_prompt,
            "response": self._dumps(entry.response, "responses"),
            "embedding": np.asarray(entry.embedding, dtype=np.float32).tobytes(),
            "model": entry.model if entry.model is not None else "",
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
            "cost": "" if entry.cost is None else repr(float(entry.cost)),
            "metadata": self._dumps(entry.metadata, "metadata"),
            "created_at": repr(float(entry.created_at)),
            "last_hit_at": "" if entry.last_hit_at is None else repr(float(entry.last_hit_at)),
            "hits": entry.hits,
        }

    @staticmethod
    def _from_mapping(entry_id: str, raw: dict) -> CacheEntry:
        h = {_s(k): v for k, v in raw.items()}
        cost, last_hit, model = _s(h.get("cost")), _s(h.get("last_hit_at")), _s(h.get("model"))
        return CacheEntry(
            id=entry_id,
            scope=_s(h["scope"]) or "",
            prompt=_s(h["prompt"]) or "",
            normalized_prompt=_s(h.get("normalized_prompt")) or "",
            response=json.loads(_s(h["response"]) or "null"),
            embedding=np.frombuffer(h["embedding"], dtype=np.float32).copy(),
            model=model or None,
            input_tokens=int(_s(h.get("input_tokens")) or 0),
            output_tokens=int(_s(h.get("output_tokens")) or 0),
            cost=float(cost) if cost else None,
            metadata=json.loads(_s(h.get("metadata")) or "{}"),
            created_at=float(_s(h["created_at"]) or 0),
            last_hit_at=float(last_hit) if last_hit else None,
            hits=int(_s(h.get("hits")) or 0),
        )

    def _load(self, ids: Iterable[str]) -> list[CacheEntry]:
        ids = list(ids)
        if not ids:
            return []
        pipe = self.r.pipeline(transaction=False)
        for entry_id in ids:
            pipe.hgetall(self._entry_key(entry_id))
        return [
            self._from_mapping(entry_id, raw)
            for entry_id, raw in zip(ids, pipe.execute(), strict=True)
            if raw
        ]

    def _ids(self, scope: str) -> list[str]:
        return sorted(_s(m) or "" for m in self.r.smembers(self._k("s", scope)))

    def _scopes(self) -> list[str]:
        return sorted(_s(m) or "" for m in self.r.smembers(self._k("scopes")))

    # -- BaseStore -----------------------------------------------------------------------

    def add(self, entry: CacheEntry) -> None:
        mapping = self._to_mapping(entry)
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(self._entry_key(entry.id), mapping=mapping)
        pipe.sadd(self._k("s", entry.scope), entry.id)
        pipe.sadd(self._k("scopes"), entry.scope)
        if entry.normalized_prompt:
            pipe.hset(self._k("x", entry.scope), entry.normalized_prompt, entry.id)
        pipe.execute()
        with self._lock:
            self._vectors[entry.id] = np.asarray(entry.embedding, dtype=np.float32).ravel()

    def get(self, entry_id: str) -> CacheEntry | None:
        found = self._load([entry_id])
        return found[0] if found else None

    def delete(self, entry_id: str) -> bool:
        key = self._entry_key(entry_id)
        scope, normalized = (_s(v) for v in self.r.hmget(key, "scope", "normalized_prompt"))
        if scope is None:
            return False
        pipe = self.r.pipeline(transaction=True)
        pipe.delete(key)
        pipe.srem(self._k("s", scope), entry_id)
        pipe.execute()
        if normalized and _s(self.r.hget(self._k("x", scope), normalized)) == entry_id:
            self.r.hdel(self._k("x", scope), normalized)
        with self._lock:
            self._vectors.pop(entry_id, None)
        return True

    def clear(self, scope: str | None = None) -> int:
        scopes = self._scopes() if scope is None else [scope]
        removed = 0
        for sc in scopes:
            ids = self._ids(sc)
            pipe = self.r.pipeline(transaction=True)
            for entry_id in ids:
                pipe.delete(self._entry_key(entry_id))
            pipe.delete(self._k("s", sc), self._k("x", sc))
            pipe.srem(self._k("scopes"), sc)
            results = pipe.execute()
            removed += sum(int(r) for r in results[: len(ids)])
            with self._lock:
                for entry_id in ids:
                    self._vectors.pop(entry_id, None)
                self._matrices.pop(sc, None)
        return removed

    def _matrix(self, scope: str) -> tuple[tuple[str, ...], np.ndarray]:
        ids = tuple(self._ids(scope))
        with self._lock:
            cached = self._matrices.get(scope)
            if cached is not None and cached[0] == ids:
                return cached
            missing = [i for i in ids if i not in self._vectors]
        if missing:
            pipe = self.r.pipeline(transaction=False)
            for entry_id in missing:
                pipe.hget(self._entry_key(entry_id), "embedding")
            gone = []
            with self._lock:
                for entry_id, raw in zip(missing, pipe.execute(), strict=True):
                    if raw is None:
                        gone.append(entry_id)  # set member whose hash was removed
                    else:
                        self._vectors[entry_id] = np.frombuffer(raw, dtype=np.float32).copy()
            if gone:
                self.r.srem(self._k("s", scope), *gone)
                ids = tuple(i for i in ids if i not in gone)
        with self._lock:
            matrix = np.vstack([self._vectors[i] for i in ids]) if ids else np.empty((0, 0))
            self._matrices[scope] = (ids, matrix)
            return ids, matrix

    def search(self, vector: np.ndarray, scope: str, k: int = 1) -> list[tuple[CacheEntry, float]]:
        ids, matrix = self._matrix(scope)
        if not ids:
            return []
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
        entries = {e.id: e for e in self._load(ids[i] for i in top)}
        return [(entries[ids[i]], float(scores[i])) for i in top if ids[i] in entries]

    def find_exact(self, normalized_prompt: str, scope: str) -> CacheEntry | None:
        entry_id = _s(self.r.hget(self._k("x", scope), normalized_prompt))
        return self.get(entry_id) if entry_id else None

    def record_hit(self, entry_id: str, when: float) -> None:
        key = self._entry_key(entry_id)
        if not self.r.exists(key):
            return
        pipe = self.r.pipeline(transaction=True)
        pipe.hincrby(key, "hits", 1)
        pipe.hset(key, "last_hit_at", repr(float(when)))
        pipe.execute()

    def entries(self, scope: str | None = None) -> Iterator[CacheEntry]:
        scopes = self._scopes() if scope is None else [scope]
        for sc in scopes:
            yield from self._load(self._ids(sc))

    def __len__(self) -> int:
        scopes = self._scopes()
        if not scopes:
            return 0
        pipe = self.r.pipeline(transaction=False)
        for sc in scopes:
            pipe.scard(self._k("s", sc))
        return sum(int(n) for n in pipe.execute())

    # -- embedder binding and shared stats -----------------------------------------------

    def bind_embedder(self, name: str) -> None:
        current = _s(self.r.hget(self._k("meta"), "embedder"))
        if current is None or len(self) == 0:
            self.r.hset(self._k("meta"), "embedder", name)
        elif current != name:
            raise EmbedderMismatchError(
                f"Redis prefix {self.prefix!r} was built with embedder {current!r}, not "
                f"{name!r}. Use the same embedder, a different prefix, or clear the cache."
            )

    def load_stats(self) -> dict[str, Any] | None:
        raw = self.r.hgetall(self._k("stats"))
        if not raw:
            self._stats_saved = {}
            return None
        flat = {_s(k) or "": float(_s(v) or 0) for k, v in raw.items()}
        self._stats_saved = dict(flat)
        data: dict[str, Any] = {"by_model": {}}
        for key, value in flat.items():
            if key.startswith("by_model."):
                model, field = key[len("by_model.") :].rsplit(".", 1)
                data["by_model"].setdefault(model, {})[field] = value
            else:
                data[key] = value
        for name in list(data):
            if name not in {"cost_saved", "cost_spent", "started_at", "by_model"}:
                data[name] = int(data[name])
        for fields in data["by_model"].values():
            for name in list(fields):
                if name not in {"cost_saved", "cost_spent"}:
                    fields[name] = int(fields[name])
        return data

    def save_stats(self, data: dict[str, Any]) -> None:
        """Add this process's new counts to the shared totals.

        Each process sends only what changed since its last save, so processes
        sharing a cache add up instead of overwriting each other.
        """
        flat = {k: v for k, v in _flatten_stats(data).items() if k.split(".")[-1] not in _DERIVED}
        key = self._k("stats")
        pipe = self.r.pipeline(transaction=True)
        changed = False
        for name, value in flat.items():
            if name == "started_at":
                continue
            delta = value - self._stats_saved.get(name, 0.0)
            if delta:
                pipe.hincrbyfloat(key, name, delta)
                changed = True
        if "started_at" in flat:
            pipe.hsetnx(key, "started_at", flat["started_at"])
            changed = True
        if changed:
            pipe.execute()
        self._stats_saved = flat

    def close(self) -> None:
        if self._owns_client:
            self.r.close()
