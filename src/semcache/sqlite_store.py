"""A persistent store backed by a single SQLite file."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .index import IndexSpec
from .store import BaseStore, CacheEntry, InMemoryStore

__all__ = ["SQLiteStore", "EmbedderMismatchError"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id                TEXT PRIMARY KEY,
    scope             TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    normalized_prompt TEXT NOT NULL,
    response          TEXT NOT NULL,
    embedding         BLOB NOT NULL,
    model             TEXT,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cost              REAL,
    metadata          TEXT NOT NULL DEFAULT '{}',
    created_at        REAL NOT NULL,
    last_hit_at       REAL,
    hits              INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS entries_scope ON entries(scope);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class EmbedderMismatchError(RuntimeError):
    """The cache file was built with a different embedder; its vectors are not comparable."""


class SQLiteStore(BaseStore):
    """Persists entries and stats to SQLite, and searches an in-memory mirror.

    All rows are loaded into memory when the store opens, so lookups are as fast as
    :class:`InMemoryStore`; every write goes to both. Responses must be JSON-serialisable
    (strings, numbers, lists, dicts).

    Writes from *other* processes become visible after :meth:`reload`. ``index`` picks
    how vectors are searched (see :mod:`semcache.index`); it is rebuilt from the file
    when the store opens.
    """

    def __init__(self, path: str | Path = "semcache.db", *, index: IndexSpec = None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._mem = InMemoryStore(index=index)
        self.reload()

    # -- (de)serialisation -------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: tuple) -> CacheEntry:
        (eid, scope, prompt, norm, response, emb, model, tin, tout, cost, meta, created,
         last_hit, hits) = row  # fmt: skip
        return CacheEntry(
            id=eid,
            scope=scope,
            prompt=prompt,
            normalized_prompt=norm,
            response=json.loads(response),
            embedding=np.frombuffer(emb, dtype=np.float32).copy(),
            model=model,
            input_tokens=tin,
            output_tokens=tout,
            cost=cost,
            metadata=json.loads(meta),
            created_at=created,
            last_hit_at=last_hit,
            hits=hits,
        )

    @staticmethod
    def _dumps(value: Any, what: str) -> str:
        try:
            return json.dumps(value)
        except TypeError as exc:
            raise TypeError(
                f"SQLiteStore can only persist JSON-serialisable {what}; "
                f"convert it first (e.g. store response.model_dump() or the text)."
            ) from exc

    # -- BaseStore ---------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read every row from disk into the in-memory index."""
        with self._lock:
            self._mem.clear()
            rows = self._conn.execute("SELECT * FROM entries").fetchall()
            self._mem.add_many([self._row_to_entry(row) for row in rows])

    def add(self, entry: CacheEntry) -> None:
        response = self._dumps(entry.response, "responses")
        metadata = self._dumps(entry.metadata, "metadata")
        embedding = np.asarray(entry.embedding, dtype=np.float32).tobytes()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entry.id,
                    entry.scope,
                    entry.prompt,
                    entry.normalized_prompt,
                    response,
                    embedding,
                    entry.model,
                    entry.input_tokens,
                    entry.output_tokens,
                    entry.cost,
                    metadata,
                    entry.created_at,
                    entry.last_hit_at,
                    entry.hits,
                ),
            )
            self._mem.add(entry)

    def get(self, entry_id: str) -> CacheEntry | None:
        return self._mem.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            return self._mem.delete(entry_id)

    def clear(self, scope: str | None = None) -> int:
        with self._lock, self._conn:
            if scope is None:
                self._conn.execute("DELETE FROM entries")
            else:
                self._conn.execute("DELETE FROM entries WHERE scope = ?", (scope,))
            return self._mem.clear(scope)

    def search(self, vector: np.ndarray, scope: str, k: int = 1) -> list[tuple[CacheEntry, float]]:
        return self._mem.search(vector, scope, k)

    def find_exact(self, normalized_prompt: str, scope: str) -> CacheEntry | None:
        return self._mem.find_exact(normalized_prompt, scope)

    def record_hit(self, entry_id: str, when: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE entries SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                (when, entry_id),
            )
            self._mem.record_hit(entry_id, when)

    def entries(self, scope: str | None = None) -> Iterator[CacheEntry]:
        return self._mem.entries(scope)

    def __len__(self) -> int:
        return len(self._mem)

    # -- metadata, embedder binding and stats ------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def bind_embedder(self, name: str) -> None:
        current = self.get_meta("embedder")
        if current is None or len(self) == 0:
            self.set_meta("embedder", name)
        elif current != name:
            raise EmbedderMismatchError(
                f"{self.path} was built with embedder {current!r}, not {name!r}. "
                "Use the same embedder, a different file, or clear the cache."
            )

    def load_stats(self) -> dict[str, Any] | None:
        raw = self.get_meta("stats")
        return json.loads(raw) if raw else None

    def save_stats(self, data: dict[str, Any]) -> None:
        self.set_meta("stats", json.dumps(data))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
