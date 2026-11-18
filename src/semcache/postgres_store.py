"""A store kept in PostgreSQL with the pgvector extension.

::

    from semcache import SemanticCache
    from semcache.postgres_store import PostgresStore

    cache = SemanticCache(store=PostgresStore("postgresql://user:pass@db:5432/app"))

Install with ``pip install "semcacher[postgres]"`` (psycopg 3 and the ``pgvector``
adapter). The database needs the `pgvector <https://github.com/pgvector/pgvector>`_
extension; the store runs ``CREATE EXTENSION IF NOT EXISTS vector``, which needs a
role allowed to create it the first time (or create it once yourself).

Every process and server that connects shares the same answers and the same stats.
Similarity search runs in the database, so nothing is loaded into memory:

* ``index="exact"`` (default): an exact search within the namespace, using an index on
  the namespace column. Fast up to tens of thousands of entries per namespace.
* ``index="hnsw"``: also builds a pgvector HNSW index for very large tables. If the
  approximate search finds nothing within the namespace (pgvector before 0.8 applies
  the namespace filter after the index search), the store retries exactly, so a small
  namespace inside a big table is still found.

Tables (``table`` defaults to ``"semcache_entries"``): the entries table, created on the
first write once the vector size is known, plus ``<table>_meta`` (embedder name and
vector size) and ``<table>_stats`` (counters every process adds to).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np

from .redis_store import _DERIVED, _flatten_stats
from .sqlite_store import EmbedderMismatchError
from .store import BaseStore, CacheEntry

__all__ = ["PostgresStore"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_COLUMNS = (
    "id, scope, prompt, normalized_prompt, response, embedding, model, input_tokens, "
    "output_tokens, cost, metadata, created_at, last_hit_at, hits"
)


class PostgresStore(BaseStore):
    """Stores entries, hit counts and stats in PostgreSQL, searched with pgvector.

    Args:
        dsn: a libpq connection string or URL, used when ``conn`` is not given.
        conn: an existing ``psycopg.Connection`` (psycopg 3). It is switched to
            autocommit and the pgvector type is registered on it.
        table: base table name (letters, digits and ``_``), so several caches can share
            one database.
        index: ``"exact"`` (default) or ``"hnsw"``; see the module docs.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        conn: Any = None,
        table: str = "semcache_entries",
        index: str = "exact",
    ) -> None:
        if not _IDENT.match(table):
            raise ValueError("table must be letters, digits and _ (max 41 characters)")
        if index not in ("exact", "hnsw"):
            raise ValueError("index must be 'exact' or 'hnsw'")
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "PostgresStore needs psycopg and pgvector: pip install 'semcacher[postgres]'"
            ) from exc
        if conn is None:
            if dsn is None:
                raise ValueError("pass a dsn or a psycopg connection")
            conn = psycopg.connect(dsn)
            self._owns_conn = True
        else:
            self._owns_conn = False
        self._psycopg = psycopg
        self.conn = conn
        self.conn.autocommit = True
        self.table = table
        self.index = index
        self._lock = threading.RLock()
        self._stats_saved: dict[str, float] = {}
        self._dim: int | None = None
        try:
            self._sql("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as exc:
            raise RuntimeError(
                "pgvector is not available in this database. Install the extension "
                "(https://github.com/pgvector/pgvector) and run CREATE EXTENSION vector "
                "as a role allowed to, then try again."
            ) from exc
        register_vector(self.conn)
        self._sql(f"CREATE TABLE IF NOT EXISTS {table}_meta (key text PRIMARY KEY, value text)")
        self._sql(
            f"CREATE TABLE IF NOT EXISTS {table}_stats "
            "(name text PRIMARY KEY, value double precision NOT NULL)"
        )
        dim = self._meta("dim")
        if dim is not None:
            self._dim = int(dim)

    # -- SQL helpers ------------------------------------------------------------------------

    def _sql(self, query: str, params: Any = None) -> Any:
        with self._lock:
            return self.conn.execute(query, params)

    def _meta(self, key: str) -> str | None:
        row = self._sql(f"SELECT value FROM {self.table}_meta WHERE key = %s", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._sql(
            f"INSERT INTO {self.table}_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def _ensure_table(self, dim: int) -> None:
        """Create the entries table the first time a vector size is known."""
        if self._has_table():
            if dim != self._dim:
                raise ValueError(
                    f"embedding dimension mismatch: table has {self._dim}, entry has {dim}"
                )
            return
        with self._lock:
            try:
                self._create_table(dim)
            except (self._psycopg.errors.UniqueViolation, self._psycopg.errors.DuplicateTable):
                pass  # another process created it at the same moment
            if self._has_table() and dim != self._dim:
                raise ValueError(
                    f"embedding dimension mismatch: table has {self._dim}, entry has {dim}"
                )
            self._set_meta("dim", str(dim))
            self._dim = dim

    def _create_table(self, dim: int) -> None:
        t = self.table
        with self._lock:
            self._sql(
                f"""CREATE TABLE IF NOT EXISTS {t} (
                    id text PRIMARY KEY,
                    scope text NOT NULL,
                    prompt text NOT NULL,
                    normalized_prompt text NOT NULL,
                    response jsonb NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    model text,
                    input_tokens integer NOT NULL DEFAULT 0,
                    output_tokens integer NOT NULL DEFAULT 0,
                    cost double precision,
                    metadata jsonb NOT NULL DEFAULT '{{}}',
                    created_at double precision NOT NULL,
                    last_hit_at double precision,
                    hits integer NOT NULL DEFAULT 0
                )"""
            )
            self._sql(f"CREATE INDEX IF NOT EXISTS {t}_scope ON {t} (scope)")
            self._sql(f"CREATE INDEX IF NOT EXISTS {t}_exact ON {t} (scope, normalized_prompt)")
            if self.index == "hnsw":
                self._sql(
                    f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} "
                    "USING hnsw (embedding vector_ip_ops)"
                )

    def _has_table(self) -> bool:
        if self._dim is None:  # another process may have created it since we connected
            dim = self._meta("dim")
            if dim is not None:
                self._dim = int(dim)
        return self._dim is not None

    @staticmethod
    def _dumps(value: Any, what: str) -> str:
        try:
            return json.dumps(value)
        except TypeError as exc:
            raise TypeError(
                f"PostgresStore can only store JSON-serialisable {what}; "
                f"convert it first (e.g. store response.model_dump() or the text)."
            ) from exc

    @staticmethod
    def _row_to_entry(row: tuple) -> CacheEntry:
        (eid, scope, prompt, norm, response, emb, model, tin, tout, cost, meta, created,
         last_hit, hits) = row  # fmt: skip
        return CacheEntry(
            id=eid,
            scope=scope,
            prompt=prompt,
            normalized_prompt=norm,
            response=response,  # psycopg decodes jsonb
            # pgvector's adapter returns a Vector (newer) or a numpy array (older versions)
            embedding=np.asarray(emb.to_numpy() if hasattr(emb, "to_numpy") else emb, np.float32),
            model=model,
            input_tokens=tin,
            output_tokens=tout,
            cost=cost,
            metadata=meta or {},
            created_at=created,
            last_hit_at=last_hit,
            hits=hits,
        )

    # -- BaseStore ----------------------------------------------------------------------------

    def add(self, entry: CacheEntry) -> None:
        vector = np.asarray(entry.embedding, dtype=np.float32).ravel()
        response = self._dumps(entry.response, "responses")
        metadata = self._dumps(entry.metadata, "metadata")
        self._ensure_table(vector.shape[0])
        self._sql(
            f"""INSERT INTO {self.table} ({_COLUMNS})
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    scope = EXCLUDED.scope, prompt = EXCLUDED.prompt,
                    normalized_prompt = EXCLUDED.normalized_prompt,
                    response = EXCLUDED.response, embedding = EXCLUDED.embedding,
                    model = EXCLUDED.model, input_tokens = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens, cost = EXCLUDED.cost,
                    metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at,
                    last_hit_at = EXCLUDED.last_hit_at, hits = EXCLUDED.hits""",
            (
                entry.id,
                entry.scope,
                entry.prompt,
                entry.normalized_prompt,
                response,
                vector,
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

    def get(self, entry_id: str) -> CacheEntry | None:
        if not self._has_table():
            return None
        row = self._sql(
            f"SELECT {_COLUMNS} FROM {self.table} WHERE id = %s", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def delete(self, entry_id: str) -> bool:
        if not self._has_table():
            return False
        cur = self._sql(f"DELETE FROM {self.table} WHERE id = %s", (entry_id,))
        return cur.rowcount > 0

    def clear(self, scope: str | None = None) -> int:
        if not self._has_table():
            return 0
        if scope is None:
            cur = self._sql(f"DELETE FROM {self.table}")
        else:
            cur = self._sql(f"DELETE FROM {self.table} WHERE scope = %s", (scope,))
        return cur.rowcount

    def search(self, vector: np.ndarray, scope: str, k: int = 1) -> list[tuple[CacheEntry, float]]:
        if not self._has_table() or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).ravel()
        if query.shape[0] != self._dim:
            raise ValueError(
                f"embedding dimension mismatch: table has {self._dim}, query has {query.shape[0]}"
            )
        # <#> is the negative inner product; vectors are unit length, so -(a <#> b) is cosine.
        order = "embedding <#> %(q)s" if self.index == "hnsw" else "(embedding <#> %(q)s) + 0"
        sql = (
            f"SELECT {_COLUMNS}, -(embedding <#> %(q)s) AS score FROM {self.table} "
            f"WHERE scope = %(scope)s ORDER BY {order} LIMIT %(k)s"
        )
        params = {"q": query, "scope": scope, "k": k}
        rows = self._sql(sql, params).fetchall()
        if not rows and self.index == "hnsw":
            # The HNSW scan can miss a small namespace entirely; search it exactly instead.
            exact = sql.replace(f"ORDER BY {order}", "ORDER BY (embedding <#> %(q)s) + 0")
            rows = self._sql(exact, params).fetchall()
        return [(self._row_to_entry(row[:-1]), float(row[-1])) for row in rows]

    def find_exact(self, normalized_prompt: str, scope: str) -> CacheEntry | None:
        if not self._has_table():
            return None
        row = self._sql(
            f"SELECT {_COLUMNS} FROM {self.table} WHERE scope = %s AND normalized_prompt = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (scope, normalized_prompt),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def record_hit(self, entry_id: str, when: float) -> None:
        if self._has_table():
            self._sql(
                f"UPDATE {self.table} SET hits = hits + 1, last_hit_at = %s WHERE id = %s",
                (when, entry_id),
            )

    def entries(self, scope: str | None = None) -> Iterator[CacheEntry]:
        if not self._has_table():
            return iter(())
        if scope is None:
            rows = self._sql(f"SELECT {_COLUMNS} FROM {self.table}").fetchall()
        else:
            rows = self._sql(
                f"SELECT {_COLUMNS} FROM {self.table} WHERE scope = %s", (scope,)
            ).fetchall()
        return iter([self._row_to_entry(row) for row in rows])

    def __len__(self) -> int:
        if not self._has_table():
            return 0
        return int(self._sql(f"SELECT count(*) FROM {self.table}").fetchone()[0])

    # -- embedder binding and shared stats -----------------------------------------------------

    def bind_embedder(self, name: str) -> None:
        current = self._meta("embedder")
        if current is None or len(self) == 0:
            self._set_meta("embedder", name)
        elif current != name:
            raise EmbedderMismatchError(
                f"table {self.table!r} was built with embedder {current!r}, not {name!r}. "
                "Use the same embedder, a different table, or clear the cache."
            )

    def load_stats(self) -> dict[str, Any] | None:
        rows = self._sql(f"SELECT name, value FROM {self.table}_stats").fetchall()
        if not rows:
            self._stats_saved = {}
            return None
        flat = {name: float(value) for name, value in rows}
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
        """Add this process's new counts to the shared totals (see ``RedisStore``)."""
        flat = {k: v for k, v in _flatten_stats(data).items() if k.split(".")[-1] not in _DERIVED}
        t = f"{self.table}_stats"
        with self._lock, self.conn.transaction():
            for name, value in flat.items():
                if name == "started_at":
                    self.conn.execute(
                        f"INSERT INTO {t} (name, value) VALUES (%s, %s) "
                        "ON CONFLICT (name) DO NOTHING",
                        (name, value),
                    )
                    continue
                delta = value - self._stats_saved.get(name, 0.0)
                if delta:
                    self.conn.execute(
                        f"INSERT INTO {t} (name, value) VALUES (%s, %s) "
                        f"ON CONFLICT (name) DO UPDATE SET value = {t}.value + EXCLUDED.value",
                        (name, delta),
                    )
        self._stats_saved = flat

    def close(self) -> None:
        if self._owns_conn:
            with self._lock:
                self.conn.close()
