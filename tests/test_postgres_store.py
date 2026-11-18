"""PostgresStore against a real PostgreSQL + pgvector.

Set SEMCACHE_TEST_POSTGRES_DSN (e.g. postgresql://postgres@localhost:5432/postgres) to run
these; CI starts a pgvector service container for them. Each test uses its own tables.
"""

import os
import time
import uuid

import pytest

from semcache import EmbedderMismatchError, HashingEmbedder, SemanticCache

DSN = os.environ.get("SEMCACHE_TEST_POSTGRES_DSN")
pytest.importorskip("psycopg")
pytest.importorskip("pgvector")
pytestmark = pytest.mark.skipif(not DSN, reason="SEMCACHE_TEST_POSTGRES_DSN not set")

from semcache.postgres_store import PostgresStore  # noqa: E402

Q = "How do I reverse a list in Python?"
PARA = "python: how to reverse a list"


@pytest.fixture
def table():
    name = "t_" + uuid.uuid4().hex[:12]
    yield name
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        for suffix in ("", "_meta", "_stats"):
            conn.execute(f"DROP TABLE IF EXISTS {name}{suffix}")


def make_cache(table, **kwargs):
    index = kwargs.pop("index", "exact")
    return SemanticCache(store=PostgresStore(DSN, table=table, index=index), **kwargs)


@pytest.mark.parametrize("index", ["exact", "hnsw"])
def test_set_get_semantic_and_exact(table, index):
    cache = make_cache(table, index=index)
    cache.set(Q, {"text": "list[::-1]"}, model="m", input_tokens=5, cost=0.02, metadata={"k": 1})
    hit = cache.lookup(PARA, model="m")
    assert hit is not None and not hit.exact
    assert hit.response == {"text": "list[::-1]"}
    assert (hit.entry.input_tokens, hit.entry.cost, hit.entry.metadata) == (5, 0.02, {"k": 1})
    assert cache.get(Q.lower(), model="m") == {"text": "list[::-1]"}
    assert cache.get("What is the best pizza topping?", model="m") is None
    assert cache.get(PARA, model="other-model") is None


def test_round_trip_types(table):
    store = PostgresStore(DSN, table=table)
    cache = SemanticCache(store=store)
    for value in ("plain text", 42, [1, "two"], {"nested": {"a": [1.5, None]}}, None):
        entry = cache.set(f"question about {value!r}", value)
        assert store.get(entry.id).response == value
    assert store.get("missing") is None


def test_two_processes_share_answers_and_see_new_tables(table):
    a = make_cache(table)
    b = make_cache(table)  # connected before the entries table exists
    assert b.get(Q) is None
    a.set(Q, "x")
    assert b.get(PARA) == "x"  # b notices the table a created
    entry = b.set("Who wrote Hamlet?", "Shakespeare")
    assert a.get("who wrote hamlet") == "Shakespeare"
    a.delete(entry.id)
    assert b.get("Who wrote Hamlet?") is None


def test_stats_add_up_across_processes(table):
    a = make_cache(table, stats_flush_every=1)
    b = make_cache(table, stats_flush_every=1)
    a.set("q", "a", cost=1.0)
    a.get("q")
    b.get("q")
    b.get("q")
    b.get("never asked before")
    a.close()
    b.close()
    s = make_cache(table).stats
    assert (s.hits, s.misses, s.writes) == (3, 1, 1)
    assert s.cost_saved == pytest.approx(3.0)
    with make_cache(table) as again:  # resuming does not double count
        again.get("q")
    assert make_cache(table).stats.hits == 4


def test_hits_delete_clear_len_and_overwrite(table):
    store = PostgresStore(DSN, table=table)
    cache = SemanticCache(store=store)
    entry = cache.set("Q?", "old")
    cache.set("q", "new")  # same question: replaces the entry
    assert len(cache) == 1 and cache.get("q") == "new"
    cache.get("Q")
    assert store.find_exact("q", "default::*").hits == 2
    cache.set("two", "2", namespace="other")
    cache.set("three", "3", namespace="other", model="m")
    assert cache.clear("other") == 2
    assert not cache.delete(entry.id)  # already replaced
    assert cache.clear() == 1 and len(cache) == 0


def test_tables_are_isolated(table):
    other = table + "_b"
    try:
        a = make_cache(table)
        b = make_cache(other)
        a.set("hello there", "from a")
        assert b.get("hello there") is None and len(b) == 0
    finally:
        import psycopg

        with psycopg.connect(DSN, autocommit=True) as conn:
            for suffix in ("", "_meta", "_stats"):
                conn.execute(f"DROP TABLE IF EXISTS {other}{suffix}")


def test_embedder_and_dimension_mismatch(table):
    make_cache(table).set("q", "a")
    with pytest.raises(EmbedderMismatchError):
        SemanticCache(store=PostgresStore(DSN, table=table), embedder=HashingEmbedder(dim=64))
    store = PostgresStore(DSN, table=table)
    with pytest.raises(ValueError, match="dimension"):
        store.search(HashingEmbedder(dim=64).embed(["x"])[0], "default::*")


def test_ttl_max_entries_and_non_json(table, monkeypatch):
    with pytest.raises(TypeError, match="JSON"):
        make_cache(table).set("q", object())
    now = time.time()
    cache = make_cache(table, ttl=10)
    cache.set(Q, "x")
    monkeypatch.setattr(time, "time", lambda: now + 11)
    assert cache.get(PARA) is None and len(cache) == 0
    monkeypatch.undo()
    small = make_cache(table, max_entries=3)
    for i in range(5):
        small.set(f"filler question number {i} about topic {i * 7}", i)
    assert len(small) <= 3


def test_hnsw_finds_small_namespace_in_big_table(table):
    store = PostgresStore(DSN, table=table, index="hnsw")
    cache = SemanticCache(store=store)
    for i in range(300):
        cache.set(f"filler question number {i} about topic {i * 7}", i, namespace="big")
    cache.set(Q, "x", namespace="tiny")
    assert cache.get(PARA, namespace="tiny") == "x"


def test_validation():
    with pytest.raises(ValueError):
        PostgresStore(DSN, table="bad-name; DROP TABLE x")
    with pytest.raises(ValueError):
        PostgresStore(DSN, index="ivfflat")
    with pytest.raises(ValueError):
        PostgresStore()
