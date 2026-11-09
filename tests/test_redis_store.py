import time

import pytest

from semcache import EmbedderMismatchError, HashingEmbedder, SemanticCache

fakeredis = pytest.importorskip("fakeredis")
from semcache.redis_store import RedisStore  # noqa: E402


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def make_store(server, **kwargs):
    return RedisStore(client=fakeredis.FakeRedis(server=server), **kwargs)


def make_cache(server, **kwargs):
    store_kwargs = {"prefix": kwargs.pop("prefix")} if "prefix" in kwargs else {}
    return SemanticCache(store=make_store(server, **store_kwargs), **kwargs)


def test_set_get_and_semantic_hit(server):
    cache = make_cache(server)
    cache.set("How do I reverse a list in Python?", {"text": "list[::-1]"}, cost=0.02)
    hit = cache.lookup("python: how to reverse a list")
    assert hit is not None and not hit.exact
    assert hit.response == {"text": "list[::-1]"}
    assert hit.entry.cost == 0.02
    assert cache.get("How do I reverse a list in Python") == {"text": "list[::-1]"}
    assert cache.get("What is the best pizza topping?") is None


def test_entry_round_trip(server):
    store = make_store(server)
    cache = SemanticCache(store=store)
    entry = cache.set(
        "q", ["a", 1], model="m", input_tokens=3, output_tokens=4, metadata={"k": "v"}
    )
    loaded = store.get(entry.id)
    assert (loaded.prompt, loaded.response, loaded.model) == ("q", ["a", 1], "m")
    assert (loaded.input_tokens, loaded.output_tokens, loaded.metadata) == (3, 4, {"k": "v"})
    assert loaded.cost is None and loaded.last_hit_at is None
    assert loaded.created_at == pytest.approx(entry.created_at)
    assert store.get("missing") is None


def test_two_processes_share_answers(server):
    a = make_cache(server)
    b = make_cache(server)
    a.set("What is the capital of France?", "Paris")
    assert b.get("what's the capital city of France") == "Paris"
    b.set("Who wrote Hamlet?", "Shakespeare")
    assert a.get("who wrote hamlet") == "Shakespeare"  # a sees b's later write


def test_search_notices_deletes_from_other_process(server):
    a = make_cache(server)
    b = make_cache(server)
    entry = a.set("How do I reverse a list in Python?", "x")
    assert b.get("python: how to reverse a list") == "x"
    a.delete(entry.id)
    assert b.get("python: how to reverse a list") is None


def test_hits_recorded(server):
    store = make_store(server)
    cache = SemanticCache(store=store)
    entry = cache.set("q", "a")
    cache.get("q")
    cache.get("Q!")
    loaded = store.get(entry.id)
    assert loaded.hits == 2 and loaded.last_hit_at is not None


def test_stats_add_up_across_processes(server):
    a = make_cache(server, stats_flush_every=1)
    b = make_cache(server, stats_flush_every=1)
    a.set("q", "a", cost=1.0)
    a.get("q")
    b.get("q")
    b.get("q")
    b.get("never asked before")
    a.close()
    b.close()
    s = make_cache(server).stats
    assert (s.hits, s.misses, s.writes) == (3, 1, 1)
    assert s.cost_saved == pytest.approx(3.0)
    assert s.cost_spent == pytest.approx(1.0)
    assert s.by_model["unknown"].hits == 3


def test_stats_resume_without_double_counting(server):
    with make_cache(server) as cache:
        cache.set("q", "a", cost=0.5)
        cache.get("q")
    with make_cache(server) as cache:
        cache.get("q")
    s = make_cache(server).stats
    assert s.hits == 2 and s.cost_saved == pytest.approx(1.0)


def test_delete_clear_and_len(server):
    cache = make_cache(server)
    a = cache.set("one", "1")
    cache.set("two", "2", namespace="other")
    cache.set("three", "3", namespace="other", model="m")
    assert len(cache) == 3
    assert cache.delete(a.id) and not cache.delete(a.id)
    assert cache.clear("other") == 2
    assert len(cache) == 0
    cache.set("four", "4")
    assert cache.clear() == 1
    assert list(cache.store.entries()) == []


def test_overwrite_same_prompt(server):
    cache = make_cache(server)
    cache.set("Q?", "old")
    cache.set("q", "new")
    assert len(cache) == 1 and cache.get("q") == "new"


def test_prefixes_are_isolated(server):
    a = make_cache(server, prefix="team-a:")
    b = make_cache(server, prefix="team-b:")
    a.set("hello there", "from a")
    assert b.get("hello there") is None
    assert len(b) == 0


def test_embedder_mismatch_is_rejected(server):
    make_cache(server).set("q", "a")
    with pytest.raises(EmbedderMismatchError):
        SemanticCache(store=make_store(server), embedder=HashingEmbedder(dim=64))


def test_non_json_response_rejected(server):
    with pytest.raises(TypeError, match="JSON"):
        make_cache(server).set("q", object())


def test_dangling_ids_are_cleaned_up(server):
    store = make_store(server)
    cache = SemanticCache(store=store)
    entry = cache.set("How do I reverse a list in Python?", "x")
    store.r.delete(store._entry_key(entry.id))  # e.g. evicted by Redis maxmemory
    other = SemanticCache(store=make_store(server))
    assert other.get("python: how to reverse a list") is None
    assert len(store) == 0


def test_ttl_and_max_entries(server, monkeypatch):
    now = time.time()
    cache = make_cache(server, ttl=10)
    cache.set("How do I reverse a list in Python?", "x")
    monkeypatch.setattr(time, "time", lambda: now + 11)
    assert cache.get("python: how to reverse a list") is None
    assert len(cache) == 0

    clock = iter(range(1000, 2000))
    monkeypatch.setattr(time, "time", lambda: float(next(clock)))
    small = make_cache(fakeredis.FakeServer(), max_entries=2)
    small.set("alpha question", "a")
    small.set("bravo question", "b")
    small.get("alpha question")
    small.set("charlie question", "c")
    assert len(small) == 2 and small.get("bravo question") is None


def test_empty_scope_search(server):
    store = make_store(server)
    assert store.search(HashingEmbedder().embed(["x"])[0], "nothing::*") == []
    assert store.find_exact("x", "nothing::*") is None
