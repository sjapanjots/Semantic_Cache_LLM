import time

import pytest

from semcache import EmbedderMismatchError, HashingEmbedder, SemanticCache, SQLiteStore


def test_entries_survive_reopen(tmp_path):
    db = tmp_path / "cache.db"
    with SemanticCache(path=db) as cache:
        cache.set("How do I reverse a list in Python?", {"text": "list[::-1]"}, cost=0.02)
    with SemanticCache(path=db) as cache:
        assert len(cache) == 1
        hit = cache.lookup("python: how to reverse a list")
        assert hit is not None and hit.response == {"text": "list[::-1]"}
        assert hit.entry.cost == 0.02


def test_stats_survive_reopen(tmp_path):
    db = tmp_path / "cache.db"
    with SemanticCache(path=db) as cache:
        cache.set("q", "a", cost=1.5)
        cache.get("q")
        cache.get("unrelated question entirely")
    with SemanticCache(path=db) as cache:
        s = cache.stats
        assert (s.hits, s.misses, s.writes) == (1, 1, 1)
        assert s.cost_saved == pytest.approx(1.5)


def test_hits_persisted(tmp_path):
    db = tmp_path / "cache.db"
    with SemanticCache(path=db) as cache:
        entry = cache.set("q", "a")
        cache.get("q")
    store = SQLiteStore(db)
    assert store.get(entry.id).hits == 1
    store.close()


def test_embedder_mismatch_is_rejected(tmp_path):
    db = tmp_path / "cache.db"
    with SemanticCache(path=db) as cache:
        cache.set("q", "a")
    with pytest.raises(EmbedderMismatchError):
        SemanticCache(path=db, embedder=HashingEmbedder(dim=64))


def test_non_json_response_rejected(tmp_path):
    with SemanticCache(path=tmp_path / "c.db") as cache:
        with pytest.raises(TypeError, match="JSON"):
            cache.set("q", object())


def test_store_and_path_are_exclusive(tmp_path):
    with pytest.raises(ValueError):
        SemanticCache(store=SQLiteStore(":memory:"), path=tmp_path / "x.db")


def test_clear_and_delete_hit_disk(tmp_path):
    db = tmp_path / "cache.db"
    with SemanticCache(path=db) as cache:
        a = cache.set("one", "1")
        cache.set("two", "2", namespace="other")
        cache.delete(a.id)
        cache.clear("other")
    with SemanticCache(path=db) as cache:
        assert len(cache) == 0


def test_ttl_expires_entries(monkeypatch):
    cache = SemanticCache(ttl=10)
    now = time.time()
    cache.set("How do I reverse a list in Python?", "list[::-1]")
    assert cache.get("python: how to reverse a list") is not None
    monkeypatch.setattr(time, "time", lambda: now + 11)
    assert cache.get("python: how to reverse a list") is None
    assert cache.get("How do I reverse a list in Python?") is None
    assert len(cache) == 0


def test_prune_expired(monkeypatch):
    cache = SemanticCache(ttl=5)
    now = time.time()
    cache.set("a question", "1")
    cache.set("another question", "2")
    monkeypatch.setattr(time, "time", lambda: now + 6)
    assert cache.prune_expired() == 2
    assert SemanticCache().prune_expired() == 0


def test_max_entries_evicts_least_recently_used(monkeypatch):
    clock = iter(range(1000, 2000))
    monkeypatch.setattr(time, "time", lambda: float(next(clock)))
    cache = SemanticCache(max_entries=2)
    cache.set("alpha question", "a")
    cache.set("bravo question", "b")
    cache.get("alpha question")  # alpha is now more recent than bravo
    cache.set("charlie question", "c")
    assert len(cache) == 2
    assert cache.get("bravo question") is None
    assert cache.get("alpha question") == "a"


@pytest.mark.parametrize("kwargs", [{"ttl": 0}, {"max_entries": 0}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        SemanticCache(**kwargs)
