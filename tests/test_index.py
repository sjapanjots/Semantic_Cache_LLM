import numpy as np
import pytest

from semcache import InMemoryStore, SemanticCache, SQLiteStore
from semcache.embedders import l2_normalize
from semcache.index import ExactIndex, make_index

hnswlib = pytest.importorskip("hnswlib")
from semcache.index import HNSWIndex  # noqa: E402

INDEXES = [ExactIndex, HNSWIndex]


def unit(*values):
    return l2_normalize(np.array(values, dtype=np.float32))[0]


@pytest.mark.parametrize("cls", INDEXES)
def test_add_search_remove_replace(cls):
    idx = cls(3)
    idx.add("x", unit(1, 0, 0))
    idx.add("y", unit(0, 1, 0))
    idx.add("xy", unit(1, 1, 0))
    assert len(idx) == 3
    results = idx.search(unit(1, 0.1, 0), k=3)
    assert [k for k, _ in results] == ["x", "xy", "y"]
    assert results[0][1] == pytest.approx(float(unit(1, 0.1, 0) @ unit(1, 0, 0)), abs=1e-5)

    idx.remove("x")
    idx.remove("missing")  # ignored
    assert len(idx) == 2
    assert idx.search(unit(1, 0, 0), k=1)[0][0] == "xy"

    idx.add("y", unit(0, 0, 1))  # replace y's vector
    assert len(idx) == 2
    assert idx.search(unit(0, 0, 1), k=1) == [("y", pytest.approx(1.0, abs=1e-5))]


@pytest.mark.parametrize("cls", INDEXES)
def test_empty_k_and_dimension_checks(cls):
    idx = cls(4)
    assert idx.search(unit(1, 0, 0, 0), k=3) == []
    idx.add("a", unit(1, 0, 0, 0))
    assert len(idx.search(unit(1, 0, 0, 0), k=10)) == 1  # k larger than the index
    assert idx.search(unit(1, 0, 0, 0), k=0) == []
    with pytest.raises(ValueError, match="dimension"):
        idx.search(unit(1, 0, 0), k=1)
    with pytest.raises(ValueError, match="dimension"):
        idx.add("b", unit(1, 0))
    with pytest.raises(ValueError):
        cls(0)


@pytest.mark.parametrize("cls", INDEXES)
def test_grows_past_initial_capacity_and_survives_churn(cls):
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.normal(size=(400, 16)))
    idx = cls(16, initial_capacity=8)
    for i in range(300):
        idx.add(str(i), vectors[i])
    for i in range(0, 300, 2):  # delete every other entry
        idx.remove(str(i))
    for i in range(300, 400):  # and add new ones into the freed space
        idx.add(str(i), vectors[i])
    assert len(idx) == 250
    for i in (1, 151, 299, 350):
        assert idx.search(vectors[i], k=1)[0][0] == str(i)
    found = {key for key, _ in idx.search(vectors[0], k=250)}
    assert "0" not in found  # removed keys never come back


def test_hnsw_recall_against_exact():
    rng = np.random.default_rng(42)
    data = l2_normalize(rng.normal(size=(3000, 32)))
    queries = l2_normalize(data[:200] + rng.normal(scale=0.3, size=(200, 32)))
    exact, hnsw = ExactIndex(32), HNSWIndex(32)
    for i, v in enumerate(data):
        exact.add(str(i), v)
        hnsw.add(str(i), v)
    hits = sum(exact.search(q, 1)[0][0] == hnsw.search(q, 1)[0][0] for q in queries)
    assert hits / len(queries) >= 0.97


def test_make_index_and_validation():
    assert isinstance(make_index(None, 4), ExactIndex)
    assert isinstance(make_index("exact", 4), ExactIndex)
    assert isinstance(make_index("hnsw", 4), HNSWIndex)
    assert make_index(lambda dim: ExactIndex(dim, initial_capacity=2), 4).dim == 4
    with pytest.raises(ValueError):
        make_index("faiss", 4)
    with pytest.raises(ValueError):
        InMemoryStore(index="faiss")
    with pytest.raises(ValueError):
        SemanticCache(store=InMemoryStore(), index="hnsw")


@pytest.mark.parametrize("index", ["exact", "hnsw"])
def test_cache_end_to_end(index):
    cache = SemanticCache(index=index, max_entries=50)
    for i in range(60):  # more than max_entries: the oldest fillers are evicted
        cache.set(f"filler question number {i} about topic {i * 7}", i)
    cache.set("How do I reverse a list in Python?", "list[::-1]")
    cache.set("What is the capital of France?", "Paris", namespace="geo")
    assert cache.get("What is the capital of France?", namespace="geo") == "Paris"
    assert cache.get("what's the capital city of France", namespace="geo") == "Paris"
    assert cache.get("What is the best pizza topping?") is None
    assert len(cache) <= 50
    entry = cache.set("C++ tutorial", "x")
    assert cache.get("C++ tutorial") == "x"
    cache.delete(entry.id)
    assert cache.get("C++ tutorial") is None
    assert cache.clear("geo") == 1


def test_sqlite_store_rebuilds_hnsw_index_on_open(tmp_path):
    db = tmp_path / "cache.db"
    with SemanticCache(path=db, index="hnsw") as cache:
        cache.set("How do I reverse a list in Python?", "list[::-1]")
    store = SQLiteStore(db, index="hnsw")
    with SemanticCache(store=store) as cache:
        assert cache.get("python: how to reverse a list") == "list[::-1]"


@pytest.mark.parametrize("cls", INDEXES)
def test_bulk_add(cls):
    from semcache.index import add_many

    rng = np.random.default_rng(1)
    vectors = l2_normalize(rng.normal(size=(500, 8)))
    idx = cls(8, initial_capacity=4)
    idx.add("0", vectors[499])  # replaced by the bulk add below
    add_many(idx, [str(i) for i in range(500)], vectors)
    assert len(idx) == 500
    for i in (0, 250, 499):
        assert idx.search(vectors[i], k=1)[0][0] == str(i)
    add_many(idx, ["dup", "dup"], vectors[:2])  # the last vector for a key wins
    assert idx.search(vectors[1], k=1)[0][0] in {"dup", "1"}
    assert len(idx) == 501


def test_in_memory_store_add_many_replaces_existing():
    from semcache.store import CacheEntry

    vectors = l2_normalize(np.eye(3, dtype=np.float32))
    store = InMemoryStore(index="hnsw")
    first = CacheEntry(prompt="a", response=1, embedding=vectors[0], id="x")
    store.add(first)
    store.add_many(
        [
            CacheEntry(prompt="a2", response=2, embedding=vectors[1], id="x"),
            CacheEntry(prompt="b", response=3, embedding=vectors[2], id="y", scope="other"),
        ]
    )
    assert len(store) == 2 and store.get("x").response == 2
    assert store.search(vectors[1], "default", 1)[0][0].id == "x"
    assert store.search(vectors[2], "other", 1)[0][0].id == "y"
