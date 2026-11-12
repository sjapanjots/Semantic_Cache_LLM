import pytest

from semcache import InMemoryStore, SemanticCache


@pytest.fixture
def cache():
    return SemanticCache()


def test_miss_then_hit_exact(cache):
    assert cache.get("What is the capital of France?") is None
    cache.set("What is the capital of France?", "Paris")
    hit = cache.lookup("what is the capital of france")
    assert hit is not None and hit.exact and hit.response == "Paris"


def test_semantic_hit(cache):
    cache.set("How do I reverse a list in Python?", "Use reversed() or list[::-1]")
    hit = cache.lookup("python: how to reverse a list")
    assert hit is not None
    assert not hit.exact
    assert hit.similarity >= cache.threshold


def test_unrelated_prompt_misses(cache):
    cache.set("How do I reverse a list in Python?", "list[::-1]")
    assert cache.get("What is the best pizza topping?") is None


def test_default_threshold_comes_from_embedder(cache):
    assert cache.threshold == 0.7


def test_per_call_threshold_override(cache):
    cache.set("How do I reverse a list in Python?", "list[::-1]")
    assert cache.lookup("python: how to reverse a list", threshold=0.999) is None


def test_models_and_namespaces_are_isolated(cache):
    cache.set("hello there", "from gpt", model="model-a")
    assert cache.get("hello there", model="model-b") is None
    assert cache.get("hello there", model="model-a") == "from gpt"
    cache.set("hello there", "team b", namespace="b")
    assert cache.get("hello there") is None
    assert cache.get("hello there", namespace="b") == "team b"


def test_set_overwrites_same_prompt(cache):
    cache.set("Q?", "old")
    cache.set("q", "new")
    assert len(cache) == 1
    assert cache.get("Q") == "new"


def test_hits_are_recorded(cache):
    entry = cache.set("What is the capital of France?", "Paris")
    cache.get("What is the capital of France?")
    cache.get("what is the capital of france?")
    assert cache.store.get(entry.id).hits == 2


def test_clear_namespace(cache):
    cache.set("one", "1", namespace="a")
    cache.set("two", "2", namespace="a", model="m")
    cache.set("three", "3", namespace="b")
    assert cache.clear("a") == 2
    assert len(cache) == 1
    assert cache.clear() == 1


def test_contains_and_delete(cache):
    entry = cache.set("ping", "pong")
    assert "ping" in cache
    assert cache.delete(entry.id)
    assert "ping" not in cache


def test_invalid_threshold():
    with pytest.raises(ValueError):
        SemanticCache(threshold=0)


def test_store_search_orders_by_similarity():
    cache = SemanticCache(store=InMemoryStore())
    cache.set("reverse a python list", "a")
    cache.set("sort a python list", "b")
    cache.set("bake sourdough bread", "c")
    q = cache.embedder.embed(["how to reverse a list in python"])[0]
    results = cache.store.search(q, "default::*", k=3)
    assert [e.response for e, _ in results][0] == "a"
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_questions_differing_only_in_symbols_are_different(cache):
    cache.set("What is 2+2?", "4")
    cache.set("What is 2-2?", "0")
    assert len(cache) == 2
    assert cache.get("What is 2+2?") == "4"
    assert cache.get("what is 2-2") == "0"
    assert cache.get("What is 2*2?") is None


def test_comparison_operators_are_not_merged(cache):
    cache.set("Is x > 5 here?", "yes")
    assert cache.get("Is x < 5 here?") is None
    assert cache.get("is x > 5 here") == "yes"


def test_clear_namespace_includes_wrapper_namespaces(cache):
    from semcache.cache import in_namespace

    cache.set("a", "1", namespace="anthropic/1a2b")
    cache.set("b", "2", namespace="anthropic")
    cache.set("c", "3", namespace="anthropic-other")
    assert cache.clear("anthropic") == 2
    assert len(cache) == 1
    assert in_namespace("anthropic/1a2b::m", "anthropic")
    assert not in_namespace("anthropic-other::*", "anthropic")
