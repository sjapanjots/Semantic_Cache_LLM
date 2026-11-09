import pytest

from semcache import CacheStats, SemanticCache


@pytest.fixture
def cache():
    return SemanticCache(pricing={"test-model": (1.0, 2.0)})


def test_hit_rate_and_money_saved(cache):
    cache.set(
        "What is the capital of France?",
        "Paris",
        model="test-model",
        input_tokens=1_000_000,
        output_tokens=500_000,
    )  # costs $1 + $1 = $2
    assert cache.get("What is the capital of France?", model="test-model") == "Paris"
    assert cache.get("what is the capital of france", model="test-model") == "Paris"
    assert cache.get("Who painted the Mona Lisa?", model="test-model") is None

    s = cache.stats
    assert s.lookups == 3
    assert s.hits == 2 and s.misses == 1
    assert s.exact_hits == 2
    assert s.hit_rate == pytest.approx(2 / 3)
    assert s.writes == 1
    assert s.cost_spent == pytest.approx(2.0)
    assert s.cost_saved == pytest.approx(4.0)
    assert s.savings_rate == pytest.approx(4 / 6)
    assert s.tokens_saved == 3_000_000
    assert s.by_model["test-model"].hits == 2
    assert s.by_model["test-model"].hit_rate == pytest.approx(2 / 3)


def test_semantic_hits_counted(cache):
    cache.set("How do I reverse a list in Python?", "list[::-1]", cost=0.01)
    hit = cache.lookup("python: how to reverse a list")
    assert hit is not None and hit.cost_saved == pytest.approx(0.01)
    assert cache.stats.semantic_hits == 1


def test_explicit_cost_overrides_pricing(cache):
    cache.set("q", "a", model="test-model", input_tokens=10**6, cost=0.5)
    cache.get("q", model="test-model")
    assert cache.stats.cost_saved == pytest.approx(0.5)


def test_reset_and_snapshot_isolation(cache):
    cache.get("nothing here")
    snap = cache.stats
    cache.get("still nothing")
    assert snap.lookups == 1
    cache.reset_stats()
    assert cache.stats.lookups == 0


def test_round_trip_dict(cache):
    cache.set("q", "a", model="test-model", cost=1.0)
    cache.get("q", model="test-model")
    data = cache.stats.to_dict()
    assert data["hit_rate"] == 1.0
    restored = CacheStats.from_dict(data)
    assert restored.cost_saved == 1.0
    assert restored.by_model["test-model"].hits == 1


def test_summary_mentions_key_numbers(cache):
    cache.set("q", "a", cost=0.25)
    cache.get("q")
    text = cache.stats.summary()
    assert "100.0%" in text and "$0.2500" in text
