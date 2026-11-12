import pytest

from semcache import DEFAULT_PRICES, ModelPrice, PriceBook


def test_model_price_cost():
    assert ModelPrice(2.0, 10.0).cost(1_000_000, 100_000) == pytest.approx(3.0)


def test_defaults_are_loaded():
    book = PriceBook()
    assert book.get("claude-opus-5-5") == DEFAULT_PRICES["claude-opus-5-5"]
    assert "claude-sonnet-5-5" in book


def test_register_and_override():
    book = PriceBook({"my-model": (1.0, 2.0)})
    assert book.cost("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    book.register("my-model", 0.5, 0.5)
    assert book.cost("my-model", 1_000_000, 1_000_000) == pytest.approx(1.0)


def test_snapshot_fallback_uses_longest_match():
    book = PriceBook({"gpt-x": (1, 1), "gpt-x-mini": (0.1, 0.1)}, include_defaults=False)
    assert book.get("gpt-x-mini-2026-01-01") == ModelPrice(0.1, 0.1)
    assert book.get("gpt-x-2026-01-01") == ModelPrice(1, 1)
    assert book.get("gpt-x-20260101") == ModelPrice(1, 1)
    assert book.get("gpt-x@20260101") == ModelPrice(1, 1)
    assert book.get("gpt-x:1") == ModelPrice(1, 1)
    assert book.get("gpt-x-latest") == ModelPrice(1, 1)


def test_other_suffixes_are_not_priced_as_the_base_model():
    book = PriceBook({"gpt-x": (2.5, 10)}, include_defaults=False)
    assert book.get("gpt-x-mini") is None  # a different, cheaper model
    assert book.get("gpt-x-turbo") is None
    assert book.get("gpt-xl") is None
    assert book.cost("gpt-x-mini", 10**6, 10**6) == 0.0
    assert PriceBook({"gpt-x": (1, 1)}, default=(9, 9)).get("gpt-x-mini") == ModelPrice(9, 9)


def test_unknown_model_costs_nothing_unless_default():
    assert PriceBook().cost("who-knows", 1000, 1000) == 0.0
    assert PriceBook(default=(1, 1)).cost("who-knows", 1_000_000, 0) == pytest.approx(1.0)
    assert PriceBook().cost(None, 1000, 1000) == 0.0
