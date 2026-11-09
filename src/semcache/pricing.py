"""Token prices, used to turn cache hits into money saved."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

__all__ = ["ModelPrice", "PriceBook", "DEFAULT_PRICES"]


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


# Anthropic first-party list prices (USD per 1M tokens) as of October 2026.
# Prices change: check your provider's pricing page and override with
# PriceBook.register(). Models from other providers are not bundled, so register the
# ones you use, e.g. book.register("gpt-4o-mini", 0.15, 0.60).
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.00, 50.00),
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

PriceLike = Union[ModelPrice, tuple, list]


def _as_price(value: PriceLike) -> ModelPrice:
    if isinstance(value, ModelPrice):
        return value
    input_price, output_price = value
    return ModelPrice(float(input_price), float(output_price))


class PriceBook:
    """Maps model names to prices.

    Lookups try an exact match first and then the longest registered prefix, so a
    suffixed name such as ``"claude-opus-5-5-fast"`` falls back to ``"claude-opus-5-5"``.

    Args:
        prices: extra or overriding prices, ``{"model": ModelPrice | (input, output)}``.
        include_defaults: start from :data:`DEFAULT_PRICES`.
        default: price for models that are not registered. ``None`` means they cost 0,
            so hits on them count toward hit rate but not toward money saved.
    """

    def __init__(
        self,
        prices: Mapping[str, PriceLike] | None = None,
        *,
        include_defaults: bool = True,
        default: PriceLike | None = None,
    ) -> None:
        self._prices: dict[str, ModelPrice] = dict(DEFAULT_PRICES) if include_defaults else {}
        for model, price in (prices or {}).items():
            self._prices[model] = _as_price(price)
        self.default = _as_price(default) if default is not None else None
        self._lock = threading.Lock()

    def register(self, model: str, input_per_million: float, output_per_million: float) -> None:
        with self._lock:
            self._prices[model] = ModelPrice(input_per_million, output_per_million)

    def get(self, model: str | None) -> ModelPrice | None:
        if model:
            price = self._prices.get(model)
            if price is not None:
                return price
            matches = [name for name in self._prices if model.startswith(name)]
            if matches:
                return self._prices[max(matches, key=len)]
        return self.default

    def cost(self, model: str | None, input_tokens: int, output_tokens: int) -> float:
        price = self.get(model)
        return price.cost(input_tokens, output_tokens) if price else 0.0

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def models(self) -> list[str]:
        return sorted(self._prices)
