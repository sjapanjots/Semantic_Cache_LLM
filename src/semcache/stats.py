"""Hit-rate and savings accounting."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["CacheStats", "ModelStats", "StatsTracker"]


@dataclass
class ModelStats:
    hits: int = 0
    misses: int = 0
    input_tokens_saved: int = 0
    output_tokens_saved: int = 0
    cost_saved: float = 0.0
    cost_spent: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class CacheStats:
    """A snapshot of what the cache has done since it started (or was reset).

    ``cost_spent`` is what the calls you stored with :meth:`SemanticCache.set` cost;
    ``cost_saved`` is what the calls served from the cache would have cost.
    """

    lookups: int = 0
    hits: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    writes: int = 0
    input_tokens_saved: int = 0
    output_tokens_saved: int = 0
    cost_saved: float = 0.0
    cost_spent: float = 0.0
    started_at: float = field(default_factory=time.time)
    by_model: dict[str, ModelStats] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def tokens_saved(self) -> int:
        return self.input_tokens_saved + self.output_tokens_saved

    @property
    def savings_rate(self) -> float:
        """Share of the would-be bill that the cache avoided."""
        total = self.cost_saved + self.cost_spent
        return self.cost_saved / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            hit_rate=self.hit_rate,
            tokens_saved=self.tokens_saved,
            savings_rate=self.savings_rate,
        )
        for name, model in self.by_model.items():
            data["by_model"][name]["hit_rate"] = model.hit_rate
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheStats:
        known = {f for f in cls.__dataclass_fields__ if f != "by_model"}
        stats = cls(**{k: v for k, v in data.items() if k in known})
        model_fields = set(ModelStats.__dataclass_fields__)
        stats.by_model = {
            name: ModelStats(**{k: v for k, v in m.items() if k in model_fields})
            for name, m in (data.get("by_model") or {}).items()
        }
        return stats

    def summary(self) -> str:
        lines = [
            f"lookups        {self.lookups:>12,}",
            f"hits           {self.hits:>12,}  ({self.hit_rate:.1%})",
            f"  exact        {self.exact_hits:>12,}",
            f"  semantic     {self.semantic_hits:>12,}",
            f"misses         {self.misses:>12,}",
            f"tokens saved   {self.tokens_saved:>12,}",
            f"money saved    {'$' + format(self.cost_saved, ',.4f'):>12}",
            f"money spent    {'$' + format(self.cost_spent, ',.4f'):>12}",
            f"savings rate   {self.savings_rate:>12.1%}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class StatsTracker:
    """Thread-safe accumulator behind :attr:`SemanticCache.stats`."""

    def __init__(self, initial: CacheStats | None = None) -> None:
        self._stats = initial or CacheStats()
        self._lock = threading.Lock()

    def _model(self, model: str | None) -> ModelStats:
        return self._stats.by_model.setdefault(model or "unknown", ModelStats())

    def record_hit(
        self,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        exact: bool,
    ) -> None:
        with self._lock:
            s = self._stats
            s.lookups += 1
            s.hits += 1
            if exact:
                s.exact_hits += 1
            else:
                s.semantic_hits += 1
            s.input_tokens_saved += input_tokens
            s.output_tokens_saved += output_tokens
            s.cost_saved += cost
            m = self._model(model)
            m.hits += 1
            m.input_tokens_saved += input_tokens
            m.output_tokens_saved += output_tokens
            m.cost_saved += cost

    def record_miss(self, model: str | None) -> None:
        with self._lock:
            self._stats.lookups += 1
            self._stats.misses += 1
            self._model(model).misses += 1

    def record_write(self, model: str | None, cost: float) -> None:
        with self._lock:
            self._stats.writes += 1
            self._stats.cost_spent += cost
            self._model(model).cost_spent += cost

    def snapshot(self) -> CacheStats:
        with self._lock:
            return CacheStats.from_dict(asdict(self._stats))

    def reset(self) -> None:
        with self._lock:
            self._stats = CacheStats()
