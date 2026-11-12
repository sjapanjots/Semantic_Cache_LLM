"""Pick a similarity threshold from labelled examples instead of guessing.

Give :func:`calibrate` pairs of prompts labelled "same meaning" or "different meaning".
It scores every pair the way :class:`~semcache.SemanticCache` would, then reports, for
each candidate threshold, how many cache hits you would get and how many of them would
serve the wrong answer.

::

    from semcache import HashingEmbedder
    from semcache.tuning import calibrate

    result = calibrate(
        HashingEmbedder(),
        [
            ("How do I reverse a list in Python?", "python: how to reverse a list", True),
            ("How do I reverse a list in Python?", "How do I sort a list in Python?", False),
            ...
        ],
        target_precision=0.99,
    )
    print(result.summary())
    cache = SemanticCache(threshold=result.threshold)

Precision is the share of hits that are correct; recall is the share of same-meaning
pairs that hit. A wrong cached answer is usually worse than a miss, so by default the
recommendation maximises F0.5, which weights precision twice as heavily as recall.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embedders import Embedder, exact_key

__all__ = ["LabeledPair", "ThresholdPoint", "CalibrationResult", "calibrate", "load_pairs"]


@dataclass(frozen=True)
class LabeledPair:
    """Two prompts and whether they should share a cached answer."""

    a: str
    b: str
    same: bool


@dataclass(frozen=True)
class ThresholdPoint:
    """Outcome of using one threshold on the labelled pairs."""

    threshold: float
    true_hits: int  # same meaning, served from cache (good)
    false_hits: int  # different meaning, served from cache (wrong answer)
    missed: int  # same meaning, not served (costs a call, no harm)
    true_misses: int  # different meaning, not served (good)

    @property
    def precision(self) -> float:
        hits = self.true_hits + self.false_hits
        return self.true_hits / hits if hits else 1.0

    @property
    def recall(self) -> float:
        same = self.true_hits + self.missed
        return self.true_hits / same if same else 0.0

    @property
    def false_hit_rate(self) -> float:
        """Share of different-meaning pairs that would wrongly hit."""
        different = self.false_hits + self.true_misses
        return self.false_hits / different if different else 0.0

    def f_beta(self, beta: float = 0.5) -> float:
        p, r = self.precision, self.recall
        if p == 0 and r == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)


@dataclass
class CalibrationResult:
    """Scores for every pair, the threshold curve and the recommended threshold."""

    pairs: list[LabeledPair]
    scores: list[float]
    points: list[ThresholdPoint]
    recommended: ThresholdPoint
    reason: str
    embedder_name: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def threshold(self) -> float:
        return self.recommended.threshold

    def at(self, threshold: float) -> ThresholdPoint:
        """The curve point closest to ``threshold``."""
        return min(self.points, key=lambda p: abs(p.threshold - threshold))

    def mistakes(self, threshold: float | None = None) -> list[tuple[LabeledPair, float]]:
        """Pairs the cache would get wrong at ``threshold`` (default: the recommendation)."""
        t = self.threshold if threshold is None else threshold
        return [
            (pair, score)
            for pair, score in zip(self.pairs, self.scores, strict=True)
            if (score >= t) != pair.same
        ]

    def summary(self, rows: int = 10) -> str:
        """A text table of the curve around the recommendation."""
        idx = self.points.index(self.recommended)
        lo = max(0, idx - rows // 2)
        shown = self.points[lo : lo + rows]
        same = sum(p.same for p in self.pairs)
        lines = [
            f"embedder   {self.embedder_name or '?'}",
            f"pairs      {len(self.pairs)} ({same} same, {len(self.pairs) - same} different)",
            f"recommend  threshold={self.threshold:.4f}  ({self.reason})",
            "",
            f"{'threshold':>9}  {'precision':>9}  {'recall':>7}  {'wrong hits':>10}  {'F0.5':>6}",
        ]
        for p in shown:
            mark = "  <-" if p is self.recommended else ""
            lines.append(
                f"{p.threshold:>9.4f}  {p.precision:>9.1%}  {p.recall:>7.1%}  "
                f"{p.false_hits:>10}  {p.f_beta():>6.3f}{mark}"
            )
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _as_pair(item: LabeledPair | Sequence) -> LabeledPair:
    if isinstance(item, LabeledPair):
        return item
    a, b, same = item
    return LabeledPair(str(a), str(b), _truthy(same))


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "same", "t"}:
            return True
        if v in {"0", "false", "no", "n", "different", "f", ""}:
            return False
        raise ValueError(f"cannot read {value!r} as a same/different label")
    return bool(value)


def _score_pairs(embedder: Embedder, pairs: list[LabeledPair]) -> list[float]:
    """Cosine similarity per pair, with 1.0 where the cache's exact-match path would hit."""
    texts = sorted({t for p in pairs for t in (p.a, p.b)})
    index = {t: i for i, t in enumerate(texts)}
    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    scores = []
    for p in pairs:
        if exact_key(p.a) and exact_key(p.a) == exact_key(p.b):
            scores.append(1.0)
        else:
            scores.append(float(vectors[index[p.a]] @ vectors[index[p.b]]))
    return scores


def _point(threshold: float, pairs: list[LabeledPair], scores: list[float]) -> ThresholdPoint:
    th = fh = mi = tm = 0
    for pair, score in zip(pairs, scores, strict=True):
        hit = score >= threshold
        if pair.same:
            th += hit
            mi += not hit
        else:
            fh += hit
            tm += not hit
    return ThresholdPoint(threshold, th, fh, mi, tm)


def calibrate(
    embedder: Embedder,
    pairs: Iterable[LabeledPair | Sequence],
    *,
    target_precision: float | None = None,
    beta: float = 0.5,
    thresholds: Iterable[float] | None = None,
) -> CalibrationResult:
    """Recommend a threshold for ``embedder`` from labelled prompt pairs.

    Args:
        embedder: the embedder you will use in the cache.
        pairs: ``(prompt_a, prompt_b, same)`` tuples or :class:`LabeledPair` objects.
        target_precision: if set, pick the *lowest* threshold whose precision reaches
            it, which gives the most hits while keeping wrong answers that rare. Falls
            back to the F-beta rule if no threshold reaches it.
        beta: weight of recall vs precision for the default rule (0.5 favours precision).
        thresholds: candidate thresholds. Defaults to 0.50, 0.51, ... 0.99 plus every
            distinct pair score in that range, so the best cut-off is never skipped.
    """
    items = [_as_pair(p) for p in pairs]
    if not items:
        raise ValueError("need at least one labelled pair")
    if target_precision is not None and not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")

    scores = _score_pairs(embedder, items)
    if thresholds is None:
        grid = {round(float(x), 2) for x in np.arange(0.50, 1.0, 0.01)}
        grid |= {s for s in scores if 0.5 <= s <= 1.0}
        candidates = sorted(grid)
    else:
        candidates = sorted({float(t) for t in thresholds})
    if not candidates or not all(0.0 < t <= 1.0 for t in candidates):
        raise ValueError("thresholds must be in (0, 1]")

    points = [_point(t, items, scores) for t in candidates]

    warnings = []
    n_same = sum(p.same for p in items)
    if n_same == 0 or n_same == len(items):
        warnings.append("all pairs have the same label; include both same and different pairs")
    if len(items) < 30:
        warnings.append(f"only {len(items)} pairs; the recommendation is rough")

    recommended: ThresholdPoint | None = None
    reason = ""
    if target_precision is not None:
        ok = [p for p in points if p.precision >= target_precision and p.true_hits > 0]
        if ok:
            recommended = min(ok, key=lambda p: p.threshold)
            reason = f"lowest threshold with precision >= {target_precision:.1%}"
        else:
            warnings.append(f"no threshold reaches precision {target_precision:.1%}")
    if recommended is None:
        # Highest F-beta; ties go to the stricter threshold.
        recommended = max(points, key=lambda p: (p.f_beta(beta), p.threshold))
        reason = f"best F{beta:g}"

    return CalibrationResult(
        pairs=items,
        scores=scores,
        points=points,
        recommended=recommended,
        reason=reason,
        embedder_name=getattr(embedder, "name", ""),
        warnings=warnings,
    )


def load_pairs(path: str | Path) -> list[LabeledPair]:
    """Read labelled pairs from a CSV file with columns ``a,b,same``.

    ``same`` accepts 1/0, true/false, yes/no or same/different. A header row is
    optional.
    """
    pairs = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != 3:
                raise ValueError(f"expected 3 columns (a,b,same), got {len(row)}: {row}")
            if not pairs and [c.strip().lower() for c in row] == ["a", "b", "same"]:
                continue
            pairs.append(LabeledPair(row[0], row[1], _truthy(row[2])))
    return pairs
