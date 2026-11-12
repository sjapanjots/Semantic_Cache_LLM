from pathlib import Path

import pytest

from semcache import CallableEmbedder, HashingEmbedder, SemanticCache
from semcache.tuning import LabeledPair, ThresholdPoint, calibrate, load_pairs

EXAMPLE_CSV = Path(__file__).resolve().parent.parent / "examples" / "tuning_pairs.csv"


def toy_embedder():
    """Each text maps to a fixed 2-d direction, so similarities are known exactly."""
    import math

    angles = {"q": 0, "same-close": 10, "same-far": 40, "diff-close": 25, "diff-far": 80}

    def embed(texts):
        return [
            [math.cos(math.radians(angles[t])), math.sin(math.radians(angles[t]))] for t in texts
        ]

    return CallableEmbedder(embed, "toy")


TOY_PAIRS = [
    ("q", "same-close", True),  # cos 10 deg = 0.985
    ("q", "same-far", True),  # cos 40 deg = 0.766
    ("q", "diff-close", False),  # cos 25 deg = 0.906
    ("q", "diff-far", False),  # cos 80 deg = 0.174
]


def test_threshold_point_metrics():
    p = ThresholdPoint(0.8, true_hits=8, false_hits=2, missed=2, true_misses=8)
    assert p.precision == pytest.approx(0.8)
    assert p.recall == pytest.approx(0.8)
    assert p.false_hit_rate == pytest.approx(0.2)
    assert p.f_beta(1.0) == pytest.approx(0.8)
    empty = ThresholdPoint(0.99, 0, 0, 5, 5)
    assert empty.precision == 1.0 and empty.recall == 0.0 and empty.f_beta() == 0.0


def test_default_rule_prefers_no_wrong_answers():
    result = calibrate(toy_embedder(), TOY_PAIRS)
    # Above diff-close (0.906) only same-close hits: precision 1.0, recall 0.5, F0.5 0.833.
    # Below same-far (0.766) both same pairs hit, but so does diff-close: F0.5 0.714.
    assert 0.906 < result.threshold <= 0.985
    assert result.recommended.false_hits == 0
    assert result.reason == "best F0.5"


def test_target_precision_picks_lowest_qualifying_threshold():
    # Precision is 2/3 anywhere at or below same-far (0.766), so 0.6 is met at the floor.
    loose = calibrate(toy_embedder(), TOY_PAIRS, target_precision=0.6)
    assert loose.threshold == 0.5
    assert loose.recommended.true_hits == 2 and loose.recommended.false_hits == 1
    # 0.7 needs diff-close (0.906) excluded: the first candidate above it is 0.91.
    strict = calibrate(toy_embedder(), TOY_PAIRS, target_precision=0.7)
    assert strict.threshold == pytest.approx(0.91)
    assert type(strict.threshold) is float


def test_unreachable_target_falls_back_with_warning():
    pairs = [("q", "diff-close", False), ("q", "same-far", True)]  # different pair scores higher
    result = calibrate(toy_embedder(), pairs, target_precision=1.0, thresholds=[0.5, 0.8])
    assert any("no threshold reaches" in w for w in result.warnings)
    assert result.reason == "best F0.5"


def test_scores_and_mistakes():
    result = calibrate(toy_embedder(), TOY_PAIRS)
    assert result.scores[0] == pytest.approx(0.985, abs=1e-3)
    wrong = result.mistakes(0.7)
    assert [p.b for p, _ in wrong] == ["diff-close"]
    assert {p.b for p, _ in result.mistakes()} == {"same-far"}


def test_exact_matches_score_one():
    result = calibrate(HashingEmbedder(), [("Hello, World!", "hello world", True)])
    assert result.scores == [1.0]


def test_custom_thresholds_and_at():
    result = calibrate(toy_embedder(), TOY_PAIRS, thresholds=[0.95, 0.5, 0.8])
    assert [p.threshold for p in result.points] == [0.5, 0.8, 0.95]
    assert result.at(0.79).threshold == 0.8


def test_warnings_for_small_or_one_sided_data():
    result = calibrate(toy_embedder(), [("q", "same-close", True)])
    assert any("same label" in w for w in result.warnings)
    assert any("only 1 pairs" in w for w in result.warnings)


@pytest.mark.parametrize(
    "kwargs",
    [{"target_precision": 0}, {"target_precision": 1.5}, {"thresholds": [0]}, {"thresholds": []}],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        calibrate(toy_embedder(), TOY_PAIRS, **kwargs)


def test_no_pairs():
    with pytest.raises(ValueError):
        calibrate(toy_embedder(), [])


def test_accepts_labeled_pair_objects_and_string_labels():
    result = calibrate(
        toy_embedder(), [LabeledPair("q", "same-close", True), ("q", "diff-far", "different")]
    )
    assert [p.same for p in result.pairs] == [True, False]
    with pytest.raises(ValueError):
        calibrate(toy_embedder(), [("q", "same-close", "maybe")])


def test_load_pairs(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("a,b,same\nhi,hello,yes\n\nbye,hello,0\n", encoding="utf-8")
    assert load_pairs(path) == [
        LabeledPair("hi", "hello", True),
        LabeledPair("bye", "hello", False),
    ]
    path.write_text("only,two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="3 columns"):
        load_pairs(path)


def test_example_file_and_summary():
    pairs = load_pairs(EXAMPLE_CSV)
    result = calibrate(HashingEmbedder(), pairs)
    assert len(pairs) == 20
    assert result.recommended.false_hits == 0
    text = result.summary()
    assert "recommend" in text and "<-" in text and "hashing-v2-1024-3" in text


def test_recommended_threshold_matches_cache_behaviour():
    pairs = load_pairs(EXAMPLE_CSV)
    result = calibrate(HashingEmbedder(), pairs)
    for pair, score in zip(result.pairs, result.scores, strict=True):
        cache = SemanticCache(threshold=result.threshold)
        cache.set(pair.a, "answer")
        assert (cache.get(pair.b) is not None) == (score >= result.threshold)
