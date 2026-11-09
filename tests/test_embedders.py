import numpy as np
import pytest

from semcache import CallableEmbedder, HashingEmbedder
from semcache.embedders import l2_normalize, normalize_text


def test_normalize_text():
    assert normalize_text("  Héllo,   WORLD!! ") == "hello world"


def test_hashing_embedder_shape_and_unit_norm():
    e = HashingEmbedder(dim=256)
    v = e.embed(["hello world", "another prompt", ""])
    assert v.shape == (3, 256)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)
    assert np.allclose(v[2], 0.0)  # empty text stays a zero vector


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder().embed(["What is the capital of France?"])
    b = HashingEmbedder().embed(["What is the capital of France?"])
    assert np.array_equal(a, b)


def test_hashing_embedder_ranks_paraphrase_above_unrelated():
    e = HashingEmbedder()
    q, para, other = e.embed(
        [
            "How do I reverse a list in Python?",
            "python: how to reverse a list",
            "What is the best pizza topping?",
        ]
    )
    assert q @ para > 0.8
    assert q @ other < 0.2


def test_hashing_embedder_rejects_bad_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_callable_embedder_normalises():
    e = CallableEmbedder(lambda texts: [[3.0, 4.0] for _ in texts], name="toy")
    v = e.embed(["a", "b"])
    assert np.allclose(v, [[0.6, 0.8], [0.6, 0.8]])
    assert e.recommended_threshold == 0.85


def test_l2_normalize_handles_1d_and_zero():
    out = l2_normalize(np.array([0.0, 0.0]))
    assert out.shape == (1, 2) and np.allclose(out, 0.0)
