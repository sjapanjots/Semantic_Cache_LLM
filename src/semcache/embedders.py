"""Embedders turn text into unit-length vectors so prompts can be compared by meaning.

Every embedder exposes:

* ``name``: a stable identifier. Vectors from different embedders are not comparable,
  so persistent stores record it and refuse to mix them.
* ``recommended_threshold`` (optional): a starting similarity cut-off for this model.
  Tune it on your own traffic.
* ``embed(texts)``: returns a ``float32`` array of shape ``(len(texts), dim)`` whose rows
  are L2-normalised, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "CallableEmbedder",
    "normalize_text",
    "exact_key",
    "l2_normalize",
]


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """Lower-case, strip accents, punctuation and repeated whitespace.

    Too aggressive for deciding that two prompts are *the same* ("2+2" and "2-2" both
    become "2 2"); use :func:`exact_key` for that.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


_TRAILING = re.compile(r"[\s?!.。？！]+$")


def exact_key(text: str) -> str:
    """Key for the exact-match fast path: two prompts with the same key are the same.

    Ignores only differences that never change the meaning: letter case, Unicode form,
    repeated whitespace and trailing ``?``, ``!`` or ``.``. Symbols and inner
    punctuation are kept, so "What is 2+2?" and "What is 2-2?" stay different.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    return _TRAILING.sub("", _WS.sub(" ", text).strip())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# A small stop-word list keeps filler words from dominating the hashed vector.
_STOP_WORDS = frozenset(
    """a an the is are was were be been am do does did of to in on at for by with and or
    but if then so than that this these those it its i me my we our you your he she they
    them their what which who whom how can could would should will shall may might must
    please tell give show explain s t d ll re ve m""".split()
)

# Words, plus operators that change meaning ("2+2" vs "2*2", "==" vs "=", "C++", "C#").
# A run of operator characters is one token, so "==" differs from "=".
_TOKEN = re.compile(r"\w+|!=|[+\-*/<>=%^&|~]+|[#@$]")
_INNER_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")  # "state-of-the-art"
_PROSE_DASH = re.compile(r"\s[-\u2013\u2014]\s|[\u2013\u2014]")  # "Python - how to ..."


def _token_weight(token: str) -> float:
    """Extra weight for short tokens that often decide the meaning."""
    if not token[0].isalnum() and token[0] != "_":
        return 6.0  # operator: "+" vs "*", "<" vs ">"
    if len(token) <= 2 or any(ch.isdigit() for ch in token):
        return 3.0  # one-letter names ("C", "R", "x") and numbers
    return 1.0


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = _PROSE_DASH.sub(" ", _INNER_HYPHEN.sub(" ", text))
    return [t for t in _TOKEN.findall(text) if t not in _STOP_WORDS]


class HashingEmbedder:
    """Dependency-free embedder built on hashed word and character n-grams.

    It is lexical, not truly semantic: "capital of France" and "France's capital city"
    match well, while "automobile" and "car" do not. It is a good default for tests,
    offline use and prompts that are mostly re-worded. For real paraphrase matching use
    :class:`SentenceTransformerEmbedder` or :class:`OpenAIEmbedder`.
    """

    def __init__(self, dim: int = 1024, char_ngram: int = 3, word_weight: float = 2.0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_ngram = char_ngram
        self.word_weight = word_weight
        # v2: one-letter words and symbols count ("C" vs "R", "x > 5" vs "x < 5").
        self.name = f"hashing-v2-{dim}-{char_ngram}"
        self.recommended_threshold = 0.7

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = _tokens(text)
        # Short tokens (one letter, a number, an operator) have few character n-grams,
        # yet often decide the meaning ("C" vs "R", "10" vs "20", "<" vs ">").
        feats: list[tuple[str, float]] = [
            ("w:" + w, self.word_weight * _token_weight(w)) for w in words
        ]
        feats += [
            ("b:" + a + " " + b, self.word_weight) for a, b in zip(words, words[1:], strict=False)
        ]
        n = self.char_ngram
        for w in words:
            padded = f"<{w}>"
            feats += [("c:" + padded[i : i + n], 1.0) for i in range(max(1, len(padded) - n + 1))]
        return feats

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                idx, sign = self._bucket(feature)
                out[row, idx] += sign * weight
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    """Local neural embeddings via ``sentence-transformers``.

    Install with ``pip install semcacher[local]``.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: Any):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "SentenceTransformerEmbedder needs sentence-transformers: "
                "pip install 'semcacher[local]'"
            ) from exc
        self._model = SentenceTransformer(model, **kwargs)
        self.name = f"st:{model}"
        self.recommended_threshold = 0.85

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return l2_normalize(vectors)


class OpenAIEmbedder:
    """Embeddings from the OpenAI API (``pip install semcacher[openai]``).

    Pass an existing ``openai.OpenAI`` client, or one is created from the environment.
    """

    def __init__(self, model: str = "text-embedding-3-small", client: Any = None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "OpenAIEmbedder needs the openai package: pip install 'semcacher[openai]'"
                ) from exc
            client = OpenAI()
        self._client = client
        self.model = model
        self.name = f"openai:{model}"
        self.recommended_threshold = 0.9

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return l2_normalize(np.array([d.embedding for d in response.data], dtype=np.float32))


class CallableEmbedder:
    """Wrap any ``fn(list[str]) -> array-like`` (e.g. a Voyage, Cohere or in-house model)."""

    def __init__(
        self, fn: Callable[[list[str]], Any], name: str, recommended_threshold: float = 0.85
    ):
        self._fn = fn
        self.name = name
        self.recommended_threshold = recommended_threshold

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalize(np.asarray(self._fn(list(texts)), dtype=np.float32))
