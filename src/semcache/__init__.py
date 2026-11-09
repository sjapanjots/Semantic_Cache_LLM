"""semcache: a semantic cache for LLM calls."""

from .cache import CacheHit, LLMResult, SemanticCache
from .embedders import (
    CallableEmbedder,
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from .pricing import DEFAULT_PRICES, ModelPrice, PriceBook
from .redis_store import RedisStore
from .sqlite_store import EmbedderMismatchError, SQLiteStore
from .stats import CacheStats, ModelStats
from .store import BaseStore, CacheEntry, InMemoryStore
from .tuning import CalibrationResult, LabeledPair, calibrate, load_pairs

__version__ = "0.2.0"

__all__ = [
    "SemanticCache",
    "CacheHit",
    "LLMResult",
    "CacheEntry",
    "CacheStats",
    "ModelStats",
    "PriceBook",
    "ModelPrice",
    "DEFAULT_PRICES",
    "BaseStore",
    "InMemoryStore",
    "SQLiteStore",
    "RedisStore",
    "EmbedderMismatchError",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "CallableEmbedder",
    "calibrate",
    "load_pairs",
    "LabeledPair",
    "CalibrationResult",
    "__version__",
]
