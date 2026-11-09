"""semcache: a semantic cache for LLM calls."""

from .cache import CacheHit, SemanticCache
from .embedders import (
    CallableEmbedder,
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from .pricing import DEFAULT_PRICES, ModelPrice, PriceBook
from .sqlite_store import EmbedderMismatchError, SQLiteStore
from .stats import CacheStats, ModelStats
from .store import BaseStore, CacheEntry, InMemoryStore

__version__ = "0.1.0"

__all__ = [
    "SemanticCache",
    "CacheHit",
    "CacheEntry",
    "CacheStats",
    "ModelStats",
    "PriceBook",
    "ModelPrice",
    "DEFAULT_PRICES",
    "BaseStore",
    "InMemoryStore",
    "SQLiteStore",
    "EmbedderMismatchError",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "CallableEmbedder",
    "__version__",
]
