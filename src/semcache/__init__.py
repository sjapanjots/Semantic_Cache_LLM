"""semcache: a semantic cache for LLM calls."""

from .cache import CacheHit, SemanticCache
from .embedders import (
    CallableEmbedder,
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from .store import BaseStore, CacheEntry, InMemoryStore

__version__ = "0.1.0"

__all__ = [
    "SemanticCache",
    "CacheHit",
    "CacheEntry",
    "BaseStore",
    "InMemoryStore",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "CallableEmbedder",
    "__version__",
]
