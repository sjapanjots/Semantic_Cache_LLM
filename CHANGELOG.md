# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-09

### Added
- First release on PyPI: `pip install semcacher` (import name `semcache`).
- Supports Python 3.10 and later.
- `SemanticCache` with exact-match fast path and cosine-similarity lookup.
- Embedders: dependency-free `HashingEmbedder`, `SentenceTransformerEmbedder`,
  `OpenAIEmbedder`, and `CallableEmbedder` for anything else.
- `InMemoryStore` and persistent `SQLiteStore`.
- Namespace and model isolation, TTL expiry, LRU `max_entries` limit.
- Stats: hit rate, exact vs semantic hits, tokens saved, money saved and spent,
  per-model breakdown; persisted with `SQLiteStore`.
- `PriceBook` with bundled Claude list prices and custom prices.
- `get_or_call`, `aget_or_call` and the `@cache.cached` decorator (sync and async).
- `wrap_openai` and `wrap_anthropic` drop-in client wrappers.
- `semcache` CLI: `stats`, `list`, `clear`, `prune`, `prices`.
- `semcache.tuning`: `calibrate()` recommends a similarity threshold from labelled
  prompt pairs, and `load_pairs()` reads them from CSV.
