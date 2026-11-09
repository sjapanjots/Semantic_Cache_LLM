# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `semcache.tuning`: `calibrate()` recommends a similarity threshold from labelled
  prompt pairs (best F0.5, or the loosest threshold meeting `target_precision`), with
  a precision/recall table and the pairs it gets wrong. `load_pairs()` reads them from
  CSV. Example data in `examples/tuning_pairs.csv`.

## [0.1.0] - 2026-11-09

### Added
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
