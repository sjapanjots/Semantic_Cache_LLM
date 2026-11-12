# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Streaming in the client wrappers (`semcache.streaming`): `stream=True` requests to
  `wrap_anthropic` and `wrap_openai`, sync and async, are now cached. A miss passes the
  provider's stream through unchanged and saves the response once it completes; a hit
  replays it as the provider's own stream events. Streamed and non-streamed requests
  share entries. Supports text, tool use, thinking and citations (Anthropic) and
  content, refusals and tool calls (OpenAI).

## [0.2.0] - 2026-11-12

### Fixed
- **Wrong answers from the exact-match path:** prompts that differed only in symbols
  were treated as the same question ("What is 2+2?" and "What is 2-2?", "x > 5" and
  "x < 5"), so `set()` overwrote one with the other and `get()` returned the wrong
  answer. Exact matching now ignores only case, spacing, Unicode form and trailing
  `?`/`!`/`.` (new `exact_key()`).
- **Wrong answers from the built-in matcher:** `HashingEmbedder` dropped one-letter
  words and operators and gave numbers little weight, so "read a file in C" and
  "in R" scored 1.0. It now keeps them and weighs them up; on the bundled sample pairs the default
  threshold gives no wrong hits and finds more rewordings.
- **Wrong answers from the client wrappers:** images and other non-text parts of the
  last message were ignored, so two different images with the same question shared
  an answer. They are now part of the match.
- Price fallback priced other models by prefix (`gpt-x-mini` at `gpt-x` prices). Only
  snapshot ids (`-2024-08-06`, `-20240806`, `@...`, `:...`, `-latest`) fall back now.
- The async API (`aget_or_call`, async `@cached`, async OpenAI/Anthropic clients)
  blocked the event loop while embedding or verifying; that work now runs in a thread.
  New `alookup()` and `aset()`.
- `cache.clear(namespace)` and `semcache clear --namespace` missed entries created by
  the client wrappers.
- A full cache (`max_entries`) scanned the whole store on every write; it now evicts
  the least recently used tenth at once.

### Changed
- **Upgrade note:** the built-in matcher is now named `hashing-v2-...`. A SQLite or
  Redis cache built with 0.1.0's default matcher is refused with
  `EmbedderMismatchError` instead of mixing incompatible vectors; clear it once with
  `semcache clear --db <file>` (or `cache.clear()`).

### Added
- `semcache.verifier`: `LLMVerifier` asks a cheap LLM whether a semantic match's saved
  answer really answers the new question, and turns the hit into a miss if not. Pass it
  as `SemanticCache(verifier=...)`. Includes `anthropic_judge` (Claude Haiku 5.5 at low
  effort by default) and `openai_judge`, `verify_below` to skip close matches,
  remembered decisions, a safe `on_error` default, and `verifier.stats`.
- `semcache.metrics`: export cache stats to monitoring. `PrometheusCollector` for
  `prometheus_client` (`[prometheus]` extra), `prometheus_text()` for any web framework
  with no extra packages, and `register_opentelemetry()` (`[otel]` extra). Covers
  lookups, exact/semantic hits, misses, tokens and money saved and spent, hit ratio,
  entries, and per-model breakdowns.

## [0.1.0] - 2026-11-09

### Added
- First release on PyPI: `pip install semcacher` (import name `semcache`).
- Supports Python 3.10 and later.
- `SemanticCache` with exact-match fast path and cosine-similarity lookup.
- Embedders: dependency-free `HashingEmbedder`, `SentenceTransformerEmbedder`,
  `OpenAIEmbedder`, and `CallableEmbedder` for anything else.
- `InMemoryStore` and persistent `SQLiteStore`.
- `RedisStore` (`pip install "semcacher[redis]"`): keep the cache in Redis so several
  processes or servers share answers and stats. Works with any Redis-compatible
  server; stats from all processes add up instead of overwriting each other.
- Namespace and model isolation, TTL expiry, LRU `max_entries` limit.
- Stats: hit rate, exact vs semantic hits, tokens saved, money saved and spent,
  per-model breakdown; persisted with `SQLiteStore`.
- `PriceBook` with bundled Claude list prices and custom prices.
- `get_or_call`, `aget_or_call` and the `@cache.cached` decorator (sync and async).
- `wrap_openai` and `wrap_anthropic` drop-in client wrappers.
- `semcache` CLI: `stats`, `list`, `clear`, `prune`, `prices`.
- `semcache.tuning`: `calibrate()` recommends a similarity threshold from labelled
  prompt pairs, and `load_pairs()` reads them from CSV.
