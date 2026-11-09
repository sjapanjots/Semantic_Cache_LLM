<p align="center">
  <img src="assets/logo.svg" alt="semcache" width="480">
</p>

<p align="center">
  <b>A semantic cache for LLM calls.</b><br>
  If a new question means the same as an old one, return the saved answer.<br>
  Track your hit rate and how much money it saved.
</p>

<p align="center">
  <a href="https://github.com/sjapanjots/Sematic_Cache_LLM/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sjapanjots/Sematic_Cache_LLM/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why

Users ask the same things in different words. An exact-match cache misses
*"How do I reverse a list in Python?"* vs *"python: how to reverse a list"*, so you pay
for the same answer twice. `semcache` embeds each prompt, finds the closest one already
answered, and returns that answer when the two are similar enough. Every hit is counted
in tokens and dollars, so you can see what the cache is worth.

- **Drop-in**: wrap an `OpenAI` or `Anthropic` client, decorate a function, or call
  `get` / `set` yourself.
- **Shows savings**: hit rate, exact vs semantic hits, tokens saved, money saved and
  spent, broken down per model.
- **Safe defaults**: answers never cross models, namespaces, system prompts or
  conversation history. Truncated or refused responses are not cached.
- **Small**: the core needs only `numpy`. Neural embedders are optional extras.
- **Persistent**: one SQLite file holds entries and stats. TTL and LRU size limits
  are built in.

## Install

```bash
pip install semcacher                       # core (numpy only)
pip install "semcacher[local]"              # + sentence-transformers, runs locally
pip install "semcacher[openai]"             # + OpenAI embeddings
```

You install `semcacher` and import `semcache`:

```python
from semcache import SemanticCache
```

From source:

```bash
pip install git+https://github.com/sjapanjots/Sematic_Cache_LLM
```

## Quickstart

```python
from semcache import SemanticCache, LLMResult

cache = SemanticCache(pricing={"demo-model": (2.50, 10.00)})  # USD per 1M input/output tokens

def call_llm(prompt):
    ...  # your real LLM call
    return LLMResult("answer", input_tokens=1200, output_tokens=400)

cache.get_or_call("How do I reverse a list in Python?", lambda: call_llm(...), model="demo-model")
cache.get_or_call("python: how to reverse a list",      lambda: call_llm(...), model="demo-model")  # cache hit

print(cache.stats)
```

Output of [`examples/quickstart.py`](examples/quickstart.py), which needs no API key:

```text
lookups                   6
hits                      3  (50.0%)
  exact                   1
  semantic                2
misses                    3
tokens saved          4,800
money saved         $0.0210
money spent         $0.0210
savings rate          50.0%
```

## Usage

### 1. Wrap your client (no other code changes)

```python
from anthropic import Anthropic
from semcache import SemanticCache, SentenceTransformerEmbedder
from semcache.integrations import wrap_anthropic

cache = SemanticCache(embedder=SentenceTransformerEmbedder(), path="cache.db")
client = wrap_anthropic(Anthropic(), cache)

client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "How should a Python dev learn Rust?"}],
)
```

`wrap_openai(OpenAI(), cache)` does the same for `client.chat.completions.create`.
Async clients (`AsyncAnthropic`, `AsyncOpenAI`) work too. On a hit you get back the
SDK's normal response object, rebuilt from the cache.

How requests are matched:

| Part of the request | Treatment |
|---|---|
| Last user message | Compared **by meaning** |
| Model | Must match exactly |
| System prompt, earlier turns, tools, temperature, max tokens, ... | Hashed; must match exactly |
| `timeout`, `extra_headers`, `user`, `metadata` | Ignored |
| `stream=True` | Passed through, not cached |
| Responses stopped by `max_tokens` / `refusal` / `length` / `content_filter` | Not cached |

### 2. Decorate a function

```python
@cache.cached(model="claude-sonnet-5-5")
def ask(prompt: str) -> LLMResult:
    response = my_llm(prompt)
    return LLMResult(response.text, input_tokens=response.in_tok, output_tokens=response.out_tok)

ask("How many days are in a leap year?")
ask("how many days does a leap year have")   # served from the cache
```

Works on `async def` functions too. Use `key=lambda *a, **kw: ...` to choose which
argument is the prompt.

### 3. Call `get` / `set` yourself

```python
answer = cache.get(prompt, model="claude-opus-5-5")
if answer is None:
    answer, usage = call_llm(prompt)
    cache.set(prompt, answer, model="claude-opus-5-5",
              input_tokens=usage.input, output_tokens=usage.output)
```

`cache.lookup(...)` returns a `CacheHit` with `.response`, `.similarity`, `.exact`,
`.cost_saved` and the stored `.entry`.

## Choosing an embedder and threshold

The threshold is the minimum cosine similarity for a semantic hit. Each embedder ships
a conservative default; tune it on your own traffic.

| Embedder | Install | Default threshold | Notes |
|---|---|---|---|
| `HashingEmbedder` (default) | core | 0.70 | Lexical (words and character n-grams). No model download. Catches rewording, not synonyms. |
| `SentenceTransformerEmbedder()` | `[local]` | 0.85 | Real semantic matching, runs on CPU. `all-MiniLM-L6-v2` by default. |
| `OpenAIEmbedder()` | `[openai]` | 0.90 | `text-embedding-3-small` by default. Costs a small amount per lookup. |
| `CallableEmbedder(fn, name)` | core | 0.85 | Bring any model: Voyage, Cohere, an in-house service... |

Too many wrong answers? Raise the threshold. Too few hits? Lower it, or switch to a
neural embedder. You can also override it per call: `cache.lookup(p, threshold=0.95)`.

### Calibrating the threshold from your own data

Instead of guessing, label some prompt pairs as same or different meaning and let
`calibrate` pick the threshold:

```python
from semcache import HashingEmbedder, SemanticCache, calibrate, load_pairs

pairs = load_pairs("examples/tuning_pairs.csv")   # CSV columns: a,b,same
result = calibrate(HashingEmbedder(), pairs)        # or target_precision=0.99
print(result.summary())
cache = SemanticCache(threshold=result.threshold)
```

```text
embedder   hashing-1024-3
pairs      20 (10 same, 10 different)
recommend  threshold=0.7099  (best F0.5)

threshold  precision   recall  wrong hits    F0.5
   0.7000      83.3%    50.0%           1   0.735
   0.7083      83.3%    50.0%           1   0.735
   0.7099     100.0%    50.0%           0   0.833  <-
   0.7100     100.0%    40.0%           0   0.769
```

*Precision* is the share of hits that are correct; *recall* is the share of
same-meaning pairs that hit. By default the recommendation maximises F0.5, which
counts a wrong answer as worse than a miss. Pass `target_precision=0.99` to get the
loosest threshold that is still right 99% of the time. `result.mistakes()` lists the
pairs it gets wrong. Use the same embedder you'll use in the cache, and aim for at
least a few dozen pairs drawn from real traffic.

## Tracking savings

```python
s = cache.stats
s.hit_rate          # 0.42
s.cost_saved        # 18.73  (USD)
s.cost_spent        # 25.90
s.savings_rate      # 0.42   share of the would-be bill avoided
s.tokens_saved      # 3_912_000
s.by_model["claude-opus-5-5"].hit_rate
s.to_dict()         # for your metrics pipeline
```

Money saved is the price of the original call, credited every time its answer is
reused. Prices come from a `PriceBook`:

```python
from semcache import PriceBook

cache = SemanticCache(pricing={"gpt-4o-mini": (0.15, 0.60)})        # merged over defaults
cache = SemanticCache(pricing=PriceBook(default=(1.0, 3.0)))        # fallback for unknown models
cache.set(prompt, answer, cost=0.0123)                              # or give the exact cost
```

Claude list prices are bundled (run `semcache prices`). Prices change, so check your
provider's pricing page. Models without a price still count toward hit rate but add
$0 to savings.

## Persistence, expiry and size limits

```python
cache = SemanticCache(
    path="cache.db",        # SQLite file; entries and stats survive restarts
    ttl=7 * 24 * 3600,      # entries expire after a week
    max_entries=50_000,     # least recently used entries are evicted
    namespace="support-bot",
)
```

- Responses stored in SQLite must be JSON-serialisable. The client wrappers handle
  this for you.
- A cache file remembers which embedder built it and refuses to open with a
  different one, because their vectors aren't comparable.
- Use `with SemanticCache(...) as cache:` or call `cache.close()` so the latest stats
  are written.

## CLI

```bash
semcache stats --db cache.db          # hit rate, tokens and money saved
semcache stats --db cache.db --json
semcache list  --db cache.db          # most-hit entries
semcache prune --db cache.db --older-than 86400
semcache clear --db cache.db [--namespace support-bot] [--stats]
semcache prices                       # bundled price table
```

## How it works

```text
prompt ──► normalise ──► exact match in scope? ──yes──► HIT (similarity 1.0)
                              │ no
                              ▼
                           embed ──► nearest neighbour in scope ──► ≥ threshold? ──yes──► HIT
                                                                        │ no
                                                                        ▼
                                                                      MISS ──► call LLM ──► set()
```

A *scope* is `namespace + model`; the client wrappers add a hash of the rest of the
request. Search is an exact cosine scan over a NumPy matrix per scope, fast enough for
tens of thousands of entries per scope.

## When not to use it

- Answers that depend on time or live data ("what's the weather now"). Use a `ttl`,
  or don't cache those calls.
- Prompts where small wording changes matter, such as code with specific values or
  math with different numbers. Use a high threshold or a separate namespace.
- Personalised answers. Put a user or tenant id in `namespace` so answers never
  leak between users.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Roadmap

- Pluggable vector indexes (FAISS, hnswlib) for millions of entries
- Redis and Postgres/pgvector stores
- Streaming support in the client wrappers
- Optional LLM-judged verification of borderline hits

## License

[MIT](LICENSE)
