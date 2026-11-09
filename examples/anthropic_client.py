"""Cache Claude calls with a drop-in wrapper.

pip install "semcache-llm[local]" anthropic
export ANTHROPIC_API_KEY=...
python examples/anthropic_client.py
"""

from anthropic import Anthropic

from semcache import SemanticCache, SentenceTransformerEmbedder
from semcache.integrations import wrap_anthropic

cache = SemanticCache(embedder=SentenceTransformerEmbedder(), path="claude-cache.db")
client = wrap_anthropic(Anthropic(), cache)

for question in [
    "What's a good way to learn Rust as a Python developer?",
    "As a Python dev, how should I go about learning Rust?",
]:
    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    print(message.content[0].text[:200], "...\n")

print(cache.stats)
cache.close()
