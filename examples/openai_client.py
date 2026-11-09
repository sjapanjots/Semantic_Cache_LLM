"""Cache OpenAI chat completions with a drop-in wrapper.

pip install "semcacher[openai]"
export OPENAI_API_KEY=...
python examples/openai_client.py
"""

from openai import OpenAI

from semcache import OpenAIEmbedder, SemanticCache
from semcache.integrations import wrap_openai

raw = OpenAI()
cache = SemanticCache(
    embedder=OpenAIEmbedder(client=raw),
    path="openai-cache.db",
    # OpenAI prices are not bundled; check https://openai.com/api/pricing and set yours.
    pricing={"gpt-4o-mini": (0.15, 0.60)},
)
client = wrap_openai(raw, cache)

for question in ["Explain recursion in one paragraph.", "In one paragraph, what is recursion?"]:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    print(completion.choices[0].message.content[:200], "...\n")

print(cache.stats)
cache.close()
