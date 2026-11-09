"""Wrap any function, whichever provider or SDK it uses, with @cache.cached."""

from semcache import LLMResult, SemanticCache

cache = SemanticCache(path="decorator-cache.db", pricing={"my-model": (1.0, 4.0)})


@cache.cached(model="my-model")
def ask(prompt: str) -> LLMResult:
    # Replace this with a real call. Returning token usage lets semcache track savings.
    text = f"pretend answer to {prompt!r}"
    return LLMResult(text, input_tokens=800, output_tokens=300)


print(ask("How many days are in a leap year?"))
print(ask("how many days does a leap year have"))  # served from the cache
print(cache.stats)
cache.close()
