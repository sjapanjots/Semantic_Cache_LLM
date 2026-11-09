"""Runs without an API key: a fake LLM stands in for the real one."""

from semcache import LLMResult, SemanticCache

cache = SemanticCache(pricing={"demo-model": (2.50, 10.00)})  # USD per 1M tokens


def fake_llm(prompt: str) -> LLMResult:
    print(f"  -> calling the LLM for: {prompt!r}")
    return LLMResult(f"(answer to: {prompt})", input_tokens=1_200, output_tokens=400)


questions = [
    "How do I reverse a list in Python?",
    "python: how to reverse a list",  # same meaning, different words
    "How do I reverse a list in Python",  # same text, different punctuation
    "What is the capital of France?",
    "what's the capital city of France",
    "How do I sort a list in Python?",  # similar words, different question
]

for q in questions:
    answer = cache.get_or_call(q, lambda q=q: fake_llm(q), model="demo-model")
    print(f"{q!r}\n  {answer}\n")

print(cache.stats)
