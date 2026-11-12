import asyncio

import pytest

from semcache import LLMResult, SemanticCache


@pytest.fixture
def cache():
    return SemanticCache(pricing={"m": (1.0, 1.0)})


def test_get_or_call_only_calls_on_miss(cache):
    calls = []

    def llm():
        calls.append(1)
        return LLMResult("Paris", input_tokens=500_000, output_tokens=500_000)

    assert cache.get_or_call("What is the capital of France?", llm, model="m") == "Paris"
    assert cache.get_or_call("what is the capital of france", llm, model="m") == "Paris"
    assert len(calls) == 1
    assert cache.stats.cost_saved == pytest.approx(1.0)


def test_get_or_call_accepts_plain_values(cache):
    assert cache.get_or_call("q", lambda: "a") == "a"
    assert cache.get("q") == "a"


def test_aget_or_call(cache):
    async def llm():
        return LLMResult("async answer", cost=0.1)

    async def run():
        first = await cache.aget_or_call("async question", llm)
        second = await cache.aget_or_call("async question", llm)
        return first, second

    assert asyncio.run(run()) == ("async answer", "async answer")
    assert cache.stats.hits == 1


def test_cached_decorator_sync(cache):
    calls = []

    @cache.cached(model="m")
    def ask(prompt: str) -> str:
        calls.append(prompt)
        return prompt.upper()

    assert ask("How do I reverse a list in Python?") == "HOW DO I REVERSE A LIST IN PYTHON?"
    assert ask("python: how to reverse a list") == "HOW DO I REVERSE A LIST IN PYTHON?"
    assert ask(prompt="python: how to reverse a list").startswith("HOW")
    assert len(calls) == 1
    assert ask.__name__ == "ask"


def test_cached_decorator_bare_and_custom_key(cache):
    @cache.cached
    def bare(prompt):
        return "x"

    @cache.cached(key=lambda user, question: question)
    def ask(user, question):
        return f"{user}:{question}"

    assert bare("hello") == "x"
    assert ask("ann", "what time is it") == "ann:what time is it"
    assert ask("bob", "what time is it") == "ann:what time is it"


def test_cached_decorator_async(cache):
    calls = []

    @cache.cached
    async def ask(prompt: str) -> LLMResult:
        calls.append(prompt)
        return LLMResult("answer", cost=0.5)

    async def run():
        return [await ask("one question"), await ask("one question")]

    assert asyncio.run(run()) == ["answer", "answer"]
    assert len(calls) == 1
    assert cache.stats.cost_saved == pytest.approx(0.5)


def test_cached_decorator_without_prompt_raises(cache):
    @cache.cached
    def f(x: int):
        return x

    with pytest.raises(TypeError):
        f(3)


def test_embedding_reused_between_miss_and_set():
    from semcache import CallableEmbedder

    seen = []

    def embed(texts):
        seen.extend(texts)
        return [[1.0, float(len(t))] for t in texts]

    cache = SemanticCache(embedder=CallableEmbedder(embed, "toy"))
    cache.get_or_call("a brand new prompt", lambda: "x")
    assert seen == ["a brand new prompt"]


def test_async_calls_do_not_block_the_event_loop():
    import time

    from semcache import CallableEmbedder

    def slow_embed(texts):
        time.sleep(0.2)
        return [[1.0, float(len(t))] for t in texts]

    cache = SemanticCache(embedder=CallableEmbedder(slow_embed, "slow"))

    async def llm():
        return "x"

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(30):
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        await cache.aget_or_call("some question", llm)
        seen = ticks
        await task
        return seen

    assert asyncio.run(run()) > 0


def test_alookup_and_aset(cache):
    async def run():
        await cache.aset("async question", "answer", model="m", cost=0.1)
        hit = await cache.alookup("async question", model="m")
        miss = await cache.alookup("something else entirely", model="m")
        return hit, miss

    hit, miss = asyncio.run(run())
    assert hit.response == "answer" and miss is None
