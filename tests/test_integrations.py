import asyncio
from types import SimpleNamespace

import pytest

from semcache import SemanticCache
from semcache.integrations import wrap_anthropic, wrap_openai


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self.usage = SimpleNamespace(**data["usage"])

    def model_dump(self, mode="json"):
        return self._data


class FakeCreate:
    def __init__(self, usage):
        self.calls = []
        self.usage = usage

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        text = f"answer #{len(self.calls)}"
        return FakeResponse({"id": str(len(self.calls)), "text": text, "usage": self.usage})


def fake_openai():
    create = FakeCreate({"prompt_tokens": 1_000_000, "completion_tokens": 0})
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models="untouched",
    )
    return client, create


def fake_anthropic():
    create = FakeCreate({"input_tokens": 1_000_000, "output_tokens": 0})
    return SimpleNamespace(messages=SimpleNamespace(create=create)), create


@pytest.fixture
def cache():
    return SemanticCache(pricing={"gpt-test": (2.0, 0.0), "claude-test": (3.0, 0.0)})


def user(text):
    return {"role": "user", "content": text}


def test_openai_wrapper_caches_semantic_repeats(cache):
    raw, create = fake_openai()
    client = wrap_openai(raw, cache)
    first = client.chat.completions.create(
        model="gpt-test", messages=[user("How do I reverse a list in Python?")]
    )
    second = client.chat.completions.create(
        model="gpt-test", messages=[user("python: how to reverse a list")]
    )
    assert len(create.calls) == 1
    assert first._data["text"] == "answer #1"
    assert second.text == "answer #1"  # rebuilt from cached JSON
    assert cache.stats.cost_saved == pytest.approx(2.0)
    assert client.models == "untouched"
    assert client.semcache is cache


def test_context_changes_bypass_cache(cache):
    raw, create = fake_openai()
    client = wrap_openai(raw, cache)
    q = [user("What is the capital of France?")]
    client.chat.completions.create(model="gpt-test", messages=q)
    client.chat.completions.create(model="gpt-test", messages=q, temperature=1.5)
    client.chat.completions.create(
        model="gpt-test", messages=[{"role": "system", "content": "Be terse."}, *q]
    )
    client.chat.completions.create(model="other-model", messages=q)
    assert len(create.calls) == 4
    client.chat.completions.create(model="gpt-test", messages=q, timeout=30)
    assert len(create.calls) == 4  # timeout does not change the answer


def test_non_user_last_message_not_cached(cache):
    raw, create = fake_openai()
    client = wrap_openai(raw, cache)
    msgs = [user("hi"), {"role": "assistant", "content": "hello"}]
    client.chat.completions.create(model="gpt-test", messages=msgs)
    client.chat.completions.create(model="gpt-test", messages=msgs)
    assert len(create.calls) == 2


def test_anthropic_wrapper_with_content_blocks(cache):
    raw, create = fake_anthropic()
    client = wrap_anthropic(raw, cache)
    blocks = [{"type": "text", "text": "Explain what a semantic cache is"}]
    for content in (blocks, "what is a semantic cache"):
        client.messages.create(
            model="claude-test", max_tokens=100, system="Be helpful.", messages=[user(content)]
        )
    assert len(create.calls) == 1
    assert cache.stats.cost_saved == pytest.approx(3.0)


def test_async_client(cache):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return FakeResponse({"text": "hi", "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = wrap_anthropic(SimpleNamespace(messages=SimpleNamespace(create=create)), cache)

    async def run():
        for _ in range(2):
            await client.messages.create(model="claude-test", messages=[user("hello there")])

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "pause_turn"])
def test_incomplete_anthropic_answers_not_cached(cache, stop_reason):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return FakeResponse(
            {"stop_reason": stop_reason, "usage": {"input_tokens": 1, "output_tokens": 1}}
        )

    client = wrap_anthropic(SimpleNamespace(messages=SimpleNamespace(create=create)), cache)
    for _ in range(2):
        client.messages.create(model="claude-test", messages=[user("tell me a story")])
    assert len(calls) == 2


def test_truncated_openai_answers_not_cached(cache):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return FakeResponse(
            {
                "choices": [{"finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = wrap_openai(raw, cache)
    for _ in range(2):
        client.chat.completions.create(model="gpt-test", messages=[user("tell me a story")])
    assert len(calls) == 2


def image_message(data, text="Describe this image"):
    return user(
        [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            },
            {"type": "text", "text": text},
        ]
    )


def test_different_images_never_share_an_answer(cache):
    raw, create = fake_anthropic()
    client = wrap_anthropic(raw, cache)
    for data in ("AAAA", "BBBB", "AAAA"):
        client.messages.create(model="claude-test", max_tokens=10, messages=[image_message(data)])
    assert len(create.calls) == 2  # the repeated image is served from the cache


def test_text_block_list_and_plain_string_match(cache):
    raw, create = fake_openai()
    client = wrap_openai(raw, cache)
    client.chat.completions.create(model="gpt-test", messages=[user("hello there")])
    client.chat.completions.create(
        model="gpt-test", messages=[user([{"type": "text", "text": "hello there"}])]
    )
    assert len(create.calls) == 1


def test_async_wrapper_does_not_block_event_loop(cache):
    import time

    from semcache import CallableEmbedder

    def slow_embed(texts):
        time.sleep(0.2)
        return [[1.0, float(len(t))] for t in texts]

    slow_cache = SemanticCache(embedder=CallableEmbedder(slow_embed, "slow"))

    async def create(**kwargs):
        return FakeResponse({"text": "hi", "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = wrap_anthropic(SimpleNamespace(messages=SimpleNamespace(create=create)), slow_cache)

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(20):
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        await client.messages.create(model="claude-test", messages=[user("hello there")])
        seen = ticks
        await task
        return seen

    assert asyncio.run(run()) > 0
