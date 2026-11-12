import asyncio
from types import SimpleNamespace

import pytest

from semcache import SemanticCache
from semcache.integrations import wrap_anthropic, wrap_openai
from semcache.streaming import (
    AnthropicAccumulator,
    OpenAIAccumulator,
    anthropic_events,
    openai_chunks,
)

at = pytest.importorskip("anthropic.types")
openai_chat = pytest.importorskip("openai.types.chat")

Q = "How do I reverse a list in Python?"
PARA = "python: how to reverse a list"

EVENT_TYPES = {
    "message_start": at.RawMessageStartEvent,
    "content_block_start": at.RawContentBlockStartEvent,
    "content_block_delta": at.RawContentBlockDeltaEvent,
    "content_block_stop": at.RawContentBlockStopEvent,
    "message_delta": at.RawMessageDeltaEvent,
    "message_stop": at.RawMessageStopEvent,
}


def sdk_events(raw):
    return [EVENT_TYPES[e["type"]].model_validate(e) for e in raw]


def anthropic_stream(text_parts, stop_reason="end_turn", extra_blocks=()):
    """Events as the Anthropic API streams them: text arrives in several deltas."""
    raw = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1_000_000, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    raw += [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}
        for t in text_parts
    ]
    raw.append({"type": "content_block_stop", "index": 0})
    for i, (start, deltas) in enumerate(extra_blocks, start=1):
        raw.append({"type": "content_block_start", "index": i, "content_block": start})
        raw += [{"type": "content_block_delta", "index": i, "delta": d} for d in deltas]
        raw.append({"type": "content_block_stop", "index": i})
    raw += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 1_000_000},
        },
        {"type": "message_stop"},
    ]
    return sdk_events(raw)


class FakeAnthropic:
    def __init__(self, events, fail_after=None):
        self.calls = []
        self.events = events
        self.fail_after = fail_after

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        text = "".join(e.delta.text for e in self.events if e.type == "content_block_delta")
        return at.Message.model_validate(
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            }
        )

    def _stream(self):
        for i, event in enumerate(self.events):
            if self.fail_after is not None and i == self.fail_after:
                raise ConnectionError("stream dropped")
            yield event


@pytest.fixture
def cache():
    return SemanticCache(pricing={"claude-test": (1.0, 2.0), "gpt-test": (1.0, 2.0)})


def ask(client, prompt, **kwargs):
    return client.messages.create(
        model="claude-test",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )


def text_of(events):
    return "".join(
        e.delta.text
        for e in events
        if getattr(e, "type", None) == "content_block_delta"
        and getattr(e.delta, "type", None) == "text_delta"
    )


def test_anthropic_stream_miss_passes_through_and_is_cached(cache):
    fake = FakeAnthropic(anthropic_stream(["Use ", "slicing: ", "list[::-1]"]))
    client = wrap_anthropic(SimpleNamespace(messages=fake), cache)

    first = list(ask(client, Q, stream=True))
    assert first == fake.events  # the caller sees the provider's events unchanged
    assert len(cache) == 1

    replay = list(ask(client, PARA, stream=True))  # a reworded question, streamed
    assert len(fake.calls) == 1
    assert text_of(replay) == "Use slicing: list[::-1]"
    assert isinstance(replay[0], at.RawMessageStartEvent)
    assert [e.type for e in replay][-2:] == ["message_delta", "message_stop"]
    assert replay[-2].delta.stop_reason == "end_turn"
    assert cache.stats.cost_saved == pytest.approx(3.0)  # 1M in at $1 + 1M out at $2


def test_streamed_answer_serves_non_streamed_request_and_back(cache):
    fake = FakeAnthropic(anthropic_stream(["Paris"]))
    client = wrap_anthropic(SimpleNamespace(messages=fake), cache)
    list(ask(client, "What is the capital of France?", stream=True))
    message = ask(client, "what is the capital of france")
    assert isinstance(message, at.Message) and message.content[0].text == "Paris"

    other = SemanticCache()
    fake2 = FakeAnthropic(anthropic_stream(["Rome"]))
    client2 = wrap_anthropic(SimpleNamespace(messages=fake2), other)
    ask(client2, "What is the capital of Italy?")  # non-streamed miss
    assert text_of(list(ask(client2, "What is the capital of Italy?", stream=True))) == "Rome"
    assert len(fake2.calls) == 1


def test_context_manager_use(cache):
    fake = FakeAnthropic(anthropic_stream(["hi"]))
    client = wrap_anthropic(SimpleNamespace(messages=fake), cache)
    with ask(client, Q, stream=True) as stream:
        assert text_of(list(stream)) == "hi"
    with ask(client, Q, stream=True) as stream:
        assert text_of(list(stream)) == "hi"
    assert len(fake.calls) == 1


def test_partial_failed_or_incomplete_streams_are_not_cached(cache):
    fake = FakeAnthropic(anthropic_stream(["a", "b", "c"]))
    client = wrap_anthropic(SimpleNamespace(messages=fake), cache)
    stream = ask(client, Q, stream=True)
    next(stream)
    next(stream)  # caller stops reading early
    assert len(cache) == 0

    dropped = FakeAnthropic(anthropic_stream(["a", "b"]), fail_after=3)
    client = wrap_anthropic(SimpleNamespace(messages=dropped), cache)
    with pytest.raises(ConnectionError):
        list(ask(client, Q, stream=True))
    assert len(cache) == 0

    for reason in ("max_tokens", "refusal"):
        cut = FakeAnthropic(anthropic_stream(["partial"], stop_reason=reason))
        client = wrap_anthropic(SimpleNamespace(messages=cut), cache)
        list(ask(client, Q, stream=True))
        assert len(cache) == 0


def test_tool_use_thinking_and_citations_round_trip():
    tool = (
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
        [
            {"type": "input_json_delta", "partial_json": '{"city": "Pa'},
            {"type": "input_json_delta", "partial_json": 'ris", "days": 3}'},
        ],
    )
    thinking = (
        {"type": "thinking", "thinking": "", "signature": ""},
        [
            {"type": "thinking_delta", "thinking": "Let me check."},
            {"type": "signature_delta", "signature": "sig123"},
        ],
    )
    events = anthropic_stream(["Checking"], stop_reason="tool_use", extra_blocks=[tool, thinking])
    acc = AnthropicAccumulator()
    for e in events:
        acc.add(e)
    message = acc.result()
    assert message["content"][1]["input"] == {"city": "Paris", "days": 3}
    assert message["content"][2] == {
        "type": "thinking",
        "thinking": "Let me check.",
        "signature": "sig123",
    }
    at.Message.model_validate(message)  # a valid Message

    # Replaying and re-accumulating gives the same message.
    acc2 = AnthropicAccumulator()
    for e in sdk_events(anthropic_events(message)):
        acc2.add(e)
    assert acc2.result() == message


def test_unknown_delta_is_not_cached():
    acc = AnthropicAccumulator()
    for e in anthropic_stream(["x"]):
        acc.add(e)
    acc.add({"type": "content_block_delta", "index": 0, "delta": {"type": "brand_new_delta"}})
    assert acc.result() is None


# -- OpenAI -----------------------------------------------------------------------------


def openai_stream(parts, finish="stop", usage=True, tool_call=False):
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test"}
    raw = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    ]
    raw += [
        {**base, "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}
        for p in parts
    ]
    if tool_call:
        for args in ('{"q": ', '"x"}'):
            raw.append(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1" if args.startswith("{") else None,
                                        "type": "function",
                                        "function": {
                                            "name": "search" if args.startswith("{") else None,
                                            "arguments": args,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
    raw.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage:
        raw.append(
            {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                    "total_tokens": 1_000_000,
                },
            }
        )
    return [openai_chat.ChatCompletionChunk.model_validate(c) for c in raw]


class FakeOpenAI:
    def __init__(self, chunks):
        self.calls = []
        self.chunks = chunks

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("stream")
        return iter(self.chunks)


def openai_client(cache, chunks):
    fake = FakeOpenAI(chunks)
    raw = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return wrap_openai(raw, cache), fake


def chat(client, prompt, **kwargs):
    return client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": prompt}], **kwargs
    )


def test_openai_stream_cached_and_replayed(cache):
    client, fake = openai_client(cache, openai_stream(["Use ", "list[::-1]"]))
    first = list(chat(client, Q, stream=True, stream_options={"include_usage": True}))
    assert first == fake.chunks
    replay = list(chat(client, PARA, stream=True))  # stream_options doesn't change the key
    assert len(fake.calls) == 1
    assert isinstance(replay[0], openai_chat.ChatCompletionChunk)
    text = "".join(c.choices[0].delta.content or "" for c in replay if c.choices)
    assert text == "Use list[::-1]"
    assert [c.choices[0].finish_reason for c in replay if c.choices][-1] == "stop"
    assert replay[-1].usage.prompt_tokens == 1_000_000
    assert cache.stats.cost_saved == pytest.approx(1.0)

    completion = chat(client, Q)  # non-streamed request served from the streamed answer
    assert isinstance(completion, openai_chat.ChatCompletion)
    assert completion.choices[0].message.content == "Use list[::-1]"


def test_openai_tool_calls_round_trip():
    acc = OpenAIAccumulator()
    for c in openai_stream([], finish="tool_calls", tool_call=True):
        acc.add(c)
    acc.end_of_stream()
    completion = acc.result()
    call = completion["choices"][0]["message"]["tool_calls"][0]
    assert call == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q": "x"}'},
    }
    openai_chat.ChatCompletion.model_validate(completion)
    acc2 = OpenAIAccumulator()
    for c in openai_chunks(completion):
        acc2.add(openai_chat.ChatCompletionChunk.model_validate(c))
    acc2.end_of_stream()
    assert acc2.result() == completion


def test_openai_truncated_or_unfinished_streams_not_cached(cache):
    client, _ = openai_client(cache, openai_stream(["partial"], finish="length"))
    list(chat(client, Q, stream=True))
    assert len(cache) == 0
    acc = OpenAIAccumulator()
    for c in openai_stream(["x"])[:2]:  # stream ended before any finish_reason
        acc.add(c)
    acc.end_of_stream()
    assert acc.result() is None


def test_openai_without_usage_still_caches(cache):
    client, fake = openai_client(cache, openai_stream(["hi"], usage=False))
    list(chat(client, Q, stream=True))
    list(chat(client, Q, stream=True))
    assert len(fake.calls) == 1
    assert cache.stats.hits == 1 and cache.stats.cost_saved == 0.0


# -- async ------------------------------------------------------------------------------


def test_async_anthropic_stream(cache):
    events = anthropic_stream(["async ", "answer"])
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)

        async def gen():
            for e in events:
                yield e

        return gen()

    client = wrap_anthropic(SimpleNamespace(messages=SimpleNamespace(create=create)), cache)

    async def run():
        out = []
        for _ in range(2):
            stream = await ask(client, Q, stream=True)
            async with stream:
                out.append([e async for e in stream])
        return out

    first, second = asyncio.run(run())
    assert first == events
    assert text_of(second) == "async answer"
    assert len(calls) == 1


def test_async_openai_stream(cache):
    chunks = openai_stream(["hello"])
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)

        async def gen():
            for c in chunks:
                yield c

        return gen()

    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = wrap_openai(raw, cache)

    async def run():
        out = []
        for _ in range(2):
            stream = await chat(client, Q, stream=True)
            out.append([c async for c in stream])
        return out

    first, second = asyncio.run(run())
    assert first == chunks
    assert "".join(c.choices[0].delta.content or "" for c in second if c.choices) == "hello"
    assert len(calls) == 1
