import asyncio
import copy
from types import SimpleNamespace

import pytest

from semcache import SemanticCache
from semcache.integrations import wrap_openai
from semcache.responses import ResponsesAccumulator, replay_events

R = pytest.importorskip("openai.types.responses")
from pydantic import TypeAdapter  # noqa: E402

EVENTS = TypeAdapter(R.ResponseStreamEvent)
Q = "How do I reverse a list in Python?"
PARA = "python: how to reverse a list"


def response_json(text="Use slicing: list[::-1]", status="completed", output=None, **extra):
    data = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1.0,
        "model": "gpt-test",
        "status": status,
        "output": output
        if output is not None
        else [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "total_tokens": 2_000_000,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    data.update(extra)
    R.Response.model_validate(data)  # always a valid Response
    return data


def live_stream(parts=("Use ", "slicing: ", "list[::-1]"), final="response.completed", **extra):
    """Events as the API streams them: text arrives in several deltas."""
    text = "".join(parts)
    final_response = response_json(text, **extra)
    start = {**copy.deepcopy(final_response), "status": "in_progress", "output": [], "usage": None}
    msg = {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress"}
    ids = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    raw = [
        {"type": "response.created", "response": start},
        {"type": "response.in_progress", "response": start},
        {"type": "response.output_item.added", "output_index": 0, "item": {**msg, "content": []}},
        {
            "type": "response.content_part.added",
            "part": {"type": "output_text", "text": "", "annotations": []},
            **ids,
        },
    ]
    raw += [
        {"type": "response.output_text.delta", "delta": p, "logprobs": [], **ids} for p in parts
    ]
    done_part = {"type": "output_text", "text": text, "annotations": []}
    raw += [
        {"type": "response.output_text.done", "text": text, "logprobs": [], **ids},
        {"type": "response.content_part.done", "part": done_part, **ids},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {**msg, "status": "completed", "content": [done_part]},
        },
        {"type": final, "response": final_response},
    ]
    return [EVENTS.validate_python({**e, "sequence_number": i}) for i, e in enumerate(raw)]


class FakeResponses:
    def __init__(self, events, fail_after=None):
        self.calls = []
        self.events = events
        self.fail_after = fail_after

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("stream"):
            final = self.events[-1]
            return R.Response.model_validate(final.response.model_dump(mode="json"))
        return self._stream()

    def _stream(self):
        for i, event in enumerate(self.events):
            if self.fail_after is not None and i == self.fail_after:
                raise ConnectionError("stream dropped")
            yield event


def client_for(cache, events, **kwargs):
    fake = FakeResponses(events, **kwargs)
    chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))
    raw = SimpleNamespace(responses=fake, chat=chat)
    return wrap_openai(raw, cache), fake


def text_of(events):
    return "".join(e.delta for e in events if e.type == "response.output_text.delta")


@pytest.fixture
def cache():
    return SemanticCache(pricing={"gpt-test": (1.0, 2.0)})


def test_stream_miss_passes_through_then_hit_replays(cache):
    client, fake = client_for(cache, live_stream())
    first = list(client.responses.create(model="gpt-test", input=Q, stream=True))
    assert first == fake.events  # the caller sees the API's events unchanged
    assert len(cache) == 1

    replay = list(client.responses.create(model="gpt-test", input=PARA, stream=True))
    assert len(fake.calls) == 1
    assert text_of(replay) == "Use slicing: list[::-1]"
    assert [e.type for e in replay][:2] == ["response.created", "response.in_progress"]
    assert replay[-1].type == "response.completed"
    assert replay[-1].response.output_text == "Use slicing: list[::-1]"
    assert isinstance(replay[0], R.ResponseCreatedEvent)
    assert isinstance(replay[-1], R.ResponseCompletedEvent)
    assert [e.sequence_number for e in replay] == list(range(len(replay)))
    assert cache.stats.cost_saved == pytest.approx(3.0)


def test_replayed_events_are_sdk_types():
    events = replay_events(response_json())
    rebuilt = [EVENTS.validate_python(e) for e in events]  # every event validates
    kinds = [type(e).__name__ for e in rebuilt]
    assert kinds[0] == "ResponseCreatedEvent" and kinds[-1] == "ResponseCompletedEvent"
    assert "ResponseTextDeltaEvent" in kinds


def test_streamed_answer_serves_plain_request_and_back(cache):
    client, fake = client_for(cache, live_stream(("Paris",)))
    list(
        client.responses.create(
            model="gpt-test", input="What is the capital of France?", stream=True
        )
    )
    plain = client.responses.create(model="gpt-test", input="what is the capital of france")
    assert isinstance(plain, R.Response) and plain.output_text == "Paris"

    other = SemanticCache()
    client2, fake2 = client_for(other, live_stream(("Rome",)))
    client2.responses.create(model="gpt-test", input="What is the capital of Italy?")  # plain miss
    streamed = list(
        client2.responses.create(
            model="gpt-test", input="What is the capital of Italy?", stream=True
        )
    )
    assert text_of(streamed) == "Rome" and len(fake2.calls) == 1


def test_context_manager_use(cache):
    client, fake = client_for(cache, live_stream(("hi",)))
    for _ in range(2):
        with client.responses.create(model="gpt-test", input=Q, stream=True) as stream:
            assert text_of(list(stream)) == "hi"
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "final,extra",
    [
        (
            "response.incomplete",
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        ),
        (
            "response.failed",
            {"status": "failed", "error": {"code": "server_error", "message": "x"}},
        ),
    ],
)
def test_unfinished_streams_are_not_cached(cache, final, extra):
    client, fake = client_for(cache, live_stream(final=final, **extra))
    list(client.responses.create(model="gpt-test", input=Q, stream=True))
    assert len(cache) == 0


def test_partial_dropped_and_error_streams_are_not_cached(cache):
    client, fake = client_for(cache, live_stream())
    stream = client.responses.create(model="gpt-test", input=Q, stream=True)
    next(stream)
    next(stream)  # caller stops reading early
    assert len(cache) == 0

    client, fake = client_for(cache, live_stream(), fail_after=5)
    with pytest.raises(ConnectionError):
        list(client.responses.create(model="gpt-test", input=Q, stream=True))
    assert len(cache) == 0

    acc = ResponsesAccumulator()
    for e in live_stream():
        acc.add(e)
    acc.add({"type": "error", "message": "late error", "sequence_number": 99})
    assert acc.result() is None


def test_function_call_and_reasoning_items_round_trip():
    output = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
        },
    ]
    data = response_json(output=output)
    events = [EVENTS.validate_python(e) for e in replay_events(data)]
    kinds = [e.type for e in events]
    assert "response.function_call_arguments.delta" in kinds
    acc = ResponsesAccumulator()
    for e in events:
        acc.add(e)

    # replay -> accumulate gives back the same response
    def dumped(d):
        return R.Response.model_validate(d).model_dump(mode="json")

    assert dumped(acc.result()) == dumped(data)


def test_async_stream(cache):
    events = live_stream(("async ", "answer"))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)

        async def gen():
            for e in events:
                yield e

        return gen()

    raw = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    client = wrap_openai(raw, cache)

    async def run():
        out = []
        for _ in range(2):
            stream = await client.responses.create(model="gpt-test", input=Q, stream=True)
            async with stream:
                out.append([e async for e in stream])
        return out

    first, second = asyncio.run(run())
    assert first == events
    assert text_of(second) == "async answer"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "final,extra",
    [
        (
            "response.incomplete",
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
        ),
        (
            "response.failed",
            {"status": "failed", "error": {"code": "server_error", "message": "x"}},
        ),
    ],
)
def test_accumulator_keeps_nothing_from_unfinished_streams(final, extra):
    # Checked directly: the cache's completeness check is a second, separate safety net.
    acc = ResponsesAccumulator()
    for e in live_stream(final=final, **extra):
        acc.add(e)
    acc.end_of_stream()
    assert acc.result() is None
    complete = ResponsesAccumulator()
    for e in live_stream():
        complete.add(e)
    assert complete.result()["status"] == "completed"
