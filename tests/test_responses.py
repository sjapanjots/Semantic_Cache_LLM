import asyncio
import copy
from types import SimpleNamespace

import pytest

from semcache import SemanticCache
from semcache.integrations import wrap_openai
from semcache.responses import is_complete, parse_request

responses_types = pytest.importorskip("openai.types.responses")
Response = responses_types.Response

Q = "How do I reverse a list in Python?"
PARA = "python: how to reverse a list"


def make_response(text="Use slicing: list[::-1]", status="completed", **extra):
    data = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1.0,
        "model": "gpt-test",
        "status": status,
        "output": [
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
    return Response.model_validate(data)


class FakeResponses:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or make_response()

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if kwargs.get("stream"):
            return iter(["event-1", "event-2"])
        return self.response


def client_for(cache, response=None):
    fake = FakeResponses(response)
    chat_calls = []

    def chat_create(**kwargs):
        chat_calls.append(kwargs)
        raise AssertionError("chat should not be called in these tests")

    raw = SimpleNamespace(
        responses=fake,
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        models="untouched",
    )
    return wrap_openai(raw, cache), fake


@pytest.fixture
def cache():
    return SemanticCache(pricing={"gpt-test": (1.0, 2.0)})


def test_string_input_cached_and_rebuilt(cache):
    client, fake = client_for(cache)
    first = client.responses.create(model="gpt-test", input=Q)
    second = client.responses.create(model="gpt-test", input=PARA)  # reworded
    assert len(fake.calls) == 1
    assert first is fake.response
    assert isinstance(second, Response) and second.output_text == "Use slicing: list[::-1]"
    assert cache.stats.cost_saved == pytest.approx(3.0)  # 1M in at $1 + 1M out at $2
    assert client.models == "untouched"


def test_message_list_input_with_text_parts(cache):
    client, fake = client_for(cache)
    as_parts = [{"role": "user", "content": [{"type": "input_text", "text": Q}]}]
    as_string = [{"role": "user", "content": PARA}]
    client.responses.create(model="gpt-test", input=as_parts)
    client.responses.create(model="gpt-test", input=as_string)
    client.responses.create(model="gpt-test", input=Q)  # plain string matches the same entry
    assert len(fake.calls) == 1


def test_context_must_match_exactly(cache):
    client, fake = client_for(cache)
    client.responses.create(model="gpt-test", input=Q)
    client.responses.create(model="gpt-test", input=Q, instructions="Answer in French.")
    client.responses.create(model="gpt-test", input=Q, previous_response_id="resp_prev")
    client.responses.create(model="gpt-test", input=Q, conversation="conv_1")
    client.responses.create(model="gpt-test", input=Q, temperature=1.5)
    client.responses.create(model="other-model", input=Q)
    history = [
        {"role": "user", "content": "I use Python 2."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": Q},
    ]
    client.responses.create(model="gpt-test", input=history)
    assert len(fake.calls) == 7
    # Options that don't change the answer are ignored when matching.
    client.responses.create(model="gpt-test", input=Q, store=False, user="u1", metadata={"a": "b"})
    client.responses.create(model="gpt-test", input=Q, prompt_cache_key="k", safety_identifier="s")
    assert len(fake.calls) == 7


def test_images_and_files_must_match(cache):
    client, fake = client_for(cache)

    def with_image(url):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": url},
                    {"type": "input_text", "text": "Describe this image"},
                ],
            }
        ]

    for url in ("https://a.example/1.png", "https://a.example/2.png", "https://a.example/1.png"):
        client.responses.create(model="gpt-test", input=with_image(url))
    assert len(fake.calls) == 2


def test_requests_that_are_not_cached(cache):
    client, fake = client_for(cache)
    tool_output = [
        {"role": "user", "content": Q},
        {"type": "function_call_output", "call_id": "c1", "output": "42"},
    ]
    for _ in range(2):
        client.responses.create(model="gpt-test", input=tool_output)
        client.responses.create(model="gpt-test", prompt={"id": "pmpt_1"})  # no input text
        client.responses.create(model="gpt-test", input="   ")
    assert len(fake.calls) == 6
    assert len(cache) == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
        {"status": "in_progress"},  # background=True
        {"status": "queued"},
        {"status": "failed", "error": {"code": "server_error", "message": "boom"}},
        {"status": "cancelled"},
    ],
)
def test_unfinished_answers_are_not_cached(cache, extra):
    status = extra.pop("status")
    client, fake = client_for(cache, make_response(status=status, **extra))
    client.responses.create(model="gpt-test", input=Q)
    client.responses.create(model="gpt-test", input=Q)
    assert len(fake.calls) == 2 and len(cache) == 0


def test_responses_and_chat_are_cached_separately_and_cleared_together(cache):
    client, fake = client_for(cache)
    client.responses.create(model="gpt-test", input=Q)
    entry = next(cache.store.entries())
    assert entry.scope.startswith("openai/responses/")
    assert cache.clear("openai") == 1


def test_async_client(cache):
    calls = []
    response = make_response("async answer")

    async def create(**kwargs):
        calls.append(kwargs)
        return response

    raw = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    client = wrap_openai(raw, cache)

    async def run():
        first = await client.responses.create(model="gpt-test", input=Q)
        second = await client.responses.create(model="gpt-test", input=PARA)
        return first, second

    first, second = asyncio.run(run())
    assert len(calls) == 1
    assert second.output_text == "async answer"


def test_client_without_responses_still_wraps_chat(cache):
    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: 1)))
    client = wrap_openai(raw, cache)
    assert not hasattr(client, "responses") or client.responses is None


def test_parse_request_and_is_complete_units():
    assert parse_request({"input": "hi"}) == ("hi", [], [])
    assert parse_request({"input": []}) is None
    assert parse_request({}) is None
    assert parse_request({"input": [{"role": "assistant", "content": "x"}]}) is None
    text, history, attachments = parse_request(
        {
            "input": [
                {"role": "user", "content": "first"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "a"},
                        {"type": "input_file", "file_id": "f1"},
                        {"type": "input_text", "text": "b"},
                    ],
                },
            ]
        }
    )
    assert text == "a\nb"
    assert history == [{"role": "user", "content": "first"}]
    assert attachments == [{"type": "input_file", "file_id": "f1"}]
    assert is_complete({"status": "completed"})
    assert not is_complete({"status": "completed", "incomplete_details": {"reason": "x"}})
    assert not is_complete({"status": "completed", "error": {"code": "e"}})
    assert not is_complete({"status": "incomplete"})
