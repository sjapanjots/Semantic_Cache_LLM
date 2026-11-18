"""Caching for OpenAI's Responses API (``client.responses.create``).

``wrap_openai`` uses this module so that both OpenAI endpoints are cached::

    from openai import OpenAI
    from semcache import SemanticCache
    from semcache.integrations import wrap_openai

    client = wrap_openai(OpenAI(), SemanticCache(path="cache.db"))
    client.responses.create(model="gpt-x", input="How do I reverse a list in Python?")

How a request is matched, mirroring chat completions:

* ``input`` as a string, or the text of the **last user message** in an ``input`` list,
  is compared by meaning.
* Everything else must be identical: ``instructions``, earlier items in ``input``,
  images and files in the last message, ``tools``, ``temperature``, and
  ``previous_response_id`` / ``conversation`` (history kept on OpenAI's side that the
  cache can't see, so an answer is only reused for the same conversation state).
* A request whose last input item isn't a user message (for example a function call
  output) is passed through uncached.
* Only finished answers are stored: ``status == "completed"`` with no
  ``incomplete_details`` or ``error``. Background (``background=True``), incomplete,
  failed and cancelled responses are not.
* ``stream=True`` is cached too. On a miss the stream reaches your code unchanged
  and the finished response, taken from the final ``response.completed`` event, is
  stored. On a hit the stored response is replayed as the standard event sequence
  (``response.created`` ... ``response.output_text.delta`` ... ``response.completed``),
  rebuilt as the SDK's own event types. Streamed and non-streamed requests share
  entries.

Responses and chat completions are cached separately (their answers have different
shapes), under ``<namespace>/responses``.
"""

from __future__ import annotations

import copy
from typing import Any

from .streaming import StreamFormat, _to_namespace

__all__ = [
    "parse_request",
    "is_complete",
    "rebuild",
    "ResponsesAccumulator",
    "replay_events",
    "STREAM_FORMAT",
]

_TEXT_PARTS = {"input_text", "text"}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def parse_request(kwargs: dict[str, Any]) -> tuple[str, list[Any], list[Any]] | None:
    """Split a ``responses.create`` request into ``(prompt, history, attachments)``.

    Returns ``None`` when the request shouldn't be cached.
    """
    data = kwargs.get("input")
    if isinstance(data, str):
        return (data, [], []) if data.strip() else None
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    if _field(last, "role") != "user" or _field(last, "type", "message") != "message":
        return None
    content = _field(last, "content")
    attachments: list[Any] = []
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = [_field(p, "text") for p in content if _field(p, "type") in _TEXT_PARTS]
        text = "\n".join(t for t in texts if isinstance(t, str))
        attachments = [_jsonable(p) for p in content if _field(p, "type") not in _TEXT_PARTS]
    else:
        return None
    if not text.strip():
        return None
    return text, [_jsonable(item) for item in data[:-1]], attachments


def is_complete(data: Any) -> bool:
    """Whether a response (as JSON) is a finished answer worth caching."""
    return (
        _field(data, "status") == "completed"
        and not _field(data, "incomplete_details")
        and not _field(data, "error")
    )


def rebuild(data: Any) -> Any:
    """Turn a cached response back into ``openai.types.responses.Response`` when possible."""
    if not isinstance(data, dict):
        return data
    try:
        from openai.types.responses import Response

        return Response.model_validate(data)
    except Exception:
        return _to_namespace(data)


# -- streaming ----------------------------------------------------------------------------


class ResponsesAccumulator:
    """Keeps the finished response from a ``responses.create(stream=True)`` stream.

    The final ``response.completed`` event carries the whole response, so nothing has
    to be pieced together from deltas. ``response.incomplete``, ``response.failed`` and
    ``error`` events mean there is nothing to cache.
    """

    def __init__(self) -> None:
        self.response: dict[str, Any] | None = None
        self.ok = True
        self.done = False

    def end_of_stream(self) -> None:
        """Completion is marked by the ``response.completed`` event, not by running out."""

    def add(self, event: Any) -> None:
        e = _jsonable(event)
        kind = _field(e, "type")
        if kind == "response.completed":
            self.response = copy.deepcopy(_field(e, "response"))
            self.done = True
        elif kind in ("response.incomplete", "response.failed", "error"):
            self.ok = False

    def result(self) -> dict[str, Any] | None:
        if not (self.ok and self.done and isinstance(self.response, dict)):
            return None
        return self.response


def replay_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``responses.create(stream=True)`` events that produce ``response``."""
    events: list[dict[str, Any]] = []

    def emit(kind: str, **fields: Any) -> None:
        events.append({"type": kind, "sequence_number": len(events), **fields})

    start = copy.deepcopy(response)
    start.update(status="in_progress", output=[], usage=None)
    emit("response.created", response=start)
    emit("response.in_progress", response=copy.deepcopy(start))
    for i, item in enumerate(response.get("output") or []):
        kind = item.get("type")
        item_id = item.get("id", "")
        if kind == "message":
            emit("response.output_item.added", output_index=i, item={**item, "content": []})
            for j, part in enumerate(item.get("content") or []):
                ids = {"item_id": item_id, "output_index": i, "content_index": j}
                if part.get("type") == "output_text":
                    empty = {**part, "text": "", "annotations": []}
                    emit("response.content_part.added", part=empty, **ids)
                    text = part.get("text", "")
                    emit("response.output_text.delta", delta=text, logprobs=[], **ids)
                    emit("response.output_text.done", text=text, logprobs=[], **ids)
                else:  # refusals and other parts arrive whole
                    emit("response.content_part.added", part=part, **ids)
                emit("response.content_part.done", part=part, **ids)
        elif kind == "function_call":
            emit("response.output_item.added", output_index=i, item={**item, "arguments": ""})
            args = item.get("arguments", "")
            ids = {"item_id": item_id, "output_index": i}
            emit("response.function_call_arguments.delta", delta=args, **ids)
            emit("response.function_call_arguments.done", arguments=args, **ids)
        else:  # reasoning, web search and other items arrive whole
            emit("response.output_item.added", output_index=i, item=item)
        emit("response.output_item.done", output_index=i, item=item)
    emit("response.completed", response=copy.deepcopy(response))
    return events


_EVENT_ADAPTER: Any = None


def _rebuild_event(data: dict[str, Any]) -> Any:
    """Rebuild an event as the SDK's own type (built once), or fall back to attributes."""
    global _EVENT_ADAPTER
    try:
        if _EVENT_ADAPTER is None:
            from openai.types.responses import ResponseStreamEvent
            from pydantic import TypeAdapter

            _EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)
        return _EVENT_ADAPTER.validate_python(data)
    except Exception:
        return _to_namespace(data)


STREAM_FORMAT = StreamFormat(ResponsesAccumulator, replay_events, _rebuild_event)
