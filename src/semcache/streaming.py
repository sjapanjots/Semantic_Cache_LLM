"""Streaming support for the client wrappers (``stream=True``).

On a **miss**, the provider's stream is handed to your code unchanged, wrapped in a
recorder that rebuilds the full response from the events as they pass. When the
stream ends normally the response is stored, exactly as a non-streamed one would be.
If you stop reading early, or the stream fails, nothing is stored.

On a **hit**, the cached response is replayed as a stream of the provider's own event
types, so code like ``for event in stream: ...`` works the same either way. The text
of each content block arrives as one event instead of many small ones.

Streamed and non-streamed requests share entries: a streamed answer can serve a later
non-streamed request, and the other way round.

Supported: Anthropic ``messages.create(stream=True)`` (text, tool use, thinking and
citations) and OpenAI ``chat.completions.create(stream=True)`` (content, refusals
and tool calls). Anthropic's ``messages.stream()`` helper is passed through uncached.
OpenAI only reports token usage in a stream when you pass
``stream_options={"include_usage": True}``; without it, hits on that entry count
toward the hit rate but not toward money saved.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

__all__ = [
    "StreamFormat",
    "ANTHROPIC",
    "OPENAI",
    "AnthropicAccumulator",
    "OpenAIAccumulator",
    "anthropic_events",
    "openai_chunks",
    "RecordingStream",
    "AsyncRecordingStream",
    "ReplayStream",
    "AsyncReplayStream",
]


def _data(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


# -- Anthropic ----------------------------------------------------------------------------


class AnthropicAccumulator:
    """Rebuilds an Anthropic ``Message`` from ``messages.create(stream=True)`` events."""

    def __init__(self) -> None:
        self.message: dict[str, Any] | None = None
        self.blocks: dict[int, dict[str, Any]] = {}
        self.ok = True
        self.done = False

    def end_of_stream(self) -> None:
        """Called when the stream runs out; Anthropic marks completion with message_stop."""

    def add(self, event: Any) -> None:
        e = _data(event)
        kind = e.get("type") if isinstance(e, dict) else None
        if kind == "message_start":
            self.message = copy.deepcopy(e["message"])
        elif kind == "content_block_start":
            block = copy.deepcopy(e["content_block"])
            if block.get("type") == "tool_use":
                block["_json"] = ""
            self.blocks[e["index"]] = block
        elif kind == "content_block_delta":
            self._delta(self.blocks.get(e["index"]), e["delta"])
        elif kind == "content_block_stop":
            block = self.blocks.get(e["index"])
            if block is not None and "_json" in block:
                raw = block.pop("_json")
                try:
                    block["input"] = json.loads(raw) if raw else {}
                except ValueError:
                    self.ok = False
        elif kind == "message_delta":
            if self.message is None:
                self.ok = False
                return
            self.message.update({k: v for k, v in e["delta"].items() if v is not None})
            usage = self.message.setdefault("usage", {})
            usage.update({k: v for k, v in (e.get("usage") or {}).items() if v is not None})
        elif kind == "message_stop":
            self.done = True
        elif kind == "error":
            self.ok = False
        # "ping" and unknown event types carry nothing to keep.

    def _delta(self, block: dict[str, Any] | None, delta: dict[str, Any]) -> None:
        kind = delta.get("type")
        if block is None:
            self.ok = False
        elif kind == "text_delta":
            block["text"] = block.get("text", "") + delta["text"]
        elif kind == "input_json_delta" and "_json" in block:
            block["_json"] += delta["partial_json"]
        elif kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta["thinking"]
        elif kind == "signature_delta":
            block["signature"] = delta["signature"]
        elif kind == "citations_delta":
            block.setdefault("citations", []).append(delta["citation"])
        else:
            self.ok = False  # a delta we can't rebuild: don't cache a partial answer

    def result(self) -> dict[str, Any] | None:
        if not (self.ok and self.done and self.message is not None):
            return None
        message = copy.deepcopy(self.message)
        message["content"] = [self.blocks[i] for i in sorted(self.blocks)]
        return message


def anthropic_events(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``messages.create(stream=True)`` events that produce ``message``."""
    usage = dict(message.get("usage") or {})
    start = copy.deepcopy(message)
    start.update(content=[], stop_reason=None, stop_sequence=None)
    start["usage"] = {**usage, "output_tokens": 0}
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for i, block in enumerate(message.get("content") or []):
        kind = block.get("type")
        deltas: list[dict[str, Any]] = []
        if kind == "text":
            first = {**block, "text": ""}
            citations = first.pop("citations", None)
            if citations:
                first["citations"] = []
                deltas += [{"type": "citations_delta", "citation": c} for c in citations]
            if block.get("text"):
                deltas.insert(0, {"type": "text_delta", "text": block["text"]})
        elif kind == "tool_use":
            first = {**block, "input": {}}
            deltas.append(
                {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}
            )
        elif kind == "thinking":
            first = {**block, "thinking": "", "signature": ""}
            if block.get("thinking"):
                deltas.append({"type": "thinking_delta", "thinking": block["thinking"]})
            if block.get("signature"):
                deltas.append({"type": "signature_delta", "signature": block["signature"]})
        else:  # redacted thinking, server tool results, ...: sent whole in the start event
            first = block
        events.append({"type": "content_block_start", "index": i, "content_block": first})
        events += [{"type": "content_block_delta", "index": i, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": i})
    events.append(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason"),
                "stop_sequence": message.get("stop_sequence"),
            },
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        }
    )
    events.append({"type": "message_stop"})
    return events


def _anthropic_event(data: dict[str, Any]) -> Any:
    names = {
        "message_start": "RawMessageStartEvent",
        "content_block_start": "RawContentBlockStartEvent",
        "content_block_delta": "RawContentBlockDeltaEvent",
        "content_block_stop": "RawContentBlockStopEvent",
        "message_delta": "RawMessageDeltaEvent",
        "message_stop": "RawMessageStopEvent",
    }
    try:
        import anthropic.types as types

        return getattr(types, names[data["type"]]).model_validate(data)
    except Exception:
        return _to_namespace(data)


# -- OpenAI -------------------------------------------------------------------------------


class OpenAIAccumulator:
    """Rebuilds a ``ChatCompletion`` from ``chat.completions.create(stream=True)`` chunks."""

    def __init__(self) -> None:
        self.base: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.ok = True
        self.done = False

    def end_of_stream(self) -> None:
        """OpenAI has no final event: running out of chunks is the end."""
        self.done = True

    def add(self, chunk: Any) -> None:
        c = _data(chunk)
        if not isinstance(c, dict):
            self.ok = False
            return
        if not self.base:
            keep = ("id", "created", "model", "system_fingerprint", "service_tier")
            self.base = {k: c.get(k) for k in keep if c.get(k) is not None}
        if c.get("usage"):
            self.usage = c["usage"]
        for choice in c.get("choices") or []:
            entry = self.choices.setdefault(
                choice["index"],
                {
                    "index": choice["index"],
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": None,
                },
            )
            message = entry["message"]
            delta = choice.get("delta") or {}
            for field in ("content", "refusal"):
                if delta.get(field):
                    message[field] = (message.get(field) or "") + delta[field]
            for call in delta.get("tool_calls") or []:
                calls = message.setdefault("tool_calls", [])
                while len(calls) <= call["index"]:
                    calls.append({"id": None, "type": "function", "function": {}})
                target = calls[call["index"]]
                if call.get("id"):
                    target["id"] = call["id"]
                fn = call.get("function") or {}
                for part in ("name", "arguments"):
                    if fn.get(part):
                        target["function"][part] = target["function"].get(part, "") + fn[part]
            if choice.get("finish_reason"):
                entry["finish_reason"] = choice["finish_reason"]

    def result(self) -> dict[str, Any] | None:
        if not (self.ok and self.done and self.choices):
            return None
        if any(c["finish_reason"] is None for c in self.choices.values()):
            return None
        for choice in self.choices.values():
            for call in choice["message"].get("tool_calls") or []:
                call["function"].setdefault("name", "")
                call["function"].setdefault("arguments", "")
        return {
            **self.base,
            "object": "chat.completion",
            "choices": [self.choices[i] for i in sorted(self.choices)],
            "usage": self.usage,
        }


def openai_chunks(completion: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``chat.completions.create(stream=True)`` chunks that produce ``completion``."""
    keep = ("id", "created", "model", "system_fingerprint", "service_tier")
    base = {k: completion[k] for k in keep if completion.get(k) is not None}
    base["object"] = "chat.completion.chunk"
    chunks = []
    for choice in completion.get("choices") or []:
        message = choice.get("message") or {}
        delta: dict[str, Any] = {"role": "assistant"}
        for field in ("content", "refusal"):
            if message.get(field):
                delta[field] = message[field]
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {"index": i, **call} for i, call in enumerate(message["tool_calls"])
            ]
        chunks.append(
            {**base, "choices": [{"index": choice["index"], "delta": delta, "finish_reason": None}]}
        )
        chunks.append(
            {
                **base,
                "choices": [
                    {
                        "index": choice["index"],
                        "delta": {},
                        "finish_reason": choice["finish_reason"],
                    }
                ],
            }
        )
    if completion.get("usage"):
        chunks.append({**base, "choices": [], "usage": completion["usage"]})
    return chunks


def _openai_chunk(data: dict[str, Any]) -> Any:
    try:
        from openai.types.chat import ChatCompletionChunk

        return ChatCompletionChunk.model_validate(data)
    except Exception:
        return _to_namespace(data)


# -- provider formats ---------------------------------------------------------------------


@dataclass(frozen=True)
class StreamFormat:
    accumulator: Callable[[], Any]
    replay: Callable[[dict[str, Any]], list[dict[str, Any]]]
    rebuild_event: Callable[[dict[str, Any]], Any]


ANTHROPIC = StreamFormat(AnthropicAccumulator, anthropic_events, _anthropic_event)
OPENAI = StreamFormat(OpenAIAccumulator, openai_chunks, _openai_chunk)


# -- stream objects ------------------------------------------------------------------------


class RecordingStream:
    """Passes a provider stream through and stores the response once it completes."""

    def __init__(
        self, stream: Any, accumulator: Any, on_complete: Callable[[dict[str, Any]], None]
    ) -> None:
        self._stream = stream
        self._iter: Iterator[Any] | None = None
        self._acc = accumulator
        self._on_complete = on_complete
        self._finished = False

    def __iter__(self) -> RecordingStream:
        return self

    def __next__(self) -> Any:
        if self._iter is None:
            self._iter = iter(self._stream)
        try:
            item = next(self._iter)
        except StopIteration:
            self._finish()
            raise
        self._acc.add(item)
        return item

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._acc.end_of_stream()
        result = self._acc.result()
        if result is not None:
            self._on_complete(result)

    def __enter__(self) -> RecordingStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncRecordingStream:
    """Async version of :class:`RecordingStream` for ``AsyncAnthropic`` / ``AsyncOpenAI``."""

    def __init__(
        self, stream: Any, accumulator: Any, on_complete: Callable[[dict[str, Any]], None]
    ) -> None:
        self._stream = stream
        self._iter: Any = None
        self._acc = accumulator
        self._on_complete = on_complete
        self._finished = False

    def __aiter__(self) -> AsyncRecordingStream:
        return self

    async def __anext__(self) -> Any:
        if self._iter is None:
            self._iter = self._stream.__aiter__()
        try:
            item = await self._iter.__anext__()
        except StopAsyncIteration:
            await self._finish()
            raise
        self._acc.add(item)
        return item

    async def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._acc.end_of_stream()
        result = self._acc.result()
        if result is not None:
            await asyncio.to_thread(self._on_complete, result)

    async def __aenter__(self) -> AsyncRecordingStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class ReplayStream:
    """A cached response replayed as the provider's stream events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)

    def __iter__(self) -> ReplayStream:
        return self

    def __next__(self) -> Any:
        return next(self._events)

    def __enter__(self) -> ReplayStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._events = iter(())


class AsyncReplayStream:
    """Async version of :class:`ReplayStream`."""

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> AsyncReplayStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> AsyncReplayStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._events = iter(())
