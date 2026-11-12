"""Drop-in caching wrappers for the OpenAI and Anthropic Python clients.

::

    from anthropic import Anthropic
    from semcache import SemanticCache
    from semcache.integrations import wrap_anthropic

    client = wrap_anthropic(Anthropic(), SemanticCache(path="cache.db"))
    client.messages.create(model="claude-opus-5-5", max_tokens=1024, messages=[...])

How requests are matched:

* The **last user message** is the text compared by meaning.
* Everything else (system prompt, earlier turns, images and files in the last
  message, tools, temperature, max tokens, ...) is hashed into the cache namespace,
  so a hit only happens when that context is identical. Two people asking the same
  question inside different conversations never share an answer.
* ``stream=True`` requests are cached too: see :mod:`semcache.streaming`.
* Incomplete answers are not stored: Anthropic ``stop_reason`` of ``max_tokens``,
  ``refusal`` or ``pause_turn``, and OpenAI ``finish_reason`` of ``length`` or
  ``content_filter``.

The wrappers are duck-typed: they only touch ``chat.completions.create`` (OpenAI) or
``messages.create`` (Anthropic), so they also work with compatible clients and fakes.
Sync and async clients are both supported.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from .cache import SemanticCache
from .streaming import (
    ANTHROPIC,
    OPENAI,
    AsyncRecordingStream,
    AsyncReplayStream,
    RecordingStream,
    ReplayStream,
    StreamFormat,
)

__all__ = ["wrap_openai", "wrap_anthropic"]

# Request options that do not change what the model answers.
_IGNORED_PARAMS = frozenset(
    {
        "messages",
        "model",
        "stream",
        "stream_options",
        "timeout",
        "extra_headers",
        "extra_query",
        "extra_body",
        "user",
        "metadata",
        "store",
        "service_tier",
    }  # fmt: skip
)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_of(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_field(p, "text") for p in content if _field(p, "type") == "text"]
        parts = [p for p in parts if isinstance(p, str)]
        return "\n".join(parts) if parts else None
    return None


def _to_jsonable(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return response


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _rehydrator(module: str, attr: str) -> Callable[[Any], Any]:
    """Rebuild the SDK's response type from cached JSON, or fall back to attribute access."""

    def rebuild(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        try:
            cls = getattr(__import__(module, fromlist=[attr]), attr)
            return cls.model_validate(data)
        except Exception:
            return _to_namespace(data)

    return rebuild


def _anthropic_complete(response: Any) -> bool:
    return _field(response, "stop_reason") not in {"max_tokens", "refusal", "pause_turn"}


def _openai_complete(response: Any) -> bool:
    choices = _field(response, "choices") or []
    return not any(_field(c, "finish_reason") in {"length", "content_filter"} for c in choices)


class _ChatCacher:
    def __init__(
        self,
        cache: SemanticCache,
        namespace: str,
        usage_fields: tuple[str, str],
        rebuild: Callable[[Any], Any],
        is_complete: Callable[[Any], bool],
        stream_format: StreamFormat,
    ) -> None:
        self.cache = cache
        self.namespace = namespace
        self.usage_fields = usage_fields
        self.rebuild = rebuild
        self.is_complete = is_complete
        self.stream_format = stream_format

    def key(self, kwargs: dict[str, Any]) -> tuple[str, str] | None:
        """Return ``(prompt, namespace)`` or ``None`` if this request should not be cached."""
        messages = list(kwargs.get("messages") or [])
        if not messages or _field(messages[-1], "role") != "user":
            return None
        content = _field(messages[-1], "content")
        prompt = _text_of(content)
        if not prompt:
            return None
        context = {k: v for k, v in kwargs.items() if k not in _IGNORED_PARAMS}
        context["history"] = [_to_jsonable(m) for m in messages[:-1]]
        if isinstance(content, list):
            # Images, documents and other non-text parts must match exactly too.
            attachments = [_to_jsonable(p) for p in content if _field(p, "type") != "text"]
            if attachments:
                context["attachments"] = attachments
        digest = hashlib.sha256(
            json.dumps(context, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return prompt, f"{self.namespace}/{digest}"

    def lookup_data(self, kwargs: dict[str, Any]) -> tuple[Any, tuple[str, str] | None]:
        """The cached response as stored (JSON), or ``None``, plus the request's key."""
        key = self.key(kwargs)
        if key is None:
            return None, None
        hit = self.cache.lookup(key[0], model=kwargs.get("model"), namespace=key[1])
        return (hit.response if hit else None), key

    def lookup(self, kwargs: dict[str, Any]) -> tuple[Any, tuple[str, str] | None]:
        data, key = self.lookup_data(kwargs)
        return (self.rebuild(data) if data is not None else None), key

    def replay(self, data: Any) -> list[Any]:
        fmt = self.stream_format
        return [fmt.rebuild_event(event) for event in fmt.replay(data)]

    def store(self, kwargs: dict[str, Any], key: tuple[str, str], response: Any) -> None:
        data = _to_jsonable(response)
        if not self.is_complete(data):
            return
        usage = _field(data, "usage")
        in_field, out_field = self.usage_fields
        self.cache.set(
            key[0],
            data,
            model=kwargs.get("model"),
            namespace=key[1],
            input_tokens=int(_field(usage, in_field, 0) or 0),
            output_tokens=int(_field(usage, out_field, 0) or 0),
        )


def _cached_create(create: Callable[..., Any], cacher: _ChatCacher) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(create):

        async def acreate(**kwargs: Any) -> Any:
            if kwargs.get("stream"):
                data, key = await asyncio.to_thread(cacher.lookup_data, kwargs)
                if data is not None:
                    return AsyncReplayStream(cacher.replay(data))
                stream = await create(**kwargs)
                if key is None:
                    return stream
                return AsyncRecordingStream(
                    stream,
                    cacher.stream_format.accumulator(),
                    lambda result: cacher.store(kwargs, key, result),
                )
            # Cache work may call an embedding API; keep it off the event loop.
            cached, key = await asyncio.to_thread(cacher.lookup, kwargs)
            if cached is not None:
                return cached
            response = await create(**kwargs)
            if key is not None:
                await asyncio.to_thread(cacher.store, kwargs, key, response)
            return response

        return acreate

    def create_(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            data, key = cacher.lookup_data(kwargs)
            if data is not None:
                return ReplayStream(cacher.replay(data))
            stream = create(**kwargs)
            if key is None:
                return stream
            return RecordingStream(
                stream,
                cacher.stream_format.accumulator(),
                lambda result: cacher.store(kwargs, key, result),
            )
        cached, key = cacher.lookup(kwargs)
        if cached is not None:
            return cached
        response = create(**kwargs)
        if key is not None:
            cacher.store(kwargs, key, response)
        return response

    return create_


class _Proxy:
    """Forwards every attribute to the wrapped object except the ones overridden."""

    def __init__(self, target: Any, **overrides: Any) -> None:
        self._target = target
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def wrap_openai(client: Any, cache: SemanticCache, *, namespace: str = "openai") -> Any:
    """Cache ``client.chat.completions.create`` for an ``OpenAI`` or ``AsyncOpenAI`` client."""
    cacher = _ChatCacher(
        cache,
        namespace,
        usage_fields=("prompt_tokens", "completion_tokens"),
        rebuild=_rehydrator("openai.types.chat", "ChatCompletion"),
        is_complete=_openai_complete,
        stream_format=OPENAI,
    )
    completions = client.chat.completions
    wrapped = _Proxy(completions, create=_cached_create(completions.create, cacher))
    return _Proxy(client, chat=_Proxy(client.chat, completions=wrapped), semcache=cache)


def wrap_anthropic(client: Any, cache: SemanticCache, *, namespace: str = "anthropic") -> Any:
    """Cache ``client.messages.create`` for an ``Anthropic`` or ``AsyncAnthropic`` client."""
    cacher = _ChatCacher(
        cache,
        namespace,
        usage_fields=("input_tokens", "output_tokens"),
        rebuild=_rehydrator("anthropic.types", "Message"),
        is_complete=_anthropic_complete,
        stream_format=ANTHROPIC,
    )
    messages = client.messages
    wrapped = _Proxy(messages, create=_cached_create(messages.create, cacher))
    return _Proxy(client, messages=wrapped, semcache=cache)
