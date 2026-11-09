"""Drop-in caching wrappers for the OpenAI and Anthropic Python clients.

::

    from anthropic import Anthropic
    from semcache import SemanticCache
    from semcache.integrations import wrap_anthropic

    client = wrap_anthropic(Anthropic(), SemanticCache(path="cache.db"))
    client.messages.create(model="claude-opus-5-5", max_tokens=1024, messages=[...])

How requests are matched:

* The **last user message** is the text compared by meaning.
* Everything else (system prompt, earlier turns, tools, temperature, max tokens, ...)
  is hashed into the cache namespace, so a hit only happens when that context is
  identical. Two people asking the same question inside different conversations
  never share an answer.
* ``stream=True`` requests are passed through uncached.
* Incomplete answers are not stored: Anthropic ``stop_reason`` of ``max_tokens``,
  ``refusal`` or ``pause_turn``, and OpenAI ``finish_reason`` of ``length`` or
  ``content_filter``.

The wrappers are duck-typed: they only touch ``chat.completions.create`` (OpenAI) or
``messages.create`` (Anthropic), so they also work with compatible clients and fakes.
Sync and async clients are both supported.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace
from typing import Any, Callable

from .cache import SemanticCache

__all__ = ["wrap_openai", "wrap_anthropic"]

# Request options that do not change what the model answers.
_IGNORED_PARAMS = frozenset(
    {
        "messages",
        "model",
        "stream",
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
    ) -> None:
        self.cache = cache
        self.namespace = namespace
        self.usage_fields = usage_fields
        self.rebuild = rebuild
        self.is_complete = is_complete

    def key(self, kwargs: dict[str, Any]) -> tuple[str, str] | None:
        """Return ``(prompt, namespace)`` or ``None`` if this request should not be cached."""
        if kwargs.get("stream"):
            return None
        messages = list(kwargs.get("messages") or [])
        if not messages or _field(messages[-1], "role") != "user":
            return None
        prompt = _text_of(_field(messages[-1], "content"))
        if not prompt:
            return None
        context = {k: v for k, v in kwargs.items() if k not in _IGNORED_PARAMS}
        context["history"] = [_to_jsonable(m) for m in messages[:-1]]
        digest = hashlib.sha256(
            json.dumps(context, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return prompt, f"{self.namespace}/{digest}"

    def lookup(self, kwargs: dict[str, Any]) -> tuple[Any, tuple[str, str] | None]:
        key = self.key(kwargs)
        if key is None:
            return None, None
        hit = self.cache.lookup(key[0], model=kwargs.get("model"), namespace=key[1])
        return (self.rebuild(hit.response) if hit else None), key

    def store(self, kwargs: dict[str, Any], key: tuple[str, str], response: Any) -> None:
        data = _to_jsonable(response)
        if not self.is_complete(data):
            return
        usage = _field(response, "usage")
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
            cached, key = cacher.lookup(kwargs)
            if cached is not None:
                return cached
            response = await create(**kwargs)
            if key is not None:
                cacher.store(kwargs, key, response)
            return response

        return acreate

    def create_(**kwargs: Any) -> Any:
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
    )
    messages = client.messages
    wrapped = _Proxy(messages, create=_cached_create(messages.create, cacher))
    return _Proxy(client, messages=wrapped, semcache=cache)
