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
* ``stream=True`` is passed through uncached for now.

Responses and chat completions are cached separately (their answers have different
shapes), under ``<namespace>/responses``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse_request", "is_complete", "rebuild"]

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
        from .streaming import _to_namespace

        return _to_namespace(data)
