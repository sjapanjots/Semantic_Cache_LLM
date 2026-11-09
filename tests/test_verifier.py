import threading
from types import SimpleNamespace

import pytest

from semcache import LLMResult, SemanticCache
from semcache.verifier import (
    SYSTEM_PROMPT,
    JudgeError,
    JudgeReply,
    LLMVerifier,
    anthropic_judge,
    openai_judge,
)

Q = "How do I reverse a list in Python?"
PARA = "python: how to reverse a list"


class FakeJudge:
    """Says YES unless the new question mentions a word from `no_words`."""

    def __init__(self, reply=None, no_words=()):
        self.calls = []
        self.reply = reply
        self.no_words = no_words

    def __call__(self, system, user):
        self.calls.append((system, user))
        if self.reply is not None:
            return self.reply
        new = user.split("<new_question>")[1]
        return "NO" if any(w in new for w in self.no_words) else "YES"


def make(judge, **kwargs):
    verifier = LLMVerifier(judge, **kwargs)
    cache = SemanticCache(verifier=verifier, threshold=0.5)
    return cache, verifier


def test_approved_hit_is_served():
    cache, verifier = make(FakeJudge())
    cache.set(Q, "list[::-1]")
    assert cache.get(PARA) == "list[::-1]"
    assert verifier.stats.approved == 1 and cache.stats.hits == 1


def test_rejected_hit_becomes_miss():
    cache, verifier = make(FakeJudge(no_words=("sort",)))
    cache.set(Q, "list[::-1]")
    assert cache.get("How do I sort a list in Python?") is None
    s = cache.stats
    assert (s.hits, s.misses) == (0, 1)
    assert verifier.stats.rejected == 1 and verifier.stats.rejection_rate == 1.0


def test_exact_hits_are_not_checked():
    judge = FakeJudge(reply="NO")
    cache, verifier = make(judge)
    cache.set(Q, "x")
    assert cache.get(Q.lower()) == "x"
    assert judge.calls == [] and verifier.stats.checks == 0


def test_verify_below_skips_close_matches():
    judge = FakeJudge(reply="NO")
    cache, verifier = make(judge, verify_below=0.8)
    cache.set(Q, "x")
    hit = cache.lookup(PARA)  # similarity ~0.89 with the default embedder
    assert hit is not None and hit.similarity >= 0.8
    assert judge.calls == [] and verifier.stats.skipped == 1


def test_decisions_are_remembered():
    judge = FakeJudge()
    cache, verifier = make(judge)
    cache.set(Q, "x")
    for _ in range(3):
        assert cache.get(PARA) == "x"
    assert len(judge.calls) == 1
    s = verifier.stats
    assert (s.checks, s.remembered, s.approved) == (3, 2, 3)


def test_remember_limit():
    judge = FakeJudge()
    cache, _ = make(judge, remember=1)
    cache.set(Q, "x")
    cache.get(PARA)
    cache.get("python how to reverse a list please")
    cache.get(PARA)  # evicted by the second question, asked again
    assert len(judge.calls) == 3


@pytest.mark.parametrize(
    "reply,expected",
    [("YES", True), ("yes.", True), (" **Yes**", True), ("NO", False), ("No, it differs", False)],
)
def test_verdict_parsing(reply, expected):
    cache, _ = make(FakeJudge(reply=reply))
    cache.set(Q, "x")
    assert (cache.get(PARA) is not None) is expected


def test_unclear_reply_and_errors_follow_on_error():
    cache, verifier = make(FakeJudge(reply="Maybe"))
    cache.set(Q, "x")
    assert cache.get(PARA) is None
    assert verifier.stats.errors == 1

    def broken(system, user):
        raise TimeoutError("judge down")

    cache, verifier = make(broken, on_error="accept")
    cache.set(Q, "x")
    assert cache.get(PARA) == "x" and verifier.stats.errors == 1

    cache, _ = make(broken, on_error="raise")
    cache.set(Q, "x")
    with pytest.raises(TimeoutError):
        cache.get(PARA)
    cache, _ = make(FakeJudge(reply="Perhaps"), on_error="raise")
    cache.set(Q, "x")
    with pytest.raises(JudgeError):
        cache.get(PARA)


def test_prompt_contents_and_answer_extraction():
    judge = FakeJudge()
    cache, _ = make(judge, max_answer_chars=10)
    anthropic_json = {"content": [{"type": "text", "text": "Use slicing: list[::-1] always"}]}
    cache.set(Q, anthropic_json)
    cache.get(PARA)
    system, user = judge.calls[0]
    assert system == SYSTEM_PROMPT
    assert f"<earlier_question>\n{Q}\n</earlier_question>" in user
    assert "<saved_answer>\nUse slicin [...]\n</saved_answer>" in user
    assert f"<new_question>\n{PARA}\n</new_question>" in user

    judge2 = FakeJudge()
    cache2, _ = make(judge2, include_answer=False)
    cache2.set(Q, {"choices": [{"message": {"content": "secret"}}]})
    cache2.get(PARA)
    assert "saved_answer" not in judge2.calls[0][1]


def test_openai_and_other_answer_shapes():
    v = LLMVerifier(FakeJudge())
    entry = SimpleNamespace(prompt="q", response={"choices": [{"message": {"content": "hi"}}]})
    assert "<saved_answer>\nhi\n</saved_answer>" in v.build_prompt("q2", entry)
    entry.response = {"text": "plain"}
    assert "plain" in v.build_prompt("q2", entry)
    entry.response = [1, 2]
    assert "[1, 2]" in v.build_prompt("q2", entry)


def test_judge_usage_is_priced():
    reply = JudgeReply(
        "YES", input_tokens=1_000_000, output_tokens=1_000_000, model="claude-haiku-5-5"
    )
    cache, verifier = make(FakeJudge(reply=reply))
    cache.set(Q, "x")
    cache.get(PARA)
    s = verifier.stats
    assert s.judge_input_tokens == 1_000_000
    assert s.judge_cost == pytest.approx(0.10 + 0.50)
    assert s.to_dict()["rejection_rate"] == 0.0


def test_works_with_get_or_call():
    cache, _ = make(FakeJudge(no_words=("sort",)))
    calls = []

    def llm(q):
        calls.append(q)
        return LLMResult(f"answer to {q}", cost=0.01)

    cache.get_or_call(Q, lambda: llm(Q))
    assert cache.get_or_call(PARA, lambda: llm(PARA)) == f"answer to {Q}"
    sort_q = "How do I sort a list in Python?"
    assert cache.get_or_call(sort_q, lambda: llm(sort_q)) == f"answer to {sort_q}"
    assert calls == [Q, sort_q]


def test_verifier_runs_without_holding_the_cache_lock():
    started, release = threading.Event(), threading.Event()

    def slow_judge(system, user):
        started.set()
        release.wait(5)
        return "YES"

    cache, _ = make(slow_judge)
    cache.set(Q, "x")
    t = threading.Thread(target=cache.get, args=(PARA,))
    t.start()
    assert started.wait(5)
    other = threading.Thread(target=cache.set, args=("unrelated question", "y"))
    other.start()
    other.join(2)
    assert not other.is_alive(), "set() blocked while the verifier was running"
    release.set()
    t.join(5)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        LLMVerifier(FakeJudge(), on_error="ignore")
    with pytest.raises(ValueError):
        LLMVerifier(FakeJudge(), verify_below=0)


def test_anthropic_judge_with_real_sdk_types():
    types = pytest.importorskip("anthropic.types")
    calls = []

    def message(stop_reason, text):
        return types.Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": text}] if text else [],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 120, "output_tokens": 3},
            }
        )

    replies = iter([message("end_turn", "YES"), message("refusal", "")])

    def create(**kwargs):
        calls.append(kwargs)
        return next(replies)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    judge = anthropic_judge(client)
    reply = judge("sys", "user")
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("YES", 120, 3)
    assert calls[0]["model"] == "claude-haiku-5-5"
    assert calls[0]["output_config"] == {"effort": "low"}
    assert calls[0]["system"] == "sys"
    assert calls[0]["messages"] == [{"role": "user", "content": "user"}]
    with pytest.raises(JudgeError):
        judge("sys", "user")

    # effort=None omits output_config, for models without an effort setting.
    no_effort_calls = []

    def create_no_effort(**kwargs):
        no_effort_calls.append(kwargs)
        return message("end_turn", "NO")

    no_effort = SimpleNamespace(messages=SimpleNamespace(create=create_no_effort))
    assert anthropic_judge(no_effort, effort=None)("s", "u").text == "NO"
    assert "output_config" not in no_effort_calls[0]


def test_openai_judge():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="NO"))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=1),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reply = openai_judge(client, "gpt-test")("sys", "user")
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.model) == (
        "NO",
        50,
        1,
        "gpt-test",
    )
    assert calls[0]["messages"][0] == {"role": "system", "content": "sys"}
