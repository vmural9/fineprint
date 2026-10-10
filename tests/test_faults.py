"""`parse_fault` and the three wrappers that fail on demand."""

import pytest

from fineprint.answer import DraftAnswer
from fineprint.providers.base import ProviderTimeoutError, ProviderUnavailableError
from fineprint.providers.faults import (
    FaultSpec,
    FaultyChatModel,
    FaultyEmbedder,
    FaultyReranker,
    parse_fault,
)
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker, FlakyChatModel

DRAFT = DraftAnswer(answer="A.", found_in_handbook=True, citations=[], confidence="high")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("chat:timeout", FaultSpec("chat", "timeout", False)),
        ("chat:unavailable:once", FaultSpec("chat", "unavailable", True)),
        ("embedding:timeout", FaultSpec("embedding", "timeout", False)),
        ("rerank:unavailable", FaultSpec("rerank", "unavailable", False)),
        ("  rerank:timeout:once ", FaultSpec("rerank", "timeout", True)),
    ],
)
def test_parse_fault_reads_target_fault_and_once(text, expected):
    assert parse_fault(text) == expected


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_value_means_no_faults(text):
    assert parse_fault(text) is None


@pytest.mark.parametrize(
    "text",
    ["chat", "chat:", "chat:explode", "llm:timeout", "chat:timeout:twice", "chat:timeout:once:x"],
)
def test_anything_else_fails_with_a_message_saying_what_is_accepted(text):
    with pytest.raises(ValueError) as raised:
        parse_fault(text)

    message = str(raised.value)
    assert repr(text.strip()) in message
    assert "chat, embedding, rerank" in message
    assert "timeout, unavailable" in message
    assert "chat:unavailable:once" in message


def test_each_fault_raises_its_typed_error_before_the_inner_model_is_called():
    inner = FakeChatModel(DRAFT)

    with pytest.raises(ProviderTimeoutError, match="injected timeout fault on chat"):
        FaultyChatModel(inner, FaultSpec("chat", "timeout", False)).complete_structured(
            "s", "u", DraftAnswer
        )
    with pytest.raises(ProviderUnavailableError):
        FaultyChatModel(inner, FaultSpec("chat", "unavailable", False)).complete_structured(
            "s", "u", DraftAnswer
        )

    assert inner.calls == []


def test_without_once_every_call_fails():
    chat = FaultyChatModel(FakeChatModel(DRAFT), FaultSpec("chat", "unavailable", False))

    for _ in range(3):
        with pytest.raises(ProviderUnavailableError):
            chat.complete_structured("s", "u", DraftAnswer)


def test_with_once_only_the_first_call_fails_and_later_calls_reach_the_inner_model():
    inner = FakeChatModel(DRAFT)
    chat = FaultyChatModel(inner, FaultSpec("chat", "unavailable", True))

    with pytest.raises(ProviderUnavailableError):
        chat.complete_structured("s", "u", DraftAnswer)
    result = chat.complete_structured("s", "u", DraftAnswer)

    assert result.parsed == DRAFT
    assert len(inner.calls) == 1


def test_the_chat_wrapper_also_fails_tool_calls_and_keeps_the_model_id():
    inner = FlakyChatModel([])
    chat = FaultyChatModel(inner, FaultSpec("chat", "timeout", False))

    with pytest.raises(ProviderTimeoutError):
        chat.complete_with_tools("s", [], [])

    assert chat.model == "flaky-chat-model"
    assert inner.calls == []


def test_the_embedder_wrapper_fails_every_embedding_method_and_passes_through_otherwise():
    inner = FakeEmbedder()
    once = FaultyEmbedder(inner, FaultSpec("embedding", "timeout", True))

    with pytest.raises(ProviderTimeoutError):
        once.embed_query("premium")
    assert once.dimension == 1024
    assert once.embed_query("premium") == inner.embed_query("premium")
    assert once.embed_documents(["a", "b"]) == inner.embed_documents(["a", "b"])
    assert once.embed(["a b c"]).input_tokens == 3

    always = FaultyEmbedder(inner, FaultSpec("embedding", "unavailable", False))
    for call in (
        lambda: always.embed(["a"]),
        lambda: always.embed_documents(["a"]),
        lambda: always.embed_query("a"),
    ):
        with pytest.raises(ProviderUnavailableError):
            call()


def test_the_reranker_wrapper_fails_then_recovers_with_once():
    reranker = FaultyReranker(FakeReranker(), FaultSpec("rerank", "unavailable", True))

    with pytest.raises(ProviderUnavailableError):
        reranker.rerank("part b premium", ["part b premium", "other"], 2)
    results = reranker.rerank("part b premium", ["other", "part b premium"], 2)

    assert [result.index for result in results] == [1, 0]
    assert reranker.model == "fake-reranker"
