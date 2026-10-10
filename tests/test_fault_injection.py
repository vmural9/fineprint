"""What the service does when a model provider fails, from the model up to the HTTP response.

The cases run `answer_question` and `Retriever.search` inside a `Tracer(RecordingSink())`, so each
asserts the response and also the trace: how many generation spans there are, which one holds the
error, which one is marked as the fallback, and what the question cost. A failed provider call
counts as zero cost, so the cost of a question that fell back is the fallback's alone.

The re-ranker cases use a real `Retriever` whose candidate search is replaced with canned chunks,
so no database is needed.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fineprint.answer import DraftAnswer, answer_question
from fineprint.api import app, get_chat, get_retriever, get_tracer
from fineprint.config import Settings
from fineprint.pricing import Price
from fineprint.providers import factory
from fineprint.providers.base import (
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from fineprint.providers.bedrock_chat import ModelRefusedError
from fineprint.providers.fallback import FallbackChatModel
from fineprint.providers.faults import FaultyReranker, parse_fault
from fineprint.retrieval import RetrievedChunk, Retriever
from fineprint.tracing import TracedChatModel, TracedReranker, Tracer
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    FlakyChatModel,
    RecordedObservation,
    RecordingSink,
)

QUESTION = "How much is the Part B premium in 2026?"


def price(model: str) -> Price:
    """$1 per million input tokens and $2 per million output tokens, so costs work out by hand."""
    return Price(
        model=model,
        input_per_million=1.0,
        output_per_million=2.0,
        per_unit=None,
        unit="tokens",
        source_url="https://prices.example/list.json",
        retrieved_on="2026-10-10",
    )


# 900 input and 120 output tokens at those prices.
ONE_ANSWER = 900 * 1.0e-6 + 120 * 2.0e-6

PRICES = {
    "flaky-chat-model": price("flaky-chat-model"),
    "fake-chat-model": price("fake-chat-model"),
    "fake-reranker": Price(
        model="fake-reranker",
        input_per_million=None,
        output_per_million=None,
        per_unit=1.0,
        unit="search units",
        source_url="https://prices.example/list.json",
        retrieved_on="2026-10-10",
    ),
}

DRAFT = DraftAnswer(
    answer="The standard Part B premium in 2026 is $202.90 a month.",
    found_in_handbook=True,
    citations=[{"chunk_id": 61, "quote": "The standard Part B premium amount in 2026 is $202.90."}],
    confidence="high",
)


def chunk(chunk_id: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        ordinal=chunk_id - 1,
        section=None,
        page_start=23,
        page_end=23,
        text=text,
        score=1.0 / chunk_id,
        rank=chunk_id - 59,
        lexical_rank=chunk_id - 59,
        vector_rank=None,
        fused_rank=chunk_id - 59,
        rerank_score=None,
    )


# In fused order. The fake re-ranker puts the one sharing most words with the question first.
CHUNKS = [
    chunk(60, "Part A covers inpatient hospital stays."),
    chunk(61, "The standard Part B premium amount in 2026 is $202.90."),
    chunk(62, "Medigap policies are sold by private companies."),
]


class CannedRetriever:
    """Returns the one chunk the answer cites."""

    def search(self, question, mode="hybrid", top_k=None):
        return [CHUNKS[1]]


class Stack:
    """A fallback chat model over a flaky primary and a fake fallback, with its own tracer."""

    def __init__(self, primary_script, fallback=None):
        self.sink = RecordingSink()
        self.tracer = Tracer(self.sink)
        self.primary = FlakyChatModel(primary_script)
        self.fallback = fallback if fallback is not None else FakeChatModel(DRAFT)
        self.chat = FallbackChatModel(
            TracedChatModel(self.primary, self.tracer, prices=PRICES),
            TracedChatModel(self.fallback, self.tracer, prices=PRICES, fallback=True),
        )

    def ask(self):
        return answer_question(QUESTION, CannedRetriever(), self.chat, tracer=self.tracer)

    @property
    def generations(self) -> list[RecordedObservation]:
        return [obs for obs in self.sink.observations if obs.kind == "generation"]


def is_marked_fallback(span: RecordedObservation) -> bool:
    return span.fields.get("metadata") == {"fallback": True}


# --- the chat model -----------------------------------------------------------------------


@pytest.mark.parametrize("error_type", [ProviderUnavailableError, ProviderTimeoutError])
def test_an_unavailable_or_slow_primary_is_answered_by_the_fallback(error_type):
    stack = Stack([error_type("the provider is down")])

    response = stack.ask()

    assert response.answer == DRAFT.answer
    assert response.fallback_used is True
    assert response.model == "fake-chat-model"
    first, second = stack.generations
    assert error_type.__name__ in first.fields["error"]
    assert not is_marked_fallback(first)
    assert "error" not in second.fields
    assert is_marked_fallback(second)
    assert second.fields["model"] == "fake-chat-model"
    assert response.cost_usd == pytest.approx(ONE_ANSWER), "the failed attempt counts as zero"
    assert len(stack.sink.traces) == 1, "both attempts are in the one trace"


def test_when_the_primary_works_again_the_fallback_is_not_called():
    stack = Stack([ProviderUnavailableError("blip"), DRAFT])

    first, second = stack.ask(), stack.ask()

    assert (first.fallback_used, second.fallback_used) == (True, False)
    assert second.model == "flaky-chat-model"
    assert len(stack.fallback.calls) == 1, "only the first question reached the fallback"
    second_trace = stack.sink.traces[1]
    (only_attempt,) = [c for c in second_trace.children if c.kind == "generation"]
    assert "error" not in only_attempt.fields and not is_marked_fallback(only_attempt)
    assert second.cost_usd == pytest.approx(ONE_ANSWER)


def test_when_both_fail_the_error_propagates_and_both_attempts_are_traced():
    stack = Stack(
        [ProviderTimeoutError("primary timed out")],
        fallback=FlakyChatModel([ProviderUnavailableError("fallback throttled")]),
    )

    with pytest.raises(ProviderUnavailableError) as raised:
        stack.ask()

    assert isinstance(raised.value.__cause__, ProviderTimeoutError)
    first, second = stack.generations
    assert "ProviderTimeoutError" in first.fields["error"]
    assert "ProviderUnavailableError" in second.fields["error"]
    assert is_marked_fallback(second)
    assert "ProviderUnavailableError" in stack.sink.traces[0].fields["error"]


def test_a_refusal_is_not_sent_to_the_fallback():
    stack = Stack([ModelRefusedError("declined")])

    with pytest.raises(ModelRefusedError):
        stack.ask()

    assert stack.fallback.calls == []
    (only_attempt,) = stack.generations
    assert "ModelRefusedError" in only_attempt.fields["error"]


# --- the API ------------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch, tmp_path) -> Iterator[tuple[TestClient, RecordingSink]]:
    monkeypatch.chdir(tmp_path)
    sink = RecordingSink()
    app.dependency_overrides[get_tracer] = lambda: Tracer(sink)
    yield TestClient(app), sink
    app.dependency_overrides.clear()


def serve(retriever, chat) -> None:
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_chat] = lambda: chat


def test_the_api_answers_503_when_both_models_fail(api):
    client, sink = api
    stack = Stack(
        [ProviderTimeoutError("primary timed out")],
        fallback=FlakyChatModel([ProviderUnavailableError("fallback throttled: slow down")]),
    )
    serve(CannedRetriever(), stack.chat)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "model provider unavailable: fallback throttled: slow down"
    }


def test_the_api_answers_from_the_fallback_with_a_200_and_says_so(api):
    client, _ = api
    stack = Stack([ProviderUnavailableError("down")])
    serve(CannedRetriever(), stack.chat)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["fallback_used"] is True
    assert response.json()["model"] == "fake-chat-model"


def test_a_refusal_is_still_a_502_not_a_503(api):
    client, _ = api
    serve(CannedRetriever(), Stack([ModelRefusedError("declined")]).chat)

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 502


# --- the re-ranker ------------------------------------------------------------------------


class RerankStack:
    """A real `Retriever` over canned candidates, with a re-ranker that may be made to fail."""

    def __init__(self, fault: str = ""):
        self.sink = RecordingSink()
        self.tracer = Tracer(self.sink)
        reranker = FakeReranker()
        spec = parse_fault(fault)
        if spec is not None:
            reranker = FaultyReranker(reranker, spec)
        self.retriever = Retriever(
            pool=None,  # type: ignore[arg-type]
            embedder=FakeEmbedder(),
            settings=Settings(_env_file=None),
            reranker=TracedReranker(reranker, self.tracer, prices=PRICES),
            tracer=self.tracer,
        )
        self.retriever.candidates = lambda question, mode="hybrid": list(CHUNKS)  # type: ignore[method-assign]


def test_a_failing_re_ranker_leaves_the_fused_order_and_the_span_holds_the_error():
    stack = RerankStack("rerank:unavailable")

    with stack.tracer.trace("search") as trace:
        chunks = stack.retriever.search("standard Part B premium amount", top_k=2)

    assert [c.chunk_id for c in chunks] == [60, 61], "the fused order, cut to top_k"
    assert [c.rerank_score for c in chunks] == [None, None]
    assert [c.score for c in chunks] == [CHUNKS[0].score, CHUNKS[1].score]
    (span,) = [obs for obs in stack.sink.observations if obs.kind == "rerank"]
    assert "ProviderUnavailableError" in span.fields["error"]
    assert trace.record.cost_usd == 0.0, "a failed provider call counts as zero"


def test_a_working_re_ranker_reorders_the_candidates():
    stack = RerankStack()

    chunks = stack.retriever.search("standard Part B premium amount", top_k=2)

    assert chunks[0].chunk_id == 61
    assert chunks[0].rerank_score is not None


def test_search_says_whether_the_re_ranker_was_used(api):
    client, _ = api
    body = {"query": "standard Part B premium amount", "top_k": 2}

    serve(RerankStack().retriever, FakeChatModel(DRAFT))
    working = client.post("/search", json=body).json()
    serve(RerankStack("rerank:timeout").retriever, FakeChatModel(DRAFT))
    degraded = client.post("/search", json=body).json()

    assert working["reranked"] is True
    assert degraded["reranked"] is False
    assert [hit["chunk_id"] for hit in degraded["results"]] == [60, 61]
    assert all(hit["rerank_score"] is None for hit in degraded["results"])


def test_a_question_is_still_answered_when_the_re_ranker_fails():
    stack = RerankStack("rerank:unavailable")
    chat = FakeChatModel(DRAFT)

    response = answer_question(QUESTION, stack.retriever, chat, top_k=3, tracer=stack.tracer)

    assert response.answer == DRAFT.answer
    assert response.retrieved_chunk_ids == [60, 61, 62]


# --- the factory --------------------------------------------------------------------------


def settings_with(**changes) -> Settings:
    return Settings(
        _env_file=None,
        llm_model="primary-model",
        fallback_llm_model="fallback-model",
        **changes,
    )


@pytest.fixture
def fake_providers(monkeypatch):
    """Make the factory build fakes, with the model id the settings asked for."""

    def build_chat(settings: Settings):
        model = FakeChatModel(DRAFT)
        model.model = settings.llm_model
        return model

    monkeypatch.setitem(factory.CHAT_MODELS, "bedrock", build_chat)
    monkeypatch.setitem(factory.EMBEDDERS, "bedrock", lambda settings: FakeEmbedder())
    monkeypatch.setitem(factory.RERANKERS, "bedrock", lambda settings: FakeReranker())


def test_the_chat_stack_is_a_fallback_over_two_traced_models(fake_providers):
    tracer = Tracer()

    chat = factory.get_chat_model(settings_with(), tracer)

    assert isinstance(chat, FallbackChatModel)
    assert isinstance(chat.primary, TracedChatModel) and not chat.primary._fallback
    assert isinstance(chat.fallback, TracedChatModel) and chat.fallback._fallback
    assert (chat.primary.model, chat.fallback.model) == ("primary-model", "fallback-model")


def test_an_empty_fallback_model_turns_the_fallback_off(fake_providers):
    chat = factory.get_chat_model(
        Settings(_env_file=None, llm_model="primary-model", fallback_llm_model=""), Tracer()
    )

    assert isinstance(chat, TracedChatModel)


def test_the_fault_switch_makes_the_first_call_fall_back_and_the_second_not(fake_providers):
    sink = RecordingSink()
    tracer = Tracer(sink)
    chat = factory.get_chat_model(settings_with(fault_injection="chat:unavailable:once"), tracer)

    first = answer_question(QUESTION, CannedRetriever(), chat, tracer=tracer)
    second = answer_question(QUESTION, CannedRetriever(), chat, tracer=tracer)

    assert (first.fallback_used, first.model) == (True, "fallback-model")
    assert (second.fallback_used, second.model) == (False, "primary-model")
    first_trace, second_trace = sink.traces
    attempts = [c for c in first_trace.children if c.kind == "generation"]
    assert [is_marked_fallback(a) for a in attempts] == [False, True]
    assert "ProviderUnavailableError" in attempts[0].fields["error"]
    assert "injected unavailable fault on chat" in attempts[0].fields["error"]
    assert len([c for c in second_trace.children if c.kind == "generation"]) == 1


def test_an_injected_fault_is_traced_inside_the_traced_wrapper_without_a_fallback(fake_providers):
    sink = RecordingSink()
    tracer = Tracer(sink)
    chat = factory.get_chat_model(
        Settings(
            _env_file=None,
            llm_model="primary-model",
            fallback_llm_model="",
            fault_injection="chat:timeout",
        ),
        tracer,
    )

    with pytest.raises(ProviderTimeoutError):
        answer_question(QUESTION, CannedRetriever(), chat, tracer=tracer)

    (span,) = [obs for obs in sink.observations if obs.kind == "generation"]
    assert "ProviderTimeoutError" in span.fields["error"]


def test_embedding_and_rerank_faults_are_traced_like_real_failures(fake_providers):
    sink = RecordingSink()
    tracer = Tracer(sink)
    settings = Settings(
        _env_file=None, reranker_provider="bedrock", fault_injection="embedding:timeout"
    )
    embedder = factory.get_embedder(settings, tracer)

    with pytest.raises(ProviderTimeoutError):
        embedder.embed_query("premium")

    (span,) = [obs for obs in sink.observations if obs.kind == "embedding"]
    assert "ProviderTimeoutError" in span.fields["error"]
    reranker = factory.get_reranker(settings, tracer)
    assert isinstance(reranker, TracedReranker)
    assert reranker.rerank("q", ["a", "b"], 2), "a fault aimed at the embedder leaves it alone"


def test_a_malformed_fault_fails_at_construction_and_says_what_is_accepted(fake_providers):
    settings = Settings(_env_file=None, fault_injection="chat:explode")

    for build in (factory.get_chat_model, factory.get_embedder, factory.get_reranker):
        with pytest.raises(ValueError, match="chat, embedding, rerank"):
            build(settings)
