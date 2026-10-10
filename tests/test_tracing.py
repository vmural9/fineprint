"""Tests for the tracer, the traced model wrappers, and the wiring that uses them.

A `RecordingSink` stands in for Langfuse, so these tests read back exactly what a sink would be
told: which observations opened inside which, and the tokens, units and cost each one reported.
Prices come from a small table of this file's own, so a change to the real price list never
breaks them.
"""

import logging
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import BaseModel

from fineprint.answer import DraftAnswer, answer_question
from fineprint.config import Settings
from fineprint.pricing import Price
from fineprint.providers import factory
from fineprint.providers.base import (
    LLMResult,
    ProviderTimeoutError,
    ToolCall,
    ToolSpec,
    ToolTurn,
)
from fineprint.providers.bedrock_chat import InvalidToolInputError
from fineprint.retrieval import RetrievedChunk, Retriever
from fineprint.tracing import (
    TracedChatModel,
    TracedEmbedder,
    TracedReranker,
    Tracer,
)
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker, RecordingSink


def price(model: str, **fields: Any) -> Price:
    return Price(
        model=model,
        input_per_million=fields.get("input_per_million"),
        output_per_million=fields.get("output_per_million"),
        per_unit=fields.get("per_unit"),
        unit="search unit" if "per_unit" in fields else "tokens",
        source_url="https://prices.example/list.json",
        retrieved_on="2026-10-10",
    )


# Round numbers, so the expected costs below can be checked by hand.
PRICES = {
    "fake-chat-model": price("fake-chat-model", input_per_million=1.0, output_per_million=2.0),
    "test-embed": price("test-embed", input_per_million=0.5),
    "fake-reranker": price("fake-reranker", per_unit=0.01),
}

QUESTION = "How much is the Part B premium in 2026?"

PREMIUM = RetrievedChunk(
    chunk_id=61,
    ordinal=60,
    section="How much does Part B coverage cost?",
    page_start=23,
    page_end=23,
    text="The standard Part B premium amount in 2026 is $202.90.",
    score=0.0325,
    rank=1,
    lexical_rank=1,
    vector_rank=2,
    fused_rank=1,
    rerank_score=None,
)

DRAFT = DraftAnswer(
    answer="The standard Part B premium in 2026 is $202.90 a month.",
    found_in_handbook=True,
    citations=[{"chunk_id": 61, "quote": "The standard Part B premium amount in 2026 is $202.90."}],
    confidence="high",
)


class PricedEmbedder(FakeEmbedder):
    """A fake embedder that names a model, so the traced wrapper can price its tokens."""

    model = "test-embed"


class Verdict(BaseModel):
    ok: bool


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def tracer(sink: RecordingSink) -> Tracer:
    return Tracer(sink)


# --- nesting ------------------------------------------------------------------------------


def test_spans_nest_the_way_the_calls_do(tracer: Tracer, sink: RecordingSink):
    embedder = TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES)
    reranker = TracedReranker(FakeReranker(), tracer, prices=PRICES)
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask", input=QUESTION):
        with tracer.span("retriever", "search"):
            embedder.embed_query(QUESTION)
            reranker.rerank(QUESTION, ["Part B premium", "hearing aids"], top_n=1)
        chat.complete_structured("system", "user", DraftAnswer)

    (root,) = sink.traces
    assert root.name == "ask"
    assert [(child.kind, child.name) for child in root.children] == [
        ("retriever", "search"),
        ("generation", "answer"),
    ]
    retriever = root.children[0]
    assert [(child.kind, child.name) for child in retriever.children] == [
        ("embedding", "embed_query"),
        ("rerank", "rerank"),
    ]
    assert all(child.parent is retriever for child in retriever.children)


def test_a_trace_passes_its_attributes_to_the_sink(tracer: Tracer, sink: RecordingSink):
    with tracer.trace(
        "ask",
        input=QUESTION,
        session_id="run-1",
        tags=["eval", "sections"],
        metadata={"question_id": "q001", "attempt": 2},
    ):
        pass

    (root,) = sink.traces
    assert root.input == QUESTION
    assert root.session_id == "run-1"
    assert root.tags == ["eval", "sections"]
    assert root.metadata == {"question_id": "q001", "attempt": "2"}


def test_the_trace_knows_the_id_the_sink_gave_it():
    with Tracer(RecordingSink(trace_id="abc123")).trace("ask") as trace:
        assert trace.trace_id == "abc123"
    assert trace.record.trace_id == "abc123"


def test_a_tracer_without_a_sink_still_measures():
    tracer = Tracer()
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask") as trace:
        chat.complete_structured("system", "user", DraftAnswer)

    assert trace.trace_id is None
    assert trace.record.trace_id is None
    assert trace.record.cost_usd == pytest.approx(0.00114)
    assert [call.kind for call in trace.record.calls] == ["generation"]


def test_a_trace_started_inside_a_trace_joins_it(tracer: Tracer, sink: RecordingSink):
    with tracer.trace("outer") as outer, tracer.trace("inner") as inner:
        assert inner is outer

    assert [root.name for root in sink.traces] == ["outer"]


def test_a_bare_span_opens_a_trace_of_its_own(tracer: Tracer, sink: RecordingSink):
    with tracer.span("tool", "lookup_cost_figure") as span:
        span.end(output="ok")

    (root,) = sink.traces
    assert root.name == "lookup_cost_figure"
    assert [(child.kind, child.name) for child in root.children] == [("tool", "lookup_cost_figure")]
    assert tracer.active_trace() is None


def test_the_active_trace_is_reachable_only_inside_its_block(tracer: Tracer):
    assert tracer.active_trace() is None
    with tracer.trace("ask") as trace:
        assert tracer.active_trace() is trace
    assert tracer.active_trace() is None


# --- what each wrapper reports --------------------------------------------------------------


def test_a_chat_call_reports_the_prompt_the_model_saw_and_what_it_cost(
    tracer: Tracer, sink: RecordingSink
):
    chat = TracedChatModel(
        FakeChatModel(DRAFT, input_tokens=900, output_tokens=120), tracer, prices=PRICES
    )

    with tracer.trace("ask") as trace:
        result = chat.complete_structured("be exact", "what is the premium?", DraftAnswer)

    assert result.parsed is DRAFT
    (generation,) = sink.observations
    assert generation.kind == "generation"
    assert generation.input == [
        {"role": "system", "content": "be exact"},
        {"role": "user", "content": "what is the premium?"},
    ]
    assert generation.fields["output"] == DRAFT.model_dump(mode="json")
    assert generation.fields["model"] == "fake-chat-model"
    assert generation.fields["input_tokens"] == 900
    assert generation.fields["output_tokens"] == 120
    cost = generation.fields["cost"]
    assert (cost.input_usd, cost.output_usd) == (pytest.approx(0.0009), pytest.approx(0.00024))
    (call,) = trace.record.calls
    assert (call.kind, call.name, call.model) == ("generation", "answer", "fake-chat-model")
    assert (call.input_tokens, call.output_tokens) == (900, 120)
    assert call.cost_usd == pytest.approx(0.00114)
    assert call.error is None
    assert call.fallback is False


def test_a_chat_span_takes_the_name_it_was_given(tracer: Tracer, sink: RecordingSink):
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, name="fallback", prices=PRICES)

    chat.complete_structured("system", "user", DraftAnswer)

    assert [observation.name for observation in sink.observations] == ["fallback"]


def test_a_fallback_answer_is_flagged_on_its_span(tracer: Tracer, sink: RecordingSink):
    class FallbackChat(FakeChatModel):
        def complete_structured(self, system, user, schema):
            result = super().complete_structured(system, user, schema)
            result.fallback_used = True
            return result

    chat = TracedChatModel(FallbackChat(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask") as trace:
        chat.complete_structured("system", "user", DraftAnswer)

    (call,) = trace.record.calls
    assert call.fallback is True
    assert sink.observations[0].fields["metadata"] == {"fallback": True}


def test_an_embedding_call_is_priced_from_the_tokens_titan_reports(
    tracer: Tracer, sink: RecordingSink
):
    embedder = TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES)

    with tracer.trace("search") as trace:
        vector = embedder.embed_query(QUESTION)

    assert len(vector) == 1024
    (embedding,) = sink.observations
    assert embedding.kind == "embedding"
    assert embedding.model == "test-embed"
    assert embedding.input == QUESTION
    assert embedding.fields["input_tokens"] == 9  # FakeEmbedder counts words
    assert embedding.fields["cost"].total_usd == pytest.approx(9 * 0.5 / 1_000_000)
    assert trace.record.calls[0].cost_usd == pytest.approx(9 * 0.5 / 1_000_000)


def test_embedding_documents_and_embed_each_open_one_span(tracer: Tracer, sink: RecordingSink):
    embedder = TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES)

    with tracer.trace("ingest"):
        embedder.embed_documents(["one two", "three"])
        result = embedder.embed(["four"])

    assert [observation.name for observation in sink.observations] == [
        "embed_documents",
        "embed",
    ]
    assert result.input_tokens == 1
    assert embedder.dimension == 1024


def test_an_embedding_without_a_token_count_has_no_cost(tracer: Tracer):
    class Silent(PricedEmbedder):
        def embed(self, texts):
            result = super().embed(texts)
            return type(result)(vectors=result.vectors, input_tokens=None, latency_ms=0.0)

    embedder = TracedEmbedder(Silent(), tracer, prices=PRICES)

    with tracer.trace("search") as trace:
        embedder.embed_query(QUESTION)

    (call,) = trace.record.calls
    assert call.input_tokens is None
    assert call.cost_usd is None
    assert trace.record.cost_usd is None


def test_a_rerank_call_is_billed_in_search_units(tracer: Tracer, sink: RecordingSink):
    reranker = TracedReranker(FakeReranker(), tracer, prices=PRICES)
    documents = [f"passage {number}" for number in range(150)]

    with tracer.trace("search") as trace:
        results = reranker.rerank("passage 7", documents, top_n=3)

    assert len(results) == 3
    assert reranker.model == "fake-reranker"
    (rerank,) = sink.observations
    assert rerank.kind == "rerank"
    assert rerank.fields["units"] == 2  # 150 passages, 100 to a unit
    assert rerank.fields["cost"].total_usd == pytest.approx(0.02)
    (call,) = trace.record.calls
    assert (call.kind, call.units, call.model) == ("rerank", 2, "fake-reranker")
    assert call.cost_usd == pytest.approx(0.02)


class ScriptedTools:
    """A model that offers `complete_with_tools` and nothing else."""

    model = "fake-chat-model"

    def complete_with_tools(self, system, messages, tools, *, force_tool=None):
        call = ToolCall(id="t1", name="record_answer", arguments={"answer": "x"})
        return ToolTurn(
            tool_calls=[call],
            content=[],
            model=self.model,
            input_tokens=500,
            output_tokens=50,
            latency_ms=3.0,
            stop_reason="tool_use",
        )


def test_a_tool_turn_is_one_generation_span_with_the_calls_as_output(
    tracer: Tracer, sink: RecordingSink
):
    chat = TracedChatModel(ScriptedTools(), tracer, name="turn 1", prices=PRICES)
    tools = [ToolSpec(name="record_answer", description="Record.", input_schema={})]
    messages = [{"role": "user", "content": QUESTION}]

    with tracer.trace("agent") as trace:
        turn = chat.complete_with_tools("system", messages, tools, force_tool="record_answer")

    assert turn.input_tokens == 500
    (generation,) = sink.observations
    assert (generation.kind, generation.name) == ("generation", "turn 1")
    assert generation.input[0] == {"role": "system", "content": "system"}
    assert generation.input[1:] == messages
    assert generation.fields["output"] == [{"name": "record_answer", "arguments": {"answer": "x"}}]
    assert trace.record.calls[0].cost_usd == pytest.approx(500 * 1e-6 + 50 * 2e-6)


def test_a_model_without_tools_fails_the_way_it_would_untraced(tracer: Tracer):
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with pytest.raises(AttributeError, match="complete_with_tools"):
        chat.complete_with_tools("system", [], [])


# --- failures -------------------------------------------------------------------------------


class ExplodingChat:
    model = "fake-chat-model"

    def complete_structured(self, system, user, schema):
        raise RuntimeError("the model is down")


def test_a_failed_call_records_the_error_and_raises_it_again(tracer: Tracer, sink: RecordingSink):
    chat = TracedChatModel(ExplodingChat(), tracer, prices=PRICES)

    with pytest.raises(RuntimeError, match="the model is down"), tracer.trace("ask") as trace:
        chat.complete_structured("system", "user", DraftAnswer)

    (generation,) = sink.observations
    assert generation.fields["error"] == "RuntimeError: the model is down"
    (call,) = trace.record.calls
    assert call.error == "RuntimeError: the model is down"
    assert (call.input_tokens, call.cost_usd) == (None, None)
    assert sink.traces[0].fields["error"] == "RuntimeError: the model is down"


class FailingChat:
    model = "fake-chat-model"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete_structured(self, system, user, schema):
        raise self.error


def trace_with_embedding_then_failure(tracer: Tracer, error: Exception):
    """Embed the question (cost 9 tokens at $0.5 per million), then fail the chat call."""
    embedder = TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES)
    chat = TracedChatModel(FailingChat(error), tracer, prices=PRICES)
    with pytest.raises(type(error)), tracer.trace("ask") as trace:
        embedder.embed_query(QUESTION)
        chat.complete_structured("system", "user", DraftAnswer)
    return trace.record


def test_a_failure_that_carries_usage_is_priced_from_it(tracer: Tracer, sink: RecordingSink):
    error = InvalidToolInputError("did not validate")
    error.input_tokens, error.output_tokens = 1800, 220

    record = trace_with_embedding_then_failure(tracer, error)

    chat_call = record.calls[-1]
    assert chat_call.error == "InvalidToolInputError: did not validate"
    assert (chat_call.input_tokens, chat_call.output_tokens) == (1800, 220)
    assert chat_call.cost_usd == pytest.approx(1800e-6 + 220 * 2e-6)
    assert record.cost_usd == pytest.approx(9 * 0.5e-6 + 1800e-6 + 220 * 2e-6)
    generation = sink.observations[-1]
    assert generation.fields["error"] == "InvalidToolInputError: did not validate"
    assert generation.fields["cost"].total_usd == pytest.approx(1800e-6 + 220 * 2e-6)


def test_a_provider_failure_counts_as_zero_cost(tracer: Tracer):
    # The provider never produced a reply, so nothing knowable was billed.
    record = trace_with_embedding_then_failure(tracer, ProviderTimeoutError("timed out"))

    chat_call = record.calls[-1]
    assert chat_call.error == "ProviderTimeoutError: timed out"
    assert chat_call.cost_usd == 0.0
    assert record.cost_usd == pytest.approx(9 * 0.5e-6)


def test_any_other_failure_without_usage_makes_the_trace_cost_unknown(tracer: Tracer):
    record = trace_with_embedding_then_failure(tracer, RuntimeError("something odd"))

    assert record.calls[-1].cost_usd is None
    assert record.cost_usd is None


def test_a_span_left_open_by_an_exception_is_still_recorded(tracer: Tracer):
    with (
        pytest.raises(ValueError),
        tracer.trace("search") as trace,
        tracer.span("tool", "lookup"),
    ):
        raise ValueError("bad figure")

    (call,) = trace.record.calls
    assert (call.kind, call.name, call.error) == ("tool", "lookup", "ValueError: bad figure")


# --- the trace record -----------------------------------------------------------------------


def test_the_trace_record_sums_the_calls_that_have_a_cost(tracer: Tracer):
    embedder = TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES)
    reranker = TracedReranker(FakeReranker(), tracer, prices=PRICES)
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask") as trace:
        with tracer.span("retriever", "search"):
            embedder.embed_query(QUESTION)
            reranker.rerank(QUESTION, ["a", "b"], top_n=1)
        chat.complete_structured("system", "user", DraftAnswer)

    record = trace.record
    assert record.name == "ask"
    assert [call.kind for call in record.calls] == [
        "embedding",
        "rerank",
        "retriever",
        "generation",
    ]
    assert record.cost_usd == pytest.approx(9 * 0.5 / 1_000_000 + 0.01 + 0.00114)
    assert record.latency_ms >= max(call.latency_ms for call in record.calls)


def test_a_price_the_table_lacks_makes_the_trace_cost_unknown_not_zero(tracer: Tracer):
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices={})
    embedder = TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES)

    with tracer.trace("ask") as trace:
        embedder.embed_query(QUESTION)
        chat.complete_structured("system", "user", DraftAnswer)

    generation = next(call for call in trace.record.calls if call.kind == "generation")
    assert generation.cost_usd is None
    assert generation.input_tokens == 900
    assert trace.record.cost_usd is None


def test_a_trace_with_nothing_to_price_costs_nothing(tracer: Tracer):
    with tracer.trace("search") as trace, tracer.span("retriever", "search") as span:
        span.end(output=[])

    assert trace.record.cost_usd == 0.0


def test_the_record_is_readable_before_the_trace_ends(tracer: Tracer):
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask") as trace:
        chat.complete_structured("system", "user", DraftAnswer)
        assert trace.record.cost_usd == pytest.approx(0.00114)


def test_flush_and_shutdown_reach_the_sink(tracer: Tracer, sink: RecordingSink):
    tracer.flush()
    tracer.shutdown()

    assert (sink.flushes, sink.shutdowns) == (1, 1)


def test_a_tracer_without_a_sink_flushes_and_shuts_down_quietly():
    tracer = Tracer()
    tracer.flush()
    tracer.shutdown()
    assert tracer.trace_url("abc") is None


def test_the_trace_url_comes_from_the_sink(tracer: Tracer):
    assert tracer.trace_url("abc") == "https://tracing.example/traces/abc"


# --- the retriever span ---------------------------------------------------------------------


class CannedRetriever(Retriever):
    """A retriever whose pool is never touched: it embeds the question and returns one chunk."""

    def candidates(self, question, mode="hybrid"):
        self.embedder.embed_query(question)
        return [PREMIUM]


def test_a_search_opens_a_retriever_span_around_its_embedding_and_rerank(
    tracer: Tracer, sink: RecordingSink
):
    settings = Settings(chunk_set="sections", _env_file=None)
    retriever = CannedRetriever(
        None,
        TracedEmbedder(PricedEmbedder(), tracer, prices=PRICES),
        settings,
        reranker=TracedReranker(FakeReranker(), tracer, prices=PRICES),
        tracer=tracer,
    )

    with tracer.trace("search"):
        chunks = retriever.search(QUESTION, mode="hybrid", top_k=3)

    assert [chunk.chunk_id for chunk in chunks] == [61]
    (root,) = sink.traces
    (search,) = root.children
    assert (search.kind, search.name) == ("retriever", "search")
    assert search.input == {
        "question": QUESTION,
        "mode": "hybrid",
        "top_k": 3,
        "chunk_set": "sections",
    }
    assert search.fields["output"] == [{"chunk_id": 61, "pages": "23", "rank": 1, "score": 7.0}]
    assert [child.kind for child in search.children] == ["embedding", "rerank"]


def test_a_search_without_a_tracer_opens_no_span(sink: RecordingSink):
    settings = Settings(_env_file=None)
    retriever = CannedRetriever(None, PricedEmbedder(), settings)

    assert [chunk.chunk_id for chunk in retriever.search(QUESTION)] == [61]
    assert sink.observations == []


# --- answer_question ------------------------------------------------------------------------


class ListRetriever:
    def search(self, question, mode="hybrid", top_k=None):
        return [PREMIUM]


def test_answering_inside_a_tracer_fills_in_the_trace_id_and_cost(
    tracer: Tracer, sink: RecordingSink
):
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    response = answer_question(QUESTION, ListRetriever(), chat, tracer=tracer)

    assert response.trace_id == "trace-1"
    assert response.cost_usd == pytest.approx(0.00114)
    assert response.fallback_used is False
    (root,) = sink.traces
    assert root.name == "ask"
    assert root.input == QUESTION
    assert root.fields["output"] == DRAFT.answer


def test_answering_inside_an_open_trace_joins_it(tracer: Tracer, sink: RecordingSink):
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask", tags=["api"]):
        response = answer_question(QUESTION, ListRetriever(), chat, tracer=tracer)

    assert [root.tags for root in sink.traces] == [["api"]]
    assert response.trace_id == "trace-1"
    assert response.cost_usd == pytest.approx(0.00114)


def test_answering_without_a_tracer_leaves_the_trace_fields_empty():
    response = answer_question(QUESTION, ListRetriever(), FakeChatModel(DRAFT))

    assert (response.trace_id, response.cost_usd, response.fallback_used) == (None, None, False)


def test_a_fallback_answer_says_so_on_the_response(tracer: Tracer):
    class FallbackChat(FakeChatModel):
        def complete_structured(self, system, user, schema):
            result = super().complete_structured(system, user, schema)
            return LLMResult(
                parsed=result.parsed,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                fallback_used=True,
            )

    response = answer_question(QUESTION, ListRetriever(), FallbackChat(DRAFT), tracer=tracer)

    assert response.fallback_used is True


# --- the factory ----------------------------------------------------------------------------


def test_get_tracer_without_langfuse_has_no_sink():
    tracer = factory.get_tracer(Settings(tracing_provider="none", _env_file=None))

    assert tracer.sink is None


def test_get_tracer_with_langfuse_builds_a_sink_from_the_three_values(monkeypatch):
    built: list[tuple[str, str, str]] = []

    class StubSink(RecordingSink):
        def __init__(self, public_key, secret_key, base_url):
            super().__init__()
            built.append((public_key, secret_key, base_url))

    monkeypatch.setattr("fineprint.langfuse_sink.LangfuseSink", StubSink)
    settings = Settings(
        tracing_provider="langfuse",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url="https://langfuse.example",
        _env_file=None,
    )

    tracer = factory.get_tracer(settings)

    assert built == [("pk-test", "sk-test", "https://langfuse.example")]
    assert isinstance(tracer.sink, StubSink)


def test_the_builders_wrap_in_the_traced_classes_only_when_given_a_tracer(monkeypatch):
    settings = Settings(reranker_provider="bedrock", _env_file=None)
    tracer = Tracer()

    plain = (
        factory.get_embedder(settings),
        factory.get_chat_model(settings),
        factory.get_reranker(settings),
    )
    traced = (
        factory.get_embedder(settings, tracer),
        factory.get_chat_model(settings, tracer),
        factory.get_reranker(settings, tracer),
    )

    assert not any(
        isinstance(model, TracedEmbedder | TracedChatModel | TracedReranker) for model in plain
    )
    assert [type(model) for model in traced] == [TracedEmbedder, TracedChatModel, TracedReranker]


def test_no_reranker_stays_none_even_with_a_tracer():
    assert (
        factory.get_reranker(Settings(reranker_provider="none", _env_file=None), Tracer()) is None
    )


# --- a sink that fails must not break a request -----------------------------------------------


class BrokenSink:
    """A sink whose every method raises, as a Langfuse outage or a bug in the SDK might."""

    def __init__(self) -> None:
        self.calls = 0

    def _fail(self, *args, **kwargs):
        self.calls += 1
        raise ConnectionError("the tracing service is down")

    @contextmanager
    def trace(self, *args, **kwargs):
        self._fail()
        yield

    @contextmanager
    def observation(self, *args, **kwargs):
        self._fail()
        yield

    current_trace_id = trace_url = flush = shutdown = _fail


def test_a_sink_that_always_fails_changes_nothing_about_the_answer(caplog):
    sink = BrokenSink()
    tracer = Tracer(sink)
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with caplog.at_level(logging.WARNING, logger="fineprint.tracing"):
        response = answer_question(QUESTION, ListRetriever(), chat, tracer=tracer)
        tracer.flush()
        tracer.shutdown()
        url = tracer.trace_url("abc")

    assert response.answer == DRAFT.answer
    assert response.trace_id is None
    assert response.cost_usd == pytest.approx(0.00114)
    assert url is None
    assert sink.calls > 0
    # Each kind of failure is logged once, not once per call.
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == len(set(messages))
    assert messages and all("the tracing service is down" in message for message in messages)


class SinkFailingOnExit(RecordingSink):
    """Opens fine and raises while closing, the way a failed flush at the end of a span would."""

    @contextmanager
    def observation(self, *args, **kwargs):
        with super().observation(*args, **kwargs) as observation:
            yield observation
        raise ConnectionError("could not send the span")


def test_a_sink_that_fails_on_exit_does_not_hide_the_errors_of_the_body():
    tracer = Tracer(SinkFailingOnExit())

    with (
        pytest.raises(ValueError, match="bad figure"),
        tracer.trace("search") as trace,
        tracer.span("tool", "lookup"),
    ):
        raise ValueError("bad figure")

    assert trace.record.calls[0].error == "ValueError: bad figure"


def test_a_sink_that_fails_on_exit_does_not_fail_a_call_that_worked():
    tracer = Tracer(SinkFailingOnExit())
    chat = TracedChatModel(FakeChatModel(DRAFT), tracer, prices=PRICES)

    with tracer.trace("ask") as trace:
        chat.complete_structured("system", "user", DraftAnswer)

    assert trace.record.cost_usd == pytest.approx(0.00114)
