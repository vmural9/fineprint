"""Do concurrent requests keep their own traces, and their spans under the right parent?

`RecordingSink` keeps its own stack, so it cannot show a mistake in how the real Langfuse SDK
carries the current span from one call to the next (it does so with OpenTelemetry context
variables). This builds a real `Langfuse` client that exports to memory, runs twelve `/ask`
requests at once through the app, and reads the finished spans back. This file and
`langfuse_sink.py` are the only places that import `langfuse`. No network call is made: the
client has no endpoint to reach, and its exporter is the in-memory one.
"""

import asyncio
import random
import time

import httpx
import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fineprint.api import app, get_chat, get_pool, get_retriever, get_settings, get_tracer
from fineprint.config import Settings
from fineprint.langfuse_sink import LangfuseSink
from fineprint.tracing import TracedChatModel, TracedEmbedder, TracedReranker, Tracer
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker
from tests.test_api import DRAFT, FAKE_PRICES, PREMIUM, FakePool

REQUESTS = 12


class SlowChat(FakeChatModel):
    """Takes a random few milliseconds, so the requests really interleave."""

    def complete_structured(self, system, user, schema):
        time.sleep(random.uniform(0.01, 0.05))
        return super().complete_structured(system, user, schema)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    client = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost",
        span_exporter=exporter,
    )
    return Tracer(LangfuseSink("pk-test", "sk-test", "http://localhost", client=client))


def test_twelve_concurrent_asks_get_twelve_traces_with_every_span_under_its_parent(
    tracer: Tracer, exporter: InMemorySpanExporter
):
    embedder = TracedEmbedder(FakeEmbedder(), tracer)
    reranker = TracedReranker(FakeReranker(), tracer)

    class Retriever:
        def search(self, question, mode="hybrid", top_k=None):
            with tracer.span("retriever", "search", input=question) as span:
                embedder.embed_query(question)
                time.sleep(random.uniform(0.01, 0.05))
                reranker.rerank(question, ["a b", "c"], 2)
                span.end(output=[PREMIUM.chunk_id])
            return [PREMIUM]

    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    app.dependency_overrides[get_pool] = lambda: FakePool()
    app.dependency_overrides[get_retriever] = lambda: Retriever()
    app.dependency_overrides[get_tracer] = lambda: tracer
    app.dependency_overrides[get_chat] = lambda: TracedChatModel(
        SlowChat(DRAFT), tracer, prices=FAKE_PRICES
    )

    async def ask_all() -> list[dict]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/ask", json={"question": f"Question {number}: the Part B premium?"}
                    )
                    for number in range(REQUESTS)
                ]
            )
        return [response.json() for response in responses]

    try:
        bodies = asyncio.run(ask_all())
    finally:
        app.dependency_overrides.clear()
    tracer.flush()

    trace_ids = [body["trace_id"] for body in bodies]
    assert all(trace_ids)
    assert len(set(trace_ids)) == REQUESTS

    spans = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == REQUESTS
    assert {format(root.context.trace_id, "032x") for root in roots} == set(trace_ids)
    assert len(spans) == REQUESTS * 5  # ask, search, embed_query, rerank, answer

    expected_parent = {
        "search": "ask",
        "answer": "ask",
        "embed_query": "search",
        "rerank": "search",
    }
    for span in spans:
        if span.parent is None:
            assert span.name == "ask"
            continue
        parent = by_id[span.parent.span_id]
        assert parent.context.trace_id == span.context.trace_id, f"{span.name} left its trace"
        assert parent.name == expected_parent[span.name], f"{span.name} under {parent.name}"
