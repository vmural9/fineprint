"""Tests for the HTTP API.

The app builds its pool, embedder, retriever and chat model in its lifespan handler, and
`TestClient` runs that handler only when it is used as a context manager. These tests never
do, so nothing here opens a database connection or builds an AWS client: each test replaces
the pieces it needs through `app.dependency_overrides`, which is the same seam FastAPI is
built around.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient

from fineprint.answer import DraftAnswer
from fineprint.api import app, get_chat, get_pool, get_retriever, get_settings
from fineprint.config import Settings
from fineprint.providers.base import LLMResult
from fineprint.retrieval import EditionNotIngestedError, RetrievedChunk
from tests.fakes import FakeChatModel

# The heading the premium passage sits under on page 23 of the handbook.
PREMIUM_SECTION = "How much does Part B coverage cost?"

PREMIUM = RetrievedChunk(
    chunk_id=61,
    ordinal=60,
    section=PREMIUM_SECTION,
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


class FakeRetriever:
    """Returns canned chunks, or raises whatever a test wants the retriever to raise."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None, error: Exception | None = None):
        self.chunks = chunks if chunks is not None else [PREMIUM]
        self.error = error
        self.calls: list[tuple[str, str, int | None]] = []

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((question, mode, top_k))
        if self.error is not None:
            raise self.error
        return list(self.chunks)


class BrokenChatModel:
    """A chat model that fails the way Bedrock does when the model will not answer."""

    def __init__(self, error: Exception):
        self.error = error

    def complete_structured(self, system: str, user: str, schema: type) -> LLMResult[Any]:
        raise self.error


class FakeCursor:
    def __init__(self, row: tuple[Any, ...]):
        self.row = row

    def fetchone(self) -> tuple[Any, ...]:
        return self.row


class FakeConnection:
    def __init__(self, row: tuple[Any, ...]):
        self.row = row

    def execute(self, sql: str, params: Any = None) -> FakeCursor:
        return FakeCursor(self.row)


class FakePool:
    """A connection pool that hands out one canned row, or refuses to connect."""

    def __init__(self, chunk_count: int = 267, error: Exception | None = None):
        self.chunk_count = chunk_count
        self.error = error

    @contextmanager
    def connection(self) -> Iterator[FakeConnection]:
        if self.error is not None:
            raise self.error
        yield FakeConnection((self.chunk_count,))


@pytest.fixture
def client(monkeypatch, tmp_path) -> Iterator[TestClient]:
    """A test client with every outside dependency replaced by a fake.

    Each override is a function of no arguments, because FastAPI reads an override's
    parameters as request parameters just as it does for any other dependency.
    """
    monkeypatch.chdir(tmp_path)  # no .env can change what the settings say
    override(get_settings, Settings())
    override(get_pool, FakePool())
    override(get_retriever, FakeRetriever())
    override(get_chat, FakeChatModel(DRAFT))
    yield TestClient(app)
    app.dependency_overrides.clear()


def override(dependency, value) -> None:
    """Point one dependency at `value` for the rest of this test."""
    app.dependency_overrides[dependency] = lambda: value


# --- /search ----------------------------------------------------------------------------


def test_search_returns_the_ranked_chunks_with_both_source_ranks(client: TestClient):
    response = client.post("/search", json={"query": "Part B premium", "mode": "hybrid"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Part B premium"
    assert body["mode"] == "hybrid"
    assert body["results"] == [
        {
            "chunk_id": 61,
            "page_start": 23,
            "page_end": 23,
            "text": "The standard Part B premium amount in 2026 is $202.90.",
            "score": 0.0325,
            "rank": 1,
            "lexical_rank": 1,
            "vector_rank": 2,
        }
    ]


def test_search_passes_the_mode_and_top_k_to_the_retriever(client: TestClient):
    retriever = FakeRetriever()
    override(get_retriever, retriever)

    client.post("/search", json={"query": "hearing aids", "mode": "lexical", "top_k": 3})

    assert retriever.calls == [("hearing aids", "lexical", 3)]


def test_search_defaults_to_hybrid_and_five_results(client: TestClient):
    retriever = FakeRetriever()
    override(get_retriever, retriever)

    client.post("/search", json={"query": "hearing aids"})

    assert retriever.calls == [("hearing aids", "hybrid", 5)]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {},
        {"query": "premium", "top_k": 0},
        {"query": "premium", "top_k": 21},
        {"query": "premium", "mode": "magic"},
    ],
    ids=["empty", "blank", "missing", "top_k too small", "top_k too large", "unknown mode"],
)
def test_search_refuses_a_request_it_cannot_run(client: TestClient, payload: dict):
    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_search_answers_502_when_the_embedder_fails(client: TestClient):
    override(get_retriever, FakeRetriever(error=RuntimeError("Bedrock is not available")))

    response = client.post("/search", json={"query": "Part B premium"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Bedrock is not available" in detail
    assert "Traceback" not in detail


def test_search_answers_503_when_the_edition_has_not_been_ingested(client: TestClient):
    override(get_retriever, FakeRetriever(error=EditionNotIngestedError("2026 is not ingested")))

    response = client.post("/search", json={"query": "Part B premium"})

    assert response.status_code == 503
    assert "not ingested" in response.json()["detail"]


# --- /ask -------------------------------------------------------------------------------


def test_ask_returns_a_cited_answer(client: TestClient):
    response = client.post("/ask", json={"question": "How much is the Part B premium in 2026?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How much is the Part B premium in 2026?"
    assert body["answer"] == "The standard Part B premium in 2026 is $202.90 a month."
    assert body["found_in_handbook"] is True
    assert body["confidence"] == "high"
    assert body["dropped_citations"] == 0
    assert body["retrieved_chunk_ids"] == [61]
    assert body["citations"] == [
        {
            "chunk_id": 61,
            "page_start": 23,
            "page_end": 23,
            "quote": "The standard Part B premium amount in 2026 is $202.90.",
            "quote_verified": True,
        }
    ]
    assert body["model"] == "fake-chat-model"


def test_ask_passes_top_k_to_the_retriever(client: TestClient):
    retriever = FakeRetriever()
    override(get_retriever, retriever)

    client.post("/ask", json={"question": "Does Medicare cover hearing aids?", "top_k": 8})

    assert retriever.calls == [("Does Medicare cover hearing aids?", "hybrid", 8)]


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "  "}, {}, {"question": "x", "top_k": 0}],
    ids=["empty", "blank", "missing", "top_k too small"],
)
def test_ask_refuses_a_request_it_cannot_run(client: TestClient, payload: dict):
    response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_ask_answers_502_when_the_model_fails(client: TestClient):
    override(get_chat, BrokenChatModel(RuntimeError("the model refused to answer")))

    response = client.post("/ask", json={"question": "How much is the Part B premium?"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "the model refused to answer" in detail
    assert "Traceback" not in detail


def test_ask_answers_503_when_the_edition_has_not_been_ingested(client: TestClient):
    override(get_retriever, FakeRetriever(error=EditionNotIngestedError("2026 is not ingested")))

    response = client.post("/ask", json={"question": "How much is the Part B premium?"})

    assert response.status_code == 503


# --- /healthz ---------------------------------------------------------------------------


def test_healthz_reports_the_database_and_what_is_in_it(client: TestClient):
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "localhost:5432/fineprint"
    assert body["edition"] == 2026
    assert body["chunk_set"] == "fixed-220w"
    assert body["chunks"] == 267


def test_healthz_answers_503_when_the_database_is_unreachable(client: TestClient):
    override(get_pool, FakePool(error=psycopg.OperationalError("connection refused")))

    response = client.get("/healthz")

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["status"] == "unavailable"
    assert "connection refused" in body["detail"]["error"]


def test_healthz_does_not_print_the_database_password(client: TestClient):
    override(get_settings, Settings(database_url="postgresql://u:hunter2@db.example:5432/handbook"))

    response = client.get("/healthz")

    assert response.json()["database"] == "db.example:5432/handbook"
    assert "hunter2" not in response.text


# --- the documented schemas -------------------------------------------------------------


def test_the_openapi_schema_describes_both_endpoints_and_the_answer(client: TestClient):
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) >= {"/search", "/ask", "/healthz"}
    assert "AnswerResponse" in schema["components"]["schemas"]
    assert "CitedChunk" in schema["components"]["schemas"]


def test_docs_renders(client: TestClient):
    response = client.get("/docs")

    assert response.status_code == 200
    assert "openapi.json" in response.text
