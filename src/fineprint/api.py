"""The HTTP service: three endpoints over the ingested handbook.

`POST /search` shows what retrieval finds, with no model in the way. It is the endpoint that
makes a bad answer explainable: if `/ask` is wrong, the first question is always whether the
right passage was even retrieved, and this is how you look.

`POST /ask` is the product — a question in, a cited answer out.

`GET /healthz` says whether the database is reachable and how many chunks of the configured
edition are in it, so a deployment can tell "up" from "up but empty".

The expensive objects — the connection pool, the embedder, the retriever and the chat model —
are built once in the lifespan handler and shared by every request. Building an AWS client
takes long enough that doing it per request would be felt, and the pool exists precisely so
connections are not opened per request either. Each one is reachable through a dependency
function, which is the seam the tests replace.

Status codes follow the spec: 422 when the request itself is wrong, 502 when a provider
fails, 503 when the database is unreachable or the edition has not been ingested. An
abstention is not an error — "the handbook doesn't say" is a 200 with
`found_in_handbook: false`.
"""

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool, PoolTimeout
from pydantic import AfterValidator, BaseModel, Field

from fineprint.answer import AnswerResponse, answer_question
from fineprint.config import Settings
from fineprint.db import connection_pool, describe_database
from fineprint.providers import factory
from fineprint.providers.base import ChatModel
from fineprint.retrieval import EditionNotIngestedError, Retriever, SearchMode

logger = logging.getLogger(__name__)

# The default number of chunks an endpoint returns, and the range a caller may ask for. The
# ceiling is a guard rather than a limit anyone should reach: 20 chunks is already more
# context than an answer needs.
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20

# How many chunks of the configured edition and chunk set the database holds. One round trip
# that proves the connection, the schema and the ingest all at once.
HEALTH_SQL = """
SELECT count(*)
FROM chunks AS c JOIN documents AS d ON d.id = c.document_id
WHERE d.edition = %s AND c.chunk_set = %s
"""


def not_blank(text: str) -> str:
    """Reject text that is empty or only spaces, and trim what is left."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("must not be empty")
    return stripped


Question = Annotated[str, AfterValidator(not_blank)]
TopK = Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)]


class SearchRequest(BaseModel):
    """A search: what to look for, how to look, and how much to return."""

    query: Question
    top_k: TopK = DEFAULT_TOP_K
    mode: SearchMode = "hybrid"


class SearchHit(BaseModel):
    """One retrieved chunk, with where it stood in each ranking that found it."""

    chunk_id: int
    page_start: int
    page_end: int
    text: str
    score: float
    rank: int
    lexical_rank: int | None
    vector_rank: int | None


class SearchResponse(BaseModel):
    """What a search found, best first."""

    query: str
    mode: SearchMode
    results: list[SearchHit]


class AskRequest(BaseModel):
    """A question and how many passages to answer it from."""

    question: Question
    top_k: TopK = DEFAULT_TOP_K


class HealthResponse(BaseModel):
    """Whether the database is reachable, and what of the handbook is in it."""

    status: str
    database: str
    edition: int
    chunk_set: str
    chunks: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    """Open the pool and build the models once, and close the pool on the way out.

    Nothing here reaches the network: `ConnectionPool` connects in the background and both
    providers build their AWS clients on first use. A service whose database is down still
    starts, and says so through `/healthz` rather than by refusing to boot.
    """
    settings = Settings()
    with connection_pool(settings.database_url) as pool:
        embedder = factory.get_embedder(settings)
        app.state.settings = settings
        app.state.pool = pool
        app.state.retriever = Retriever(pool, embedder, settings)
        app.state.chat = factory.get_chat_model(settings)
        logger.info(
            "serving edition %s, chunk set %s, from %s",
            settings.corpus_edition,
            settings.chunk_set,
            describe_database(settings.database_url),
        )
        yield


app = FastAPI(
    title="fineprint",
    version="0.1.0",
    summary='Answers Medicare questions from the official "Medicare & You 2026" handbook, '
    "with the page each answer came from.",
    lifespan=lifespan,
)


def get_settings(request: Request) -> Settings:
    """The settings this process started with."""
    return request.app.state.settings


def get_pool(request: Request) -> ConnectionPool:
    """The shared connection pool."""
    return request.app.state.pool


def get_retriever(request: Request) -> Retriever:
    """The shared retriever, with the embedder already behind it."""
    return request.app.state.retriever


def get_chat(request: Request) -> ChatModel:
    """The shared chat model."""
    return request.app.state.chat


SettingsDep = Annotated[Settings, Depends(get_settings)]
PoolDep = Annotated[ConnectionPool, Depends(get_pool)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
ChatDep = Annotated[ChatModel, Depends(get_chat)]


def first_line(error: Exception) -> str:
    """The first line of an error, which is the sentence a caller can act on."""
    return str(error).splitlines()[0] if str(error) else error.__class__.__name__


@contextmanager
def provider_failures(doing: str) -> Iterator[None]:
    """Turn a failure behind the endpoint into the status code the spec asks for.

    A caller gets one short sentence. The traceback goes to the service's own log, where it
    belongs: a stack trace in an HTTP response tells an attacker about the service and tells
    an ordinary caller nothing.
    """
    try:
        yield
    except EditionNotIngestedError as error:
        raise HTTPException(status_code=503, detail=first_line(error)) from error
    except (psycopg.Error, PoolTimeout) as error:
        raise HTTPException(
            status_code=503, detail=f"the database is not available: {first_line(error)}"
        ) from error
    except Exception as error:
        logger.exception("%s failed", doing)
        raise HTTPException(
            status_code=502, detail=f"{doing} failed: {first_line(error)}"
        ) from error


@app.post("/search", summary="Find the handbook passages a question is about")
def search(request: SearchRequest, retriever: RetrieverDep) -> SearchResponse:
    """Return the ranked chunks for a query, without asking a model anything.

    `lexical_rank` and `vector_rank` say which retriever found each chunk and where it stood
    in that retriever's own list; either is null when that retriever did not return it.
    """
    with provider_failures("search"):
        chunks = retriever.search(request.query, mode=request.mode, top_k=request.top_k)
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        results=[SearchHit(**asdict(chunk)) for chunk in chunks],
    )


@app.post("/ask", summary="Answer a Medicare question from the handbook, with citations")
def ask(request: AskRequest, retriever: RetrieverDep, chat: ChatDep) -> AnswerResponse:
    """Retrieve, answer, and check the citations.

    An answer the handbook does not support comes back with `found_in_handbook: false` and a
    pointer to somewhere that does know. That is a successful request, not an error.
    """
    with provider_failures("answering"):
        return answer_question(request.question, retriever, chat, top_k=request.top_k)


@app.get("/healthz", summary="Check the database connection")
def healthz(settings: SettingsDep, pool: PoolDep) -> HealthResponse:
    """Count the configured edition's chunks, which needs the database to be working.

    `chunks: 0` means the database is up but the handbook has not been ingested into it.
    """
    try:
        with pool.connection() as connection:
            (chunks,) = connection.execute(
                HEALTH_SQL, (settings.corpus_edition, settings.chunk_set)
            ).fetchone()
    except (psycopg.Error, PoolTimeout) as error:
        raise HTTPException(
            status_code=503,
            detail={"status": "unavailable", "error": first_line(error)},
        ) from error
    return HealthResponse(
        status="ok",
        database=describe_database(settings.database_url),
        edition=settings.corpus_edition,
        chunk_set=settings.chunk_set,
        chunks=chunks,
    )
