"""Tests for the `fineprint` command.

Nothing here reaches the database or AWS except the one test marked `integration`: the ingest
tests replace the pool, the embedder, and the ingest run itself, so what is under test is the
wiring — which edition and chunk set a flag chooses, and what the command prints.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import psycopg
import pytest

from fineprint.answer import Citation, DraftAnswer
from fineprint.cli import main
from fineprint.ingest import IngestError, IngestSummary, WordStats
from fineprint.retrieval import EditionNotIngestedError, RetrievedChunk
from tests.fakes import FakeChatModel, FakeEmbedder

FAKE_DATABASE_URL = "postgresql://fineprint:hunter2@db.example:5432/handbook"

# What a real run of the 2026 handbook reports, so the test asserts on realistic output.
SUMMARY = IngestSummary(
    edition=2026,
    chunk_set="fixed-220w",
    page_count=128,
    chunk_count=257,
    words=WordStats(minimum=43, median=220.0, maximum=220),
)


@pytest.fixture
def no_dot_env(monkeypatch, tmp_path):
    """Run where no `.env` file can change what the command reads."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def offline_ingest(monkeypatch, no_dot_env) -> list[tuple[int, str]]:
    """Stub the pool and the embedder, and record the edition and chunk set ingest was given."""
    monkeypatch.setenv("DATABASE_URL", FAKE_DATABASE_URL)
    monkeypatch.setattr("fineprint.cli.get_embedder", lambda settings: FakeEmbedder())

    @contextmanager
    def fake_pool(database_url: str) -> Iterator[None]:
        yield None

    monkeypatch.setattr("fineprint.cli.connection_pool", fake_pool)

    asked: list[tuple[int, str]] = []

    def fake_ingest(pool, embedder, edition, chunk_set) -> IngestSummary:
        asked.append((edition.year, chunk_set))
        return SUMMARY

    monkeypatch.setattr("fineprint.cli.ingest", fake_ingest)
    return asked


def test_init_db_applies_the_schema_to_the_configured_database(monkeypatch, capsys, no_dot_env):
    monkeypatch.setenv("DATABASE_URL", FAKE_DATABASE_URL)
    applied: list[str] = []
    monkeypatch.setattr("fineprint.cli.init_db", applied.append)

    exit_code = main(["init-db"])

    assert exit_code == 0
    assert applied == [FAKE_DATABASE_URL]


def test_init_db_reports_where_it_wrote_without_printing_the_password(
    monkeypatch, capsys, no_dot_env
):
    monkeypatch.setenv("DATABASE_URL", FAKE_DATABASE_URL)
    monkeypatch.setattr("fineprint.cli.init_db", lambda database_url: None)

    main(["init-db"])

    printed = capsys.readouterr().out
    assert "db.example:5432/handbook" in printed
    assert "hunter2" not in printed


def test_the_command_asks_for_a_subcommand(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    assert "init-db" in capsys.readouterr().err


def test_an_unknown_subcommand_is_refused(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["reindex"])

    assert excinfo.value.code == 2


def test_ingest_reports_what_it_wrote_and_how_long_it_took(offline_ingest, capsys):
    exit_code = main(["ingest"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "128" in printed, "the page count"
    assert "257" in printed, "the chunk count"
    assert "min 43, median 220, max 220" in printed
    assert "elapsed" in printed
    assert "db.example:5432/handbook" in printed and "hunter2" not in printed


def test_ingest_falls_back_to_the_configured_edition_and_chunk_set(offline_ingest, monkeypatch):
    monkeypatch.setenv("CORPUS_EDITION", "2025")
    monkeypatch.setenv("CHUNK_SET", "fixed-220w")

    main(["ingest"])

    assert offline_ingest == [(2025, "fixed-220w")]


def test_the_flags_win_over_the_configuration(offline_ingest, monkeypatch):
    monkeypatch.setenv("CORPUS_EDITION", "2025")

    main(["ingest", "--edition", "2026", "--chunk-set", "fixed-220w"])

    assert offline_ingest == [(2026, "fixed-220w")]


def test_an_edition_that_is_not_pinned_is_refused(offline_ingest, monkeypatch, capsys):
    monkeypatch.setenv("CORPUS_EDITION", "2019")

    exit_code = main(["ingest"])

    assert exit_code == 1
    assert "2019" in capsys.readouterr().err
    assert offline_ingest == [], "nothing should have been ingested"


def test_a_refusal_is_printed_and_exits_non_zero(offline_ingest, monkeypatch, capsys):
    def refuse(pool, embedder, edition, chunk_set) -> IngestSummary:
        raise IngestError("the embedder returns vectors of 384 numbers")

    monkeypatch.setattr("fineprint.cli.ingest", refuse)

    exit_code = main(["ingest"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "384" in captured.err
    assert captured.out == "", "a refusal belongs on stderr, and nothing was written"


@pytest.mark.integration
def test_init_db_works_on_an_empty_database_and_again_on_an_initialized_one(
    new_test_database: Callable[[], str], monkeypatch, no_dot_env
):
    database_url = new_test_database()
    monkeypatch.setenv("DATABASE_URL", database_url)

    on_empty = main(["init-db"])
    on_initialized = main(["init-db"])

    assert (on_empty, on_initialized) == (0, 0)
    with psycopg.connect(database_url) as connection:
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    assert {name for (name,) in tables} == {"documents", "pages", "chunks"}


# --- search, ask, serve -----------------------------------------------------------------

PREMIUM_TEXT = (
    "The standard Part B premium amount in 2026 is $202.90. Most people pay the standard "
    "Part B premium amount every month. If your modified adjusted gross income is above a "
    "certain amount, you may pay an Income-Related Monthly Adjustment Amount (IRMAA)."
)
HEARING_TEXT = (
    "Note: Original Medicare doesn't cover hearing aids or exams for fitting hearing aids. "
    "Medicare covers diagnostic hearing and balance (fall risk) exams if your doctor orders them."
)
PREMIUM_QUOTE = "The standard Part B premium amount in 2026 is $202.90."

CHUNKS = [
    RetrievedChunk(
        chunk_id=61,
        ordinal=60,
        section=None,
        page_start=23,
        page_end=23,
        text=PREMIUM_TEXT,
        score=0.0325,
        rank=1,
        lexical_rank=1,
        vector_rank=2,
        fused_rank=1,
        rerank_score=None,
    ),
    RetrievedChunk(
        chunk_id=118,
        ordinal=117,
        section=None,
        page_start=42,
        page_end=43,
        text=HEARING_TEXT,
        score=0.0161,
        rank=2,
        lexical_rank=None,
        vector_rank=4,
        fused_rank=2,
        rerank_score=None,
    ),
]

PREMIUM_DRAFT = DraftAnswer(
    answer="The standard Part B premium in 2026 is $202.90 a month.",
    found_in_handbook=True,
    citations=[Citation(chunk_id=61, quote=PREMIUM_QUOTE)],
    confidence="high",
)


class FakeRetriever:
    """Returns canned chunks and remembers how the command asked for them."""

    def __init__(self, chunks: list[RetrievedChunk], error: Exception | None = None):
        self.chunks = chunks
        self.error = error
        self.calls: list[tuple[str, str, int | None]] = []

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((question, mode, top_k))
        if self.error is not None:
            raise self.error
        return list(self.chunks)


@pytest.fixture
def offline_service(monkeypatch, no_dot_env) -> SimpleNamespace:
    """Replace the pool, the embedder, the retriever and the chat model with fakes.

    `service.retriever` records what the command searched for and `service.draft` is what
    the model "returns", so a test can change either before running the command.
    """
    monkeypatch.setenv("DATABASE_URL", FAKE_DATABASE_URL)
    monkeypatch.setattr("fineprint.cli.get_embedder", lambda settings: FakeEmbedder())

    @contextmanager
    def fake_pool(database_url: str) -> Iterator[None]:
        yield None

    monkeypatch.setattr("fineprint.cli.connection_pool", fake_pool)

    service = SimpleNamespace(retriever=FakeRetriever(CHUNKS), draft=PREMIUM_DRAFT)
    monkeypatch.setattr(
        "fineprint.cli.Retriever", lambda pool, embedder, settings: service.retriever
    )
    monkeypatch.setattr(
        "fineprint.cli.get_chat_model", lambda settings: FakeChatModel(service.draft)
    )
    return service


def test_search_prints_ranks_pages_both_source_ranks_and_the_start_of_each_chunk(
    offline_service, capsys
):
    exit_code = main(["search", "How much is the Part B premium in 2026?"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "How much is the Part B premium in 2026?" in printed
    assert "23" in printed and "42-43" in printed, "page ranges"
    assert "0.0325" in printed, "the fused score"
    assert PREMIUM_TEXT[:100] in printed
    assert PREMIUM_TEXT[:120] not in printed, "only the first 100 characters"
    # Both source ranks, and a dash where a retriever did not return the chunk.
    assert "lexical 1" in printed and "vector 2" in printed
    assert "lexical -" in printed and "vector 4" in printed


def test_search_uses_the_mode_and_top_k_it_is_given(offline_service):
    main(["search", "hearing aids", "--mode", "lexical", "--top-k", "3"])

    assert offline_service.retriever.calls == [("hearing aids", "lexical", 3)]


def test_search_falls_back_to_hybrid_and_the_configured_top_k(offline_service, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")

    main(["search", "hearing aids"])

    assert offline_service.retriever.calls == [("hearing aids", "hybrid", 7)]


def test_search_says_so_when_nothing_matched(offline_service, capsys):
    offline_service.retriever = FakeRetriever([])

    exit_code = main(["search", "the of and"])

    assert exit_code == 0
    assert "no chunks" in capsys.readouterr().out.lower()


def test_search_refuses_an_unknown_mode(offline_service, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["search", "premium", "--mode", "magic"])

    assert excinfo.value.code == 2


def test_a_missing_edition_is_reported_without_a_traceback(offline_service, capsys):
    offline_service.retriever = FakeRetriever(
        [], error=EditionNotIngestedError("the 2026 edition has not been ingested")
    )

    exit_code = main(["search", "premium"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "has not been ingested" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_a_provider_failure_is_reported_without_a_traceback(offline_service, capsys):
    offline_service.retriever = FakeRetriever([], error=RuntimeError("Bedrock is throttling"))

    exit_code = main(["ask", "How much is the Part B premium?"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Bedrock is throttling" in captured.err
    assert "Traceback" not in captured.err


def test_ask_prints_the_answer_its_citations_and_what_the_call_cost(offline_service, capsys):
    exit_code = main(["ask", "How much is the Part B premium in 2026?"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "$202.90" in printed
    assert "found in the handbook" in printed.lower()
    assert "yes" in printed.lower()
    assert "high" in printed, "the confidence"
    assert "page 23" in printed
    assert PREMIUM_QUOTE in printed
    assert "verified" in printed.lower()
    assert "fake-chat-model" in printed
    assert "900" in printed and "120" in printed, "the token counts"
    assert "ms" in printed or "s" in printed, "the latency"


def test_ask_marks_a_quote_it_could_not_find_in_the_chunk(offline_service, capsys):
    offline_service.draft = DraftAnswer(
        answer="The standard Part B premium in 2026 is $164.90 a month.",
        found_in_handbook=True,
        citations=[Citation(chunk_id=61, quote="The standard Part B premium in 2026 is $164.90.")],
        confidence="high",
    )

    main(["ask", "How much is the Part B premium in 2026?"])

    printed = capsys.readouterr().out.lower()
    assert "not verified" in printed


def test_ask_reports_a_citation_it_had_to_drop(offline_service, capsys):
    offline_service.draft = DraftAnswer(
        answer="Medicare pays for everything.",
        found_in_handbook=True,
        citations=[Citation(chunk_id=999, quote="Medicare pays for everything.")],
        confidence="low",
    )

    main(["ask", "Does Medicare pay for everything?"])

    printed = capsys.readouterr().out.lower()
    assert "dropped" in printed


def test_ask_prints_an_abstention_plainly(offline_service, capsys):
    offline_service.draft = DraftAnswer(
        answer="The handbook doesn't list covered drugs. Check the plan's formulary.",
        found_in_handbook=False,
        citations=[],
        confidence="low",
    )

    main(["ask", "Does Medicare drug coverage pay for Eliquis?"])

    printed = capsys.readouterr().out.lower()
    assert "no" in printed
    assert "formulary" in printed


def test_serve_runs_uvicorn_on_the_host_and_port_it_is_given(monkeypatch, no_dot_env, capsys):
    asked: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "fineprint.cli.uvicorn.run",
        lambda app, **options: asked.append((app, options)),
    )

    exit_code = main(["serve", "--host", "0.0.0.0", "--port", "9001"])

    assert exit_code == 0
    (app, options) = asked[0]
    assert app == "fineprint.api:app"
    assert (options["host"], options["port"]) == ("0.0.0.0", 9001)
    assert "9001" in capsys.readouterr().out


def test_serve_defaults_to_localhost_on_port_8000(monkeypatch, no_dot_env, capsys):
    asked: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "fineprint.cli.uvicorn.run",
        lambda app, **options: asked.append((app, options)),
    )

    main(["serve"])

    (_, options) = asked[0]
    assert (options["host"], options["port"]) == ("127.0.0.1", 8000)
