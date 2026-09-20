"""Tests for the `fineprint` command.

Nothing here reaches the database or AWS except the one test marked `integration`: the ingest
tests replace the pool, the embedder, and the ingest run itself, so what is under test is the
wiring — which edition and chunk set a flag chooses, and what the command prints.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psycopg
import pytest

from fineprint.cli import main
from fineprint.ingest import IngestError, IngestSummary, WordStats
from tests.fakes import FakeEmbedder

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
