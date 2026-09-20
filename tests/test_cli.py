"""Tests for the `fineprint` command."""

from collections.abc import Callable

import psycopg
import pytest

from fineprint.cli import main

FAKE_DATABASE_URL = "postgresql://fineprint:hunter2@db.example:5432/handbook"


@pytest.fixture
def no_dot_env(monkeypatch, tmp_path):
    """Run where no `.env` file can change what the command reads."""
    monkeypatch.chdir(tmp_path)


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
        main(["ingest"])

    assert excinfo.value.code == 2


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
