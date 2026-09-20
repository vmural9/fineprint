"""Fixtures shared by the tests, including the databases the integration tests run against.

Integration tests never use the developer's own `fineprint` database. They create databases of
their own, named `fineprint_test_<random>`, and drop them when the session ends.
"""

import secrets
from collections.abc import Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg_pool import ConnectionPool

from fineprint.config import Settings
from fineprint.db import connection_pool, describe_database, init_db

# Seconds to wait for Postgres before deciding it is not running.
CONNECT_TIMEOUT = 3


def with_database_name(database_url: str, name: str) -> str:
    """Return the same connection string pointed at a different database."""
    return urlunsplit(urlsplit(database_url)._replace(path=f"/{name}"))


@pytest.fixture(scope="session")
def new_test_database() -> Iterator[Callable[[], str]]:
    """Return a function that makes a throwaway database and gives back its URL.

    Every database made this way is dropped when the test session ends. Tests that must start
    from nothing ask for one of their own.
    """
    database_url = Settings().database_url
    # "postgres" always exists; we connect to it only to create and drop our own databases.
    maintenance_url = with_database_name(database_url, "postgres")
    try:
        maintenance = psycopg.connect(
            maintenance_url, connect_timeout=CONNECT_TIMEOUT, autocommit=True
        )
    except psycopg.OperationalError as error:
        # Only the first line: libpq adds a paragraph of hints, and the next sentence is the
        # one the reader needs.
        reason = str(error).splitlines()[0].rstrip(". ")
        pytest.skip(
            f"Postgres is not reachable at {describe_database(database_url)}: {reason}. "
            "Start it with: docker compose up -d --wait"
        )

    names: list[str] = []

    def make() -> str:
        name = f"fineprint_test_{secrets.token_hex(4)}"
        maintenance.execute(f'CREATE DATABASE "{name}"')
        names.append(name)
        return with_database_name(database_url, name)

    with maintenance:
        try:
            yield make
        finally:
            for name in names:
                # FORCE disconnects anything still attached, so one leaked connection cannot
                # leave a throwaway database behind.
                maintenance.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def initialized_database(new_test_database: Callable[[], str]) -> str:
    """A throwaway database with the schema applied."""
    database_url = new_test_database()
    init_db(database_url)
    return database_url


@pytest.fixture
def pool(initialized_database: str) -> Iterator[ConnectionPool]:
    """A connection pool on the initialized database, with the tables empty."""
    with connection_pool(initialized_database) as open_pool:
        with open_pool.connection() as connection:
            connection.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
        yield open_pool
