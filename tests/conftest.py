"""Fixtures shared by the tests, including the database the integration tests run against.

Integration tests never use the developer's own `fineprint` database. They create a database of
their own, named `fineprint_test_<random>`, and drop it when the session ends.
"""

import secrets
from collections.abc import Iterator
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
def test_database_url() -> Iterator[str]:
    """Create a throwaway database for this test session and drop it afterwards."""
    settings = Settings()
    # "postgres" always exists; we connect to it only to create and drop our own database.
    maintenance_url = with_database_name(settings.database_url, "postgres")
    try:
        maintenance = psycopg.connect(
            maintenance_url, connect_timeout=CONNECT_TIMEOUT, autocommit=True
        )
    except psycopg.OperationalError as error:
        pytest.skip(
            f"Postgres is not reachable at {describe_database(settings.database_url)}: "
            f"{str(error).strip()} Start it with: docker compose up -d --wait"
        )

    name = f"fineprint_test_{secrets.token_hex(4)}"
    with maintenance:
        maintenance.execute(f'CREATE DATABASE "{name}"')
        try:
            yield with_database_name(settings.database_url, name)
        finally:
            # FORCE disconnects anything still attached, so one leaked connection cannot
            # leave the throwaway database behind.
            maintenance.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def initialized_database(test_database_url: str) -> str:
    """The throwaway database with the schema applied."""
    init_db(test_database_url)
    return test_database_url


@pytest.fixture
def pool(initialized_database: str) -> Iterator[ConnectionPool]:
    """A connection pool on the throwaway database, with the tables empty."""
    with connection_pool(initialized_database) as open_pool:
        with open_pool.connection() as connection:
            connection.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
        yield open_pool
