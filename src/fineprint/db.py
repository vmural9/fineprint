"""Talking to Postgres: the schema, a connection pool, and the one-time setup.

There is no ORM. Every query in this project is SQL written out in full, and this module only
supplies the connections it runs on.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from urllib.parse import urlsplit

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DEFAULT_PORT = 5432

# The pool keeps one connection ready and opens more only while several requests overlap.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 4


def read_schema() -> str:
    """Return the DDL in `schema.sql`.

    The file ships inside the package, so `fineprint init-db` works from an installed wheel
    with no checkout in sight.
    """
    return resources.files("fineprint").joinpath("schema.sql").read_text(encoding="utf-8")


def init_db(database_url: str) -> None:
    """Apply `schema.sql`: the extension, the three tables, and the word-search index.

    Safe to run on an empty database and on one that already has the schema. This opens a plain
    connection rather than taking one from the pool, because the pool teaches its connections
    about pgvector's types and those types do not exist until this function has created the
    extension.
    """
    with psycopg.connect(database_url) as connection:
        connection.execute(read_schema())


@contextmanager
def connection_pool(database_url: str) -> Iterator[ConnectionPool]:
    """Open a pool of connections for the duration of the block, and close it afterwards.

    Each connection is taught pgvector's types on the way out, so a `Vector` can be passed to a
    query as an ordinary parameter and an `embedding` column comes back as a `Vector`. The
    database must already have the schema; run `fineprint init-db` first.
    """
    with ConnectionPool(
        database_url,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        configure=register_vector,
        open=True,
    ) as pool:
        yield pool


def describe_database(database_url: str) -> str:
    """Return `host:port/name` for a connection string, with the credentials left out.

    A connection string carries a password, so this is the form that goes into anything a
    person reads.
    """
    url = urlsplit(database_url)
    return f"{url.hostname or 'localhost'}:{url.port or DEFAULT_PORT}/{url.path.lstrip('/')}"
