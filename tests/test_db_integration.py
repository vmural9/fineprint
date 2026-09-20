"""Database tests: they need Postgres, so every test here carries the `integration` marker."""

import psycopg
import pytest
from pgvector import Vector
from psycopg_pool import ConnectionPool

from fineprint.db import init_db

pytestmark = pytest.mark.integration

# As wide as the `embedding` column and as the Bedrock Titan v2 model. Every value is a multiple
# of 1/1024, which float32 stores exactly, so the numbers that come back are the numbers that
# went in.
SAMPLE_EMBEDDING = [index / 1024 for index in range(1024)]

CHUNK_TEXT = "The standard Part B premium amount in 2026 is $202.90 a month."


def table_names(database_url: str) -> set[str]:
    """The tables the schema created, read back from Postgres itself."""
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {name for (name,) in rows}


def insert_chunk(connection: psycopg.Connection, embedding: list[float]) -> int:
    """Insert one handbook, one page, and one chunk, and return the chunk's id."""
    document_id = connection.execute(
        "INSERT INTO documents (edition, title, source_url, sha256, page_count)"
        " VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (2026, "Medicare & You 2026", "https://example.invalid/handbook.pdf", "0" * 64, 128),
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO pages (document_id, page_number, text) VALUES (%s, %s, %s)",
        (document_id, 23, CHUNK_TEXT),
    )
    return connection.execute(
        "INSERT INTO chunks"
        " (document_id, chunk_set, ordinal, page_start, page_end, text, embedding)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (document_id, "fixed-220w", 0, 23, 24, CHUNK_TEXT, Vector(embedding)),
    ).fetchone()[0]


def test_init_db_creates_the_extension_and_the_tables(initialized_database):
    with psycopg.connect(initialized_database) as connection:
        extensions = connection.execute("SELECT extname FROM pg_extension").fetchall()

    assert ("vector",) in extensions
    assert {"documents", "pages", "chunks"} <= table_names(initialized_database)


def test_init_db_runs_again_on_an_initialized_database(initialized_database):
    """The fixture applied the schema once already; these are the second and third runs."""
    before = table_names(initialized_database)

    init_db(initialized_database)
    init_db(initialized_database)

    assert table_names(initialized_database) == before


def test_a_chunk_round_trips_with_its_pages_and_its_vector(pool: ConnectionPool):
    with pool.connection() as connection:
        chunk_id = insert_chunk(connection, SAMPLE_EMBEDDING)

    with pool.connection() as connection:
        chunk_set, ordinal, page_start, page_end, section, text, embedding = connection.execute(
            "SELECT chunk_set, ordinal, page_start, page_end, section, text, embedding"
            " FROM chunks WHERE id = %s",
            (chunk_id,),
        ).fetchone()

    assert (chunk_set, ordinal, page_start, page_end) == ("fixed-220w", 0, 23, 24)
    assert section is None, "part 1 does not fill in sections; the part 2 chunker will"
    assert text == CHUNK_TEXT
    assert isinstance(embedding, Vector), "the pool registered pgvector's types"
    assert embedding.to_list() == SAMPLE_EMBEDDING


def test_the_stored_vector_is_its_own_nearest_neighbour(pool: ConnectionPool):
    """Proves the vector really reached Postgres: cosine distance to itself is zero."""
    with pool.connection() as connection:
        insert_chunk(connection, SAMPLE_EMBEDDING)
        (distance,) = connection.execute(
            "SELECT embedding <=> %s FROM chunks", (Vector(SAMPLE_EMBEDDING),)
        ).fetchone()

    assert distance == pytest.approx(0.0, abs=1e-6)


def test_the_generated_tsv_column_indexes_the_chunk_text(pool: ConnectionPool):
    """Nothing writes `tsv`: Postgres fills it from `text`, stemmed, on every insert."""
    with pool.connection() as connection:
        chunk_id = insert_chunk(connection, SAMPLE_EMBEDDING)
        matches = connection.execute(
            "SELECT id FROM chunks WHERE tsv @@ plainto_tsquery('english', %s)", ("premiums",)
        ).fetchall()

    assert matches == [(chunk_id,)], "'premiums' should find a chunk that says 'premium'"


def test_a_vector_of_the_wrong_width_is_rejected(pool: ConnectionPool):
    """The column width is the guard rail that stops a chunk set mixing embedding models."""
    with pytest.raises(psycopg.Error) as excinfo, pool.connection() as connection:
        insert_chunk(connection, [0.1] * 384)

    assert "1024 dimensions" in str(excinfo.value)
