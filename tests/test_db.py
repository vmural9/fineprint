"""Tests for the packaged schema and the database helpers that need no database."""

from fineprint.db import describe_database, read_schema


def test_the_schema_travels_inside_the_package():
    """`init-db` reads the DDL from the installed package, not from the checkout."""
    schema = read_schema()

    assert "CREATE EXTENSION IF NOT EXISTS vector" in schema
    for table in ("documents", "pages", "chunks"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema


def test_the_chunks_table_has_a_column_for_each_kind_of_search():
    """Word search reads `tsv`, meaning search reads `embedding`."""
    schema = " ".join(read_schema().split())

    assert "tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED" in schema
    assert "embedding vector(1024) NOT NULL" in schema


def test_every_statement_may_be_run_twice():
    """`fineprint init-db` must be safe on a database that already has the schema."""
    statements = [line for line in read_schema().splitlines() if line.startswith("CREATE")]

    assert statements, "no CREATE statements found in schema.sql"
    for statement in statements:
        assert "IF NOT EXISTS" in statement, statement


def test_describe_database_leaves_out_the_credentials():
    """The connection string carries a password, so nothing prints it back."""
    target = describe_database("postgresql://fineprint:hunter2@localhost:5432/fineprint")

    assert target == "localhost:5432/fineprint"
    assert "hunter2" not in target


def test_describe_database_fills_in_the_default_port():
    assert describe_database("postgresql://db.example/handbook") == "db.example:5432/handbook"
