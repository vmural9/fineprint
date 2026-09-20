"""Turning the handbook PDF into rows the service can search.

Ingestion is the one-time preparation that stands between a 128-page PDF and a question like
"what is the Part B premium?". It reads the pages, cuts them into short passages, asks the
embedding model for each passage's vector, and writes the lot to Postgres. Retrieval later only
reads those rows; nothing at question time opens the PDF.

Five steps, in this order, and the cheap checks come first so that a run that cannot possibly
work costs nothing:

1. verify the PDF is the pinned bytes, because answers are only true of one printing;
2. check the embedder's width against the `chunks.embedding` column, because a mismatch would
   otherwise be discovered after several hundred paid embedding calls;
3. extract the pages and cut them into chunks;
4. embed the chunks in batches;
5. write `documents`, `pages`, and `chunks` in one transaction.

Re-running replaces the document's rows for that chunk set instead of adding a second copy, so
`fineprint ingest` can be run twice in a row — after a crash, or with a new chunker — without
anyone having to clean up first. Other chunk sets of the same document are left alone, which is
what lets part 2 compare two chunkings side by side.
"""

import re
import statistics
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import psycopg
from pgvector import Vector
from psycopg_pool import ConnectionPool

from fineprint.chunking import CHUNKERS, Chunk
from fineprint.editions import Edition, pdf_path, sha256_of
from fineprint.pdf import Page, extract_pages
from fineprint.providers.base import Embedder

# How many chunks are handed to the embedder at a time. The Bedrock embedder spreads a batch
# over its own thread pool, so this is about keeping one failure small and progress steady,
# not about a limit of the API.
EMBED_BATCH_SIZE = 64

# How Postgres prints the type of a pgvector column: `vector(1024)`.
VECTOR_TYPE = re.compile(r"vector\((\d+)\)")

# The width of `chunks.embedding`, read from the database rather than from `schema.sql`, so the
# check is against the database that is about to be written to.
EMBEDDING_TYPE_SQL = """
    SELECT format_type(atttypid, atttypmod)
    FROM pg_attribute
    WHERE attrelid = to_regclass('chunks') AND attname = 'embedding' AND NOT attisdropped
"""

# One row per edition. Re-ingesting keeps the row and its id, so the pages and chunks of the
# other chunk sets keep pointing at it.
UPSERT_DOCUMENT_SQL = """
    INSERT INTO documents (edition, title, source_url, sha256, page_count)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (edition) DO UPDATE SET
        title = EXCLUDED.title,
        source_url = EXCLUDED.source_url,
        sha256 = EXCLUDED.sha256,
        page_count = EXCLUDED.page_count,
        ingested_at = now()
    RETURNING id
"""

INSERT_PAGE_SQL = "INSERT INTO pages (document_id, page_number, text) VALUES (%s, %s, %s)"

# `tsv` is missing on purpose: Postgres generates it from `text` on every insert.
INSERT_CHUNK_SQL = """
    INSERT INTO chunks
        (document_id, chunk_set, ordinal, page_start, page_end, section, text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


class IngestError(RuntimeError):
    """Ingestion refused to run, or could not finish. The message says what to do about it."""


@dataclass(frozen=True)
class WordStats:
    """How long the chunks came out, in words.

    Worth printing because it is the first place a broken chunker shows: a median far from the
    chunker's window size means the pages did not extract the way they were expected to.
    """

    minimum: int
    median: float
    maximum: int


@dataclass(frozen=True)
class IngestSummary:
    """What one run put in the database."""

    edition: int
    chunk_set: str
    page_count: int
    chunk_count: int
    words: WordStats


def word_stats(texts: list[str]) -> WordStats:
    """The shortest, middle, and longest of `texts`, measured in words.

    A document with no chunks reports zeros rather than raising: an empty result is a thing to
    print and look at, not a crash.
    """
    counts = sorted(len(text.split()) for text in texts)
    if not counts:
        return WordStats(minimum=0, median=0.0, maximum=0)
    return WordStats(minimum=counts[0], median=float(statistics.median(counts)), maximum=counts[-1])


def dimension_of(declared_type: str) -> int:
    """The width in `vector(1024)`, the type Postgres reports for an embedding column."""
    match = VECTOR_TYPE.fullmatch(declared_type.strip())
    if match is None:
        raise IngestError(
            f"the chunks.embedding column is declared {declared_type!r}, which is not a sized "
            "vector type. Apply the schema with: fineprint init-db"
        )
    return int(match.group(1))


def embedding_column_dimension(pool: ConnectionPool) -> int:
    """How many numbers one row of `chunks.embedding` holds in this database."""
    with pool.connection() as connection:
        row = connection.execute(EMBEDDING_TYPE_SQL).fetchone()
    if row is None:
        raise IngestError(
            "this database has no chunks.embedding column to write to. Create the schema with: "
            "fineprint init-db"
        )
    return dimension_of(row[0])


def verified_pdf(edition: Edition, path: Path | None = None) -> Path:
    """Return the path to this edition's PDF, once its bytes are the pinned ones.

    The hash is the whole reason the corpus is reproducible: CMS revises the handbook during the
    year, so a file with the right name can still hold last year's dollar amounts.
    """
    path = pdf_path(edition) if path is None else path
    if not path.is_file():
        raise IngestError(
            f"no handbook at {path}. Fetch the pinned PDF with: "
            f"uv run python scripts/download_handbook.py --edition {edition.year}"
        )
    actual = sha256_of(path)
    if actual != edition.sha256:
        raise IngestError(
            f"{path} is not the pinned {edition.year} edition: its sha256 is {actual}, and the "
            f"pinned value is {edition.sha256}. Re-fetch it with: "
            f"uv run python scripts/download_handbook.py --edition {edition.year} --force"
        )
    return path


def embed_in_batches(
    embedder: Embedder, texts: list[str], batch_size: int = EMBED_BATCH_SIZE
) -> list[list[float]]:
    """The vector for every text, in the order the texts were given."""
    vectors: list[list[float]] = []
    for batch in batched(texts, batch_size):
        vectors.extend(embedder.embed_documents(list(batch)))
    if len(vectors) != len(texts):
        raise IngestError(
            f"the embedder returned {len(vectors)} vectors for {len(texts)} chunks; every chunk "
            "needs its own vector, and they must come back in order"
        )
    return vectors


def ingest(
    pool: ConnectionPool,
    embedder: Embedder,
    edition: Edition,
    chunk_set: str,
    *,
    path: Path | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> IngestSummary:
    """Read one edition of the handbook into the database, and say what was written.

    `path` defaults to `data/raw/medicare-and-you-<year>.pdf` under the working directory.
    Running this again for the same edition and chunk set replaces those rows.
    """
    chunker = CHUNKERS.get(chunk_set)
    if chunker is None:
        raise IngestError(
            f"no chunker is registered under the chunk set {chunk_set!r}. "
            f"Known chunk sets: {', '.join(sorted(CHUNKERS))}."
        )
    path = verified_pdf(edition, path)

    column_dimension = embedding_column_dimension(pool)
    if embedder.dimension != column_dimension:
        raise IngestError(
            f"the embedder returns vectors of {embedder.dimension} numbers but the "
            f"chunks.embedding column holds {column_dimension}. Embeddings from two models "
            "cannot be compared, so change EMBEDDING_MODEL back, or edit the vector(...) width "
            "in schema.sql and re-ingest into a fresh database."
        )

    pages = extract_pages(path)
    chunks = chunker(pages)
    vectors = embed_in_batches(embedder, [chunk.text for chunk in chunks], batch_size)

    with pool.connection() as connection:
        store(connection, edition, chunk_set, pages, chunks, vectors)

    return IngestSummary(
        edition=edition.year,
        chunk_set=chunk_set,
        page_count=len(pages),
        chunk_count=len(chunks),
        words=word_stats([chunk.text for chunk in chunks]),
    )


def store(
    connection: psycopg.Connection,
    edition: Edition,
    chunk_set: str,
    pages: list[Page],
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> int:
    """Write the document, its pages, and one chunk set, and return the document's id.

    Everything here runs in the caller's transaction, so a failure anywhere leaves the database
    exactly as it was rather than half-ingested. Replacing rather than adding is what makes a
    second run safe: the old pages and the old rows of *this* chunk set go first, and other
    chunk sets of the same document are untouched.
    """
    document_id = connection.execute(
        UPSERT_DOCUMENT_SQL,
        (edition.year, edition.title, edition.primary_url, edition.sha256, len(pages)),
    ).fetchone()[0]

    connection.execute("DELETE FROM pages WHERE document_id = %s", (document_id,))
    connection.execute(
        "DELETE FROM chunks WHERE document_id = %s AND chunk_set = %s", (document_id, chunk_set)
    )

    with connection.cursor() as cursor:
        cursor.executemany(
            INSERT_PAGE_SQL,
            [(document_id, page.number, page.text) for page in pages],
        )
        cursor.executemany(
            INSERT_CHUNK_SQL,
            [
                (
                    document_id,
                    chunk_set,
                    chunk.ordinal,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.section,
                    chunk.text,
                    Vector(vector),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    return document_id
