"""Tests for ingestion: the pure helpers, and the whole run against a throwaway database.

Nothing here downloads the handbook or calls AWS. The integration tests build a three-page PDF
of their own, pin it by its own hash the way the real corpus is pinned, and embed it with
`FakeEmbedder`, so a full ingest runs offline in a fraction of a second.
"""

from dataclasses import replace
from pathlib import Path

import pytest
from psycopg_pool import ConnectionPool

from fineprint.editions import Edition, sha256_of
from fineprint.ingest import (
    IngestError,
    dimension_of,
    embed_in_batches,
    ingest,
    word_stats,
)
from tests.fakes import FakeEmbedder

# One real sentence per page, so the fixture reads like the handbook, padded with words that
# name their own page (`p2w007` is the eighth word of page 2) so a test can say exactly which
# page a chunk's first and last word came from.
SENTENCES = {
    1: "Medicare Part A covers inpatient hospital stays, skilled nursing care, and hospice care.",
    2: "The standard Part B premium amount in 2026 is $202.90 each month.",
    3: "Part D plans cap what you pay out of pocket for covered prescription drugs each year.",
}
WORDS_PER_PAGE = 150

# 450 words cut into 220-word windows that overlap by 40 gives three chunks, the last one short.
EXPECTED_CHUNKS = 3
EXPECTED_PAGE_RANGES = [(1, 2), (2, 3), (3, 3)]
EXPECTED_WORD_COUNTS = [220, 220, 90]


def page_text(number: int, word_count: int = WORDS_PER_PAGE) -> str:
    """One page of the stand-in handbook: a real sentence, then filler that names its page."""
    sentence = SENTENCES[number].split()
    filler = [f"p{number}w{index:03d}" for index in range(word_count - len(sentence))]
    return " ".join(sentence + filler)


def minimal_pdf(pages: list[str]) -> bytes:
    """A PDF with one page per string, written out by hand.

    Building the fixture here rather than committing a binary keeps the test readable: the words
    a test asserts on are in the test. The objects are the smallest set a PDF reader accepts —
    a catalog, a page tree, one font, and a page plus a content stream for each page.
    """
    page_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, pages, strict=True):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    table_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        table_at,
    )
    return bytes(out)


class RecordingEmbedder(FakeEmbedder):
    """A `FakeEmbedder` that remembers how many texts it was handed each time."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed_documents(texts)


class NarrowEmbedder(FakeEmbedder):
    """An embedder whose vectors are too short for the `chunks.embedding` column."""

    dimension = 384


@pytest.fixture
def handbook(tmp_path: Path) -> tuple[Edition, Path]:
    """A three-page stand-in handbook, pinned by its own hash the way the real one is."""
    path = tmp_path / "medicare-and-you-2026.pdf"
    path.write_bytes(minimal_pdf([page_text(number) for number in sorted(SENTENCES)]))
    edition = Edition(
        year=2026,
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        pages=len(SENTENCES),
        revision="three-page stand-in built by the tests",
        urls=("https://example.invalid/medicare-and-you-2026.pdf",),
    )
    return edition, path


def counts(pool: ConnectionPool) -> tuple[int, int, int]:
    """How many documents, pages, and chunks the database holds."""
    with pool.connection() as connection:
        return tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("documents", "pages", "chunks")
        )


# --- pure helpers -----------------------------------------------------------------------


def test_the_column_width_is_read_out_of_the_declared_type():
    assert dimension_of("vector(1024)") == 1024


@pytest.mark.parametrize("declared", ["vector", "text", "vector(many)", ""])
def test_a_column_that_is_not_a_sized_vector_is_refused(declared: str):
    with pytest.raises(IngestError, match="vector"):
        dimension_of(declared)


def test_word_stats_report_the_shortest_the_middle_and_the_longest_chunk():
    stats = word_stats(["one two three", "one", "one two three four five"])

    assert (stats.minimum, stats.median, stats.maximum) == (1, 3, 5)


def test_the_median_of_an_even_number_of_chunks_is_the_average_of_the_middle_two():
    stats = word_stats(["one", "one two", "one two three", "one two three four"])

    assert stats.median == 2.5


def test_word_stats_of_a_document_with_no_chunks_are_zero():
    stats = word_stats([])

    assert (stats.minimum, stats.median, stats.maximum) == (0, 0, 0)


def test_chunks_are_embedded_in_batches_and_come_back_in_their_own_order():
    embedder = RecordingEmbedder()
    texts = [f"chunk {index}" for index in range(5)]

    vectors = embed_in_batches(embedder, texts, batch_size=2)

    assert embedder.batch_sizes == [2, 2, 1]
    assert vectors == [FakeEmbedder().embed_query(text) for text in texts]


def test_an_embedder_that_loses_a_vector_is_refused():
    class ForgetfulEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents(texts)[:-1]

    with pytest.raises(IngestError, match="3 chunks"):
        embed_in_batches(ForgetfulEmbedder(), ["a", "b", "c"], batch_size=3)


# --- the whole run ----------------------------------------------------------------------


@pytest.mark.integration
def test_ingest_writes_the_document_its_pages_and_its_chunks(
    pool: ConnectionPool, handbook: tuple[Edition, Path]
):
    edition, path = handbook

    summary = ingest(pool, FakeEmbedder(), edition, "fixed-220w", path=path)

    assert (summary.page_count, summary.chunk_count) == (3, EXPECTED_CHUNKS)
    assert [summary.words.minimum, summary.words.maximum] == [90, 220]
    assert counts(pool) == (1, 3, EXPECTED_CHUNKS)

    with pool.connection() as connection:
        title, source_url, sha256, page_count = connection.execute(
            "SELECT title, source_url, sha256, page_count FROM documents"
        ).fetchone()
        chunks = connection.execute(
            "SELECT ordinal, page_start, page_end, section, text, tsv FROM chunks ORDER BY ordinal"
        ).fetchall()

    assert (title, source_url, sha256, page_count) == (
        "Medicare & You 2026",
        edition.primary_url,
        edition.sha256,
        3,
    )
    page_ranges = [(page_start, page_end) for _, page_start, page_end, *_ in chunks]
    assert page_ranges == EXPECTED_PAGE_RANGES
    assert [len(text.split()) for *_, text, _ in chunks] == EXPECTED_WORD_COUNTS
    assert all(section is None for *_, section, _, _ in chunks), "part 2's chunker fills sections"
    assert chunks[0][-1], "Postgres fills tsv from text on insert, so word search has an index"


@pytest.mark.integration
def test_a_second_run_replaces_the_rows_rather_than_adding_to_them(
    pool: ConnectionPool, handbook: tuple[Edition, Path]
):
    """Ingest is safe to re-run: the second run leaves the same rows, not twice as many."""
    edition, path = handbook
    first = ingest(pool, FakeEmbedder(), edition, "fixed-220w", path=path)
    with pool.connection() as connection:
        document_id = connection.execute("SELECT id FROM documents").fetchone()[0]

    second = ingest(pool, FakeEmbedder(), edition, "fixed-220w", path=path)

    assert second == first
    assert counts(pool) == (1, 3, EXPECTED_CHUNKS)
    with pool.connection() as connection:
        assert connection.execute("SELECT id FROM documents").fetchone()[0] == document_id


@pytest.mark.integration
def test_a_second_chunk_set_lives_beside_the_first(
    pool: ConnectionPool, handbook: tuple[Edition, Path], monkeypatch
):
    """Part 2 compares chunkers side by side, so re-ingesting one set must not drop the other."""
    from fineprint import ingest as ingest_module

    edition, path = handbook
    ingest(pool, FakeEmbedder(), edition, "fixed-220w", path=path)
    monkeypatch.setitem(ingest_module.CHUNKERS, "half-page", ingest_module.CHUNKERS["fixed-220w"])

    ingest(pool, FakeEmbedder(), edition, "half-page", path=path)

    assert counts(pool) == (1, 3, 2 * EXPECTED_CHUNKS)


@pytest.mark.integration
def test_an_embedder_of_the_wrong_width_is_refused_before_anything_is_written(
    pool: ConnectionPool, handbook: tuple[Edition, Path]
):
    edition, path = handbook

    with pytest.raises(IngestError) as excinfo:
        ingest(pool, NarrowEmbedder(), edition, "fixed-220w", path=path)

    assert "384" in str(excinfo.value) and "1024" in str(excinfo.value)
    assert counts(pool) == (0, 0, 0)


@pytest.mark.integration
def test_a_pdf_that_is_not_the_pinned_bytes_is_refused(
    pool: ConnectionPool, handbook: tuple[Edition, Path]
):
    """The corpus is one revision of one edition; anything else would answer from other numbers."""
    edition, path = handbook
    tampered = replace(edition, sha256="0" * 64)

    with pytest.raises(IngestError, match="sha256"):
        ingest(pool, FakeEmbedder(), tampered, "fixed-220w", path=path)

    assert counts(pool) == (0, 0, 0)


@pytest.mark.integration
def test_a_missing_pdf_says_how_to_fetch_it(pool: ConnectionPool, handbook: tuple[Edition, Path]):
    edition, path = handbook

    with pytest.raises(IngestError, match="download_handbook"):
        ingest(pool, FakeEmbedder(), edition, "fixed-220w", path=path.with_name("absent.pdf"))


@pytest.mark.integration
def test_an_unknown_chunk_set_is_refused(pool: ConnectionPool, handbook: tuple[Edition, Path]):
    edition, path = handbook

    with pytest.raises(IngestError, match="fixed-220w"):
        ingest(pool, FakeEmbedder(), edition, "sentences", path=path)
