"""Retrieval against a real Postgres: every test here carries the `integration` marker.

The rows are written with plain SQL rather than through the ingest command, so these tests
exercise retrieval and nothing else. The fake embedder stands in for Bedrock, and because it
turns a text into the same vector every time, embedding a chunk's own text is a question that
lands exactly on that chunk: a query built to favour it.
"""

import psycopg
import pytest
from pgvector import Vector
from psycopg_pool import ConnectionPool

from fineprint.config import Settings
from fineprint.retrieval import EditionNotIngestedError, Retriever
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

EDITION = 2026
CHUNK_SET = "fixed-220w"

# Four passages in the handbook's voice, each with the pages it would have come from.
CHUNKS: dict[str, tuple[int, int, str]] = {
    "premium": (
        23,
        23,
        "The standard Part B premium amount in 2026 is $202.90 each month. Most people pay "
        "the standard Part B premium amount. If your modified adjusted gross income is above "
        "a certain amount, you may pay more.",
    ),
    "hospital": (
        28,
        29,
        "In 2026 you pay a $1,736 deductible for each inpatient benefit period before "
        "Original Medicare starts to pay. For days 1 through 60 of a stay there is no "
        "coinsurance once the deductible is met.",
    ),
    "helpline": (
        110,
        110,
        "Call 1-800-MEDICARE (1-800-633-4227) to ask about your coverage or to file a "
        "complaint. TTY users can call 1-877-486-2048.",
    ),
    "drugs": (
        85,
        85,
        "Medicare drug coverage, Part D, helps pay for prescription drugs. In 2026 what you "
        "pay out of pocket for covered drugs is capped once you reach the yearly limit.",
    ),
}

# A passage in another edition and another chunk set, holding the same words as the premium
# passage. Nothing that searches the 2026 fixed-220w set may ever return one of these.
DECOY_TEXT = "The standard Part B premium amount is $174.70 each month in this other printing."


def insert_document(connection: psycopg.Connection, edition: int) -> int:
    """One row in `documents`, and its id."""
    return connection.execute(
        "INSERT INTO documents (edition, title, source_url, sha256, page_count)"
        " VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (edition, f"Medicare & You {edition}", "https://example.invalid/h.pdf", "0" * 64, 128),
    ).fetchone()[0]


def insert_chunk(
    connection: psycopg.Connection,
    document_id: int,
    chunk_set: str,
    ordinal: int,
    pages: tuple[int, int],
    text: str,
) -> int:
    """One row in `chunks`, embedded with the fake embedder, and its id."""
    embedding = FakeEmbedder().embed_documents([text])[0]
    return connection.execute(
        "INSERT INTO chunks (document_id, chunk_set, ordinal, page_start, page_end, text,"
        " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (document_id, chunk_set, ordinal, pages[0], pages[1], text, Vector(embedding)),
    ).fetchone()[0]


@pytest.fixture
def chunk_ids(pool: ConnectionPool) -> dict[str, int]:
    """Load the four passages, plus the two decoys, and return the id of each passage."""
    with pool.connection() as connection:
        document_id = insert_document(connection, EDITION)
        ids = {
            label: insert_chunk(connection, document_id, CHUNK_SET, ordinal, (start, end), text)
            for ordinal, (label, (start, end, text)) in enumerate(CHUNKS.items())
        }
        insert_chunk(connection, document_id, "other-chunker", 0, (23, 23), DECOY_TEXT)
        older = insert_document(connection, 2025)
        insert_chunk(connection, older, CHUNK_SET, 0, (23, 23), DECOY_TEXT)
    return ids


@pytest.fixture
def settings(initialized_database: str) -> Settings:
    return Settings(database_url=initialized_database, corpus_edition=EDITION, chunk_set=CHUNK_SET)


@pytest.fixture
def retriever(pool: ConnectionPool, settings: Settings, chunk_ids: dict[str, int]) -> Retriever:
    return Retriever(pool, FakeEmbedder(), settings)


def test_a_full_question_finds_the_passage_that_shares_its_words(
    retriever: Retriever, chunk_ids: dict[str, int]
):
    hits = retriever.search_lexical("How much is the standard Part B premium in 2026?")

    assert hits, "a question in the handbook's own vocabulary must match something"
    assert hits[0].chunk_id == chunk_ids["premium"]
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_a_word_the_handbook_never_uses_does_not_wipe_out_the_results(
    retriever: Retriever, chunk_ids: dict[str, int]
):
    """The query is an OR of the question's lexemes, so one unknown word costs nothing.

    Were it an AND query, "zebra" alone would turn a good question into no results at all.
    """
    hits = retriever.search_lexical("How much is the zebra Part B premium?")

    assert [hit.chunk_id for hit in hits][:1] == [chunk_ids["premium"]]


@pytest.mark.parametrize(
    "question",
    [
        "What is Medicare's Part B premium?",
        "Is the Part B premium $202.90 a month?",
        "Part B: what does it cost?",
        "Who answers at 1-800-MEDICARE?",
        "Does 'Part B' & !coverage mean anything here?",
    ],
    ids=["apostrophe", "dollar", "colon", "phone-number", "tsquery-operators"],
)
def test_punctuation_in_a_question_is_never_read_as_query_syntax(
    retriever: Retriever, question: str
):
    """Every lexeme is quoted before the cast to tsquery, so none of these raises."""
    assert isinstance(retriever.search_lexical(question), list)


def test_the_helpline_number_is_found_by_its_own_words(
    retriever: Retriever, chunk_ids: dict[str, int]
):
    """A phone number survives stemming and quoting, which is lexical search's whole job."""
    hits = retriever.search_lexical("Who answers at 1-800-MEDICARE?")

    assert hits[0].chunk_id == chunk_ids["helpline"]


def test_a_question_of_only_stop_words_finds_nothing(retriever: Retriever):
    """No lexemes means a NULL tsquery, which matches no row: an empty list, not an error."""
    assert retriever.search_lexical("What is it about?") == []
    assert retriever.search("What is it about?", mode="lexical") == []


@pytest.mark.parametrize("mode", ["hybrid", "vector", "lexical"])
def test_every_mode_puts_the_expected_chunk_first(
    retriever: Retriever, chunk_ids: dict[str, int], mode: str
):
    """The question is the premium passage itself: its own words and its own vector."""
    results = retriever.search(CHUNKS["premium"][2], mode=mode)

    assert results[0].chunk_id == chunk_ids["premium"]
    assert results[0].rank == 1
    assert results[0].page_start == 23
    assert results[0].text.startswith("The standard Part B premium")
    assert results[0].lexical_rank == (None if mode == "vector" else 1)
    assert results[0].vector_rank == (None if mode == "lexical" else 1)


def test_the_vector_score_is_one_minus_the_cosine_distance(retriever: Retriever):
    """A question identical to a chunk sits on top of it, so the similarity is 1."""
    results = retriever.search(CHUNKS["premium"][2], mode="vector")

    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_hybrid_keeps_a_chunk_only_one_retriever_found(
    retriever: Retriever, chunk_ids: dict[str, int]
):
    """Vector search returns every chunk; lexical search returns only the word matches."""
    results = retriever.search("What is the inpatient deductible?", mode="hybrid")
    by_id = {result.chunk_id: result for result in results}

    assert by_id[chunk_ids["hospital"]].lexical_rank is not None
    assert any(result.lexical_rank is None and result.vector_rank is not None for result in results)


def test_only_the_configured_edition_and_chunk_set_are_searched(
    retriever: Retriever, chunk_ids: dict[str, int]
):
    """The decoys hold the same words in another chunk set and another edition."""
    for mode in ("hybrid", "vector", "lexical"):
        found = {result.chunk_id for result in retriever.search(DECOY_TEXT, mode=mode, top_k=20)}
        assert found <= set(chunk_ids.values()), f"{mode} reached outside its corpus"


def test_top_k_decides_how_many_results_come_back(retriever: Retriever):
    assert len(retriever.search("Part B premium", top_k=2)) == 2


def test_top_k_defaults_to_the_configured_number(
    pool: ConnectionPool, settings: Settings, chunk_ids: dict[str, int]
):
    retriever = Retriever(pool, FakeEmbedder(), settings.model_copy(update={"retrieval_top_k": 3}))

    assert len(retriever.search("Part B premium")) == 3


def test_an_unknown_mode_is_refused(retriever: Retriever):
    with pytest.raises(ValueError, match="mode must be one of"):
        retriever.search("Part B premium", mode="fuzzy")


def test_an_edition_that_was_never_ingested_says_how_to_ingest_it(
    pool: ConnectionPool, settings: Settings, chunk_ids: dict[str, int]
):
    retriever = Retriever(
        pool, FakeEmbedder(), settings.model_copy(update={"corpus_edition": 1999})
    )

    with pytest.raises(EditionNotIngestedError) as excinfo:
        retriever.search("Part B premium")

    message = str(excinfo.value)
    assert "1999" in message
    assert "fineprint ingest --edition 1999" in message
