"""Finding the handbook passages a question is about.

Two searches run over the same `chunks` table and are good at different things.

*Lexical search* matches the question's words against the words in each chunk, after Postgres
has stemmed both sides. It is literal: it is the one that reliably finds "$202.90" or
"1-800-MEDICARE", and the one that finds nothing when the reader's words are not the handbook's
words. "What will I owe if I go to the hospital?" shares almost no words with a passage about
the inpatient hospital deductible.

*Vector search* compares the question's embedding with each chunk's embedding, so it matches on
meaning and answers that hospital question. It is vague about exact strings: asked for the Part B
premium it is just as happy with the Part A premium, because the two passages mean nearly the
same thing.

*Hybrid search* runs both and merges the two ranked lists with reciprocal rank fusion, which
reads only each chunk's position in each list. That is deliberate: `ts_rank_cd` scores and cosine
similarities are on different scales, so adding them would be meaningless, while positions are
comparable by construction.

The ranking Postgres does here is its own built-in `ts_rank_cd`, not BM25 (see decision D3), so
everything in this module calls it "lexical" and never "BM25".
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

from pgvector import Vector
from psycopg_pool import ConnectionPool

from fineprint.config import Settings
from fineprint.db import describe_database
from fineprint.providers.base import Embedder

SearchMode = Literal["hybrid", "vector", "lexical"]
SEARCH_MODES: tuple[SearchMode, ...] = ("hybrid", "vector", "lexical")

# Lexical search over one document's chunks.
#
# `to_tsvector` turns the question into the same stemmed lexemes the stored `tsv` column holds,
# and `unnest` spreads them over rows. `string_agg(quote_literal(lexeme), ' | ')` joins them into
# an OR query: a question is not an AND query, because then one word the handbook never uses
# would return nothing at all. The `::tsquery` cast takes those lexemes as they are, so they are
# not stemmed a second time, and `quote_literal` is what keeps a lexeme like `-800` (from
# 1-800-MEDICARE) or a stray `&` from being read as tsquery syntax.
#
# A question made only of stop words has no lexemes, so `string_agg` returns NULL, `tsv @@ NULL`
# is NULL, and no row matches: an empty result, which is the honest answer.
LEXICAL_SQL = """
WITH q AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
    FROM unnest(to_tsvector('english', %(question)s))
)
SELECT c.id, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks AS c, q
WHERE c.document_id = %(document_id)s AND c.chunk_set = %(chunk_set)s AND c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(limit)s
"""

# Vector search over the same chunks. `<=>` is pgvector's cosine distance: 0 for vectors pointing
# the same way, 2 for opposite ones. Reporting `1 - distance` turns it into a similarity that
# grows with relevance, like the lexical score, so both retrievers are read the same way.
VECTOR_SQL = """
SELECT id, 1 - (embedding <=> %(query_vector)s) AS score
FROM chunks
WHERE document_id = %(document_id)s AND chunk_set = %(chunk_set)s
ORDER BY embedding <=> %(query_vector)s, id
LIMIT %(limit)s
"""

CHUNK_BODIES_SQL = "SELECT id, page_start, page_end, text FROM chunks WHERE id = ANY(%s)"


class Hit(NamedTuple):
    """One row from one retriever: the chunk, its 1-based place in that list, and its score."""

    chunk_id: int
    rank: int
    score: float


class ChunkBody(NamedTuple):
    """The stored columns a result carries besides its ranks."""

    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk a search returned, with where it stood in each list that found it.

    `score` is the fused score in hybrid mode and the retriever's own score otherwise.
    `lexical_rank` and `vector_rank` are None when that retriever did not return this chunk —
    which is always the case for the retriever a single-mode search never ran. Keeping both is
    what lets `/search` show a reader why a passage is here.
    """

    chunk_id: int
    page_start: int
    page_end: int
    text: str
    score: float
    rank: int
    lexical_rank: int | None
    vector_rank: int | None


class EditionNotIngestedError(RuntimeError):
    """`CORPUS_EDITION` names an edition that is not in the database."""


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists of chunk ids into one list, using positions and never scores.

    Reciprocal rank fusion gives a chunk `1 / (k + rank)` for every list it appears in, with
    ranks starting at 1, and orders chunks by the sum. Chunks both retrievers found therefore
    rise above chunks only one of them found. The constant `k` (60 by default, the value the
    original paper used) flattens the difference between the top few places, so being first
    rather than third in one list does not decide the outcome on its own.

    Pure: no database, no configuration. Ties are broken by chunk id so the order is stable.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda scored: (-scored[1], scored[0]))


class Retriever:
    """Searches one edition's chunks, by words, by meaning, or by both.

    Holds the connection pool, the embedder that turns a question into a vector, and the
    settings that say which edition and which chunk set every query is restricted to.
    """

    def __init__(self, pool: ConnectionPool, embedder: Embedder, settings: Settings) -> None:
        self.pool = pool
        self.embedder = embedder
        self.settings = settings
        self._document_id: int | None = None

    @property
    def document_id(self) -> int:
        """The `documents.id` of the configured edition, looked up once and remembered.

        Resolved on first use rather than in `__init__` so that building a `Retriever` never
        touches the database, which keeps the API's startup and the tests simple.
        """
        if self._document_id is None:
            self._document_id = self._load_document_id()
        return self._document_id

    def _load_document_id(self) -> int:
        edition = self.settings.corpus_edition
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT id FROM documents WHERE edition = %s", (edition,)
            ).fetchone()
        if row is None:
            raise EditionNotIngestedError(
                f"The {edition} edition of the handbook has not been ingested into "
                f"{describe_database(self.settings.database_url)}. "
                f"Load it with: fineprint ingest --edition {edition}"
            )
        return row[0]

    def search_lexical(self, question: str, limit: int | None = None) -> list[Hit]:
        """The chunks whose words match the question's, best first.

        Empty when the question has no lexemes of its own — "What is it about?" is nothing but
        stop words, and stop words are not in the index.
        """
        parameters = {
            "question": question,
            "document_id": self.document_id,
            "chunk_set": self.settings.chunk_set,
            "limit": limit if limit is not None else self.settings.retrieval_candidates,
        }
        with self.pool.connection() as connection:
            rows = connection.execute(LEXICAL_SQL, parameters).fetchall()
        return [
            Hit(chunk_id, rank, float(score))
            for rank, (chunk_id, score) in enumerate(rows, start=1)
        ]

    def search_vector(self, question: str, limit: int | None = None) -> list[Hit]:
        """The chunks whose meaning is closest to the question's, nearest first.

        Embedding the question is a Bedrock call, so this costs money on every search.
        """
        parameters = {
            "query_vector": Vector(self.embedder.embed_query(question)),
            "document_id": self.document_id,
            "chunk_set": self.settings.chunk_set,
            "limit": limit if limit is not None else self.settings.retrieval_candidates,
        }
        with self.pool.connection() as connection:
            rows = connection.execute(VECTOR_SQL, parameters).fetchall()
        return [
            Hit(chunk_id, rank, float(score))
            for rank, (chunk_id, score) in enumerate(rows, start=1)
        ]

    def search(
        self, question: str, mode: SearchMode = "hybrid", top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """The best `top_k` chunks for a question.

        Each retriever is asked for `RETRIEVAL_CANDIDATES` rows so fusion has something to work
        with, and only the survivors are read out of the table. `vector` and `lexical` run one
        retriever and return its own scores; `hybrid` runs both and fuses them.
        """
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {', '.join(SEARCH_MODES)}, not {mode!r}")
        kept = self.settings.retrieval_top_k if top_k is None else top_k

        lexical = self.search_lexical(question) if mode in ("hybrid", "lexical") else []
        vector = self.search_vector(question) if mode in ("hybrid", "vector") else []

        if mode == "hybrid":
            ordered = rrf_fuse(
                [[hit.chunk_id for hit in lexical], [hit.chunk_id for hit in vector]],
                k=self.settings.rrf_k,
            )
        else:
            hits = lexical if mode == "lexical" else vector
            ordered = [(hit.chunk_id, hit.score) for hit in hits]
        ordered = ordered[:kept]

        lexical_ranks = {hit.chunk_id: hit.rank for hit in lexical}
        vector_ranks = {hit.chunk_id: hit.rank for hit in vector}
        bodies = self._load_chunk_bodies([chunk_id for chunk_id, _ in ordered])
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                page_start=bodies[chunk_id].page_start,
                page_end=bodies[chunk_id].page_end,
                text=bodies[chunk_id].text,
                score=score,
                rank=rank,
                lexical_rank=lexical_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
            )
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]

    def _load_chunk_bodies(self, chunk_ids: list[int]) -> dict[int, ChunkBody]:
        """Pages and text for the chunks that survived, in one query."""
        if not chunk_ids:
            return {}
        with self.pool.connection() as connection:
            rows = connection.execute(CHUNK_BODIES_SQL, (chunk_ids,)).fetchall()
        return {
            chunk_id: ChunkBody(page_start, page_end, text)
            for chunk_id, page_start, page_end, text in rows
        }
