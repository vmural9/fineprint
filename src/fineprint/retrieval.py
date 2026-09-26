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

*Re-ranking* is an optional last step in hybrid mode. Neither retriever reads a chunk beside the
question: one counts shared words, the other compares two vectors made separately. A re-ranker
reads the two together, so it can tell the passage that answers the question from one that only
talks about the same thing. Reading every pair costs far more than comparing vectors, so it reads
only the first `RERANK_CANDIDATES` of the fused list, and its order decides which `top_k` are kept.

The ranking Postgres does here is its own built-in `ts_rank_cd`, not BM25 (see decision D3), so
everything in this module calls it "lexical" and never "BM25".
"""

from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

from pgvector import Vector
from psycopg_pool import ConnectionPool

from fineprint.config import Settings
from fineprint.db import describe_database
from fineprint.providers.base import Embedder, Reranker

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

CHUNK_BODIES_SQL = (
    "SELECT id, ordinal, section, page_start, page_end, text FROM chunks WHERE id = ANY(%s)"
)


class Hit(NamedTuple):
    """One row from one retriever: the chunk, its 1-based place in that list, and its score."""

    chunk_id: int
    rank: int
    score: float


class ChunkBody(NamedTuple):
    """The stored columns a result carries besides its ranks."""

    ordinal: int
    section: str | None
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk a search returned, with where it stood in each list that found it.

    `ordinal` is the chunk's place in its chunk set, which survives a re-ingest where
    `chunk_id` does not, and `section` is the heading it sits under when its chunker records one.

    `score` is the re-ranker's score when a re-ranker ran, the fused score in hybrid mode
    otherwise, and the retriever's own score in a single mode. `rank` is the chunk's place in
    the list that was returned. `lexical_rank` and `vector_rank` are None when that retriever
    did not return this chunk — which is always the case for the retriever a single-mode
    search never ran. `fused_rank` is its place after fusion and before any re-ranking, so it
    is None outside hybrid mode and equals `rank` when nothing re-ranked the list;
    `rerank_score` is None when no re-ranker ran. Keeping every rank is what lets `/search`
    show a reader why a passage is here, and how far the re-ranker moved it.
    """

    chunk_id: int
    ordinal: int
    section: str | None
    page_start: int
    page_end: int
    text: str
    score: float
    rank: int
    lexical_rank: int | None
    vector_rank: int | None
    fused_rank: int | None
    rerank_score: float | None


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

    Holds the connection pool, the embedder that turns a question into a vector, the settings
    that say which edition and which chunk set every query is restricted to, and the re-ranker
    that re-orders hybrid results, if one is configured.
    """

    def __init__(
        self,
        pool: ConnectionPool,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker
        self._document_id: int | None = None

    @property
    def document_id(self) -> int:
        """The `documents.id` of the configured edition, looked up once and remembered.

        Resolved on first use rather than in `__init__` so that building a `Retriever` never
        touches the database, which keeps the API's startup and the tests simple. It is kept
        for the life of the object, so a long-running process that re-ingests the edition
        underneath itself needs a restart to see the new row.
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

        Without a re-ranker, these are the first `top_k` of `candidates()`. With one, in hybrid
        mode, the re-ranker reads the question beside each of the first `RERANK_CANDIDATES`
        and the best `top_k` are kept in its order. Each keeps its fused rank, and its score
        becomes the re-ranker's. A `top_k` above `RERANK_CANDIDATES` widens what it reads to
        `top_k`, so a search never returns fewer than it was asked for while the pool has them.
        `vector` and `lexical` are never re-ranked: they exist to measure one retriever on its
        own.
        """
        kept = self.settings.retrieval_top_k if top_k is None else top_k
        ordered = self.candidates(question, mode)
        if self.reranker is None or mode != "hybrid":
            return ordered[:kept]

        shortlist = ordered[: max(self.settings.rerank_candidates, kept)]
        reranked = self.reranker.rerank(question, [chunk.text for chunk in shortlist], top_n=kept)
        return [
            replace(
                shortlist[result.index], score=result.score, rank=rank, rerank_score=result.score
            )
            for rank, result in enumerate(reranked[:kept], start=1)
        ]

    def candidates(self, question: str, mode: SearchMode = "hybrid") -> list[RetrievedChunk]:
        """Every chunk the search found, in order, with its text: the pool `search` cuts from.

        Each retriever is asked for `RETRIEVAL_CANDIDATES` rows. `hybrid` runs both and fuses
        them; `vector` and `lexical` run one and keep its own scores. Nothing is re-ranked or
        cut here, so a caller can see where a chunk stood even when it missed the top `k`.
        `rank` is the place in this pool, and in hybrid mode `fused_rank` is the same.
        """
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {', '.join(SEARCH_MODES)}, not {mode!r}")

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

        lexical_ranks = {hit.chunk_id: hit.rank for hit in lexical}
        vector_ranks = {hit.chunk_id: hit.rank for hit in vector}
        bodies = self._load_chunk_bodies([chunk_id for chunk_id, _ in ordered])

        results: list[RetrievedChunk] = []
        for chunk_id, score in ordered:
            body = bodies.get(chunk_id)
            if body is None:
                # A re-ingest deleted this chunk after the ranking query saw it. Leave it out;
                # the ones after it move up, so ranks stay 1, 2, 3 with no gap.
                continue
            rank = len(results) + 1
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    ordinal=body.ordinal,
                    section=body.section,
                    page_start=body.page_start,
                    page_end=body.page_end,
                    text=body.text,
                    score=score,
                    rank=rank,
                    lexical_rank=lexical_ranks.get(chunk_id),
                    vector_rank=vector_ranks.get(chunk_id),
                    fused_rank=rank if mode == "hybrid" else None,
                    rerank_score=None,
                )
            )
        return results

    def _load_chunk_bodies(self, chunk_ids: list[int]) -> dict[int, ChunkBody]:
        """The stored columns of these chunks, in one query, keyed by id.

        An id the table no longer holds is simply absent from the result.
        """
        if not chunk_ids:
            return {}
        with self.pool.connection() as connection:
            rows = connection.execute(CHUNK_BODIES_SQL, (chunk_ids,)).fetchall()
        return {
            chunk_id: ChunkBody(ordinal, section, page_start, page_end, text)
            for chunk_id, ordinal, section, page_start, page_end, text in rows
        }
