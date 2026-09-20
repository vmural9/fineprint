-- The whole database, applied by `fineprint init-db`.
--
-- Every statement says IF NOT EXISTS, so running init-db on a database that already has the
-- schema changes nothing and reports no error.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per handbook edition. The sha256 is the pinned hash the download script verifies, so
-- a stored answer can always be traced back to the exact bytes it was read from.
CREATE TABLE IF NOT EXISTS documents (
    id          serial PRIMARY KEY,
    edition     int  NOT NULL UNIQUE,          -- 2026
    title       text NOT NULL,
    source_url  text NOT NULL,
    sha256      text NOT NULL,
    page_count  int  NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

-- The extracted text of one page, kept whole so a citation can be checked against the page it
-- claims to come from.
CREATE TABLE IF NOT EXISTS pages (
    document_id int  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number int  NOT NULL,                 -- see "Corpus facts" for the numbering rule
    text        text NOT NULL,
    PRIMARY KEY (document_id, page_number)
);

-- What retrieval actually searches. Each chunk carries both of its search representations:
-- `tsv` for word search and `embedding` for meaning search. Postgres keeps `tsv` in step with
-- `text` by itself; `embedding` is written by the ingest command.
CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id int  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_set   text NOT NULL,                 -- 'fixed-220w'; one set per chunking configuration
    ordinal     int  NOT NULL,                 -- position within the document
    page_start  int  NOT NULL,
    page_end    int  NOT NULL,
    section     text,                          -- NULL in part 1; the part 2 chunker fills it
    text        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding   vector(1024) NOT NULL,         -- the width of amazon.titan-embed-text-v2:0
    UNIQUE (document_id, chunk_set, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- No approximate vector index in part 1: a few hundred rows scan exactly in milliseconds, and
-- exact results keep evals deterministic. When the table grows, add:
--   CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
-- and set hnsw.iterative_scan, because every query also filters by document and chunk set.
