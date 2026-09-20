# Part 1: RAG service (build spec)

This is the complete build spec for part 1 of the series described in [docs/PLAN.md](../PLAN.md).
It is written so that a session with no other context can build part 1 from it. Read `CLAUDE.md`
first; its rules apply to every task below.

- **Tag at the end of this part:** `part-1`
- **Status:** not started. The setup session (2026-09-21) created the repo skeleton, the handbook
  download script, and the first 15 golden-set questions.

## What part 1 delivers

A working question-answering service over the handbook, plus the first measured row of the
scoreboard.

- Ingestion: PDF to pages, pages to chunks, chunks to embeddings, all stored in Postgres 16 with
  pgvector.
- Hybrid retrieval: Postgres full-text search plus vector similarity, fused with reciprocal rank
  fusion (RRF).
- A FastAPI service with `POST /search`, `POST /ask`, and `GET /healthz`.
- Answer generation that returns a Pydantic-validated structure: the answer, the cited chunks with
  page numbers, and a confidence label.
- Golden set v1: about 40 questions in `evals/golden_set.jsonl`.
- An eval CLI that runs the golden set, stores the raw results, and generates
  `evals/scoreboard.md`. Part 1 fills two headline columns: page hit rate and manual pass rate.
- A README section for part 1, the tag, and the walkthrough video.

**Out of scope, on purpose.** RAGAS, structure-aware chunking, and the re-ranker are part 2.
Tracing, the tool-using agent, retries and fallback, and the cost and latency columns are part 3.
The LLM judge, the adversarial slice, promptfoo, and the CI gate are part 4. Part 1 only leaves
clean seams for them (see "Seams for later parts").

## Corpus facts

Established by the setup session from the PDF itself, not from memory. Re-check any of them with
`scripts/download_handbook.py` and pypdf.

- **Edition.** Medicare & You 2026, the January 2026 printing. CMS revises the handbook during
  the year: the September 2025 printing of the 2026 edition still prints 2025 dollar amounts (a
  $185 Part B premium), and only the January 2026 printing prints the final 2026 amounts. The
  golden set's expected answers match the pinned printing and no other.
- **Source.** medicare.gov serves the handbook at a single unversioned URL, which has served the
  2027 edition since September 2026, and CMS keeps no per-year archive. So
  `scripts/download_handbook.py` fetches the pinned bytes from the Internet Archive (a primary and
  a fallback snapshot) and verifies sha256
  `d7a341bc3d2d3dab59af746a0875752761e6c2f2dc107d95e9078a23933513d6` (4,064,150 bytes). The 2025
  edition is pinned the same way behind `--edition 2025`, ready for part 4.
- **Pages.** 128 PDF pages. The printed page number equals the 1-based PDF page index for pages 2
  to 126, with no exceptions; page 1 is the cover, 127 is blank, and 128 is the back cover, and
  none of those three carries a printed number. So `page_number`, `page_start`, `page_end`, and
  `expected_pages` all mean the PDF page index, which is also the number a reader sees on the
  page. The PDF has no `/PageLabels` entry, so `pypdf.page_labels` returns a synthesised `1..128`;
  it happens to be right but is not authoritative. Do not rely on it.
- **Section map.** Printed pages, which are also PDF indices: cover 1; What's new & important? 2;
  Contents 3; Index of topics 4–8; What are the parts of Medicare? 9; Your Medicare options 10; At
  a glance: Original Medicare vs. Medicare Advantage Plan 11–12; Get started with Medicare 13–14;
  Section 1 Signing up for Medicare 15–24; Section 2 Find out what Medicare covers 25–56; Section
  3 Original Medicare 57–60; Section 4 Medicare Advantage Plans & other options 61–74; Section 5
  Medicare Supplement Insurance (Medigap) 75–78; Section 6 Medicare drug coverage (Part D) 79–90;
  Section 7 Get help paying your health & drug costs 91–96; Section 8 Your Medicare rights &
  protections 97–106; Section 9 Find helpful contacts and more information 107–118; Section 10
  Definitions 119–122; Nondiscrimination Notice 123; Accessible Communications 124; Looking for
  help in other languages? 125–126; blank 127; back cover 128. Pages 96 and 118 are "Notes"
  filler. The PDF outline (one root entry, 22 children, no deeper nesting) agrees with the table
  of contents on every shared entry. Part 2's structure-aware chunker starts here.
- **Extraction library.** pypdf, already a dependency because the golden-set verifier uses it.
  pdfplumber interleaves multi-column pages line by line, which scrambles the comparison tables
  and the index; pypdf keeps cells and entries contiguous.
- **Known noise.** Running headers and page numbers appear in the extracted text of every body
  page, and on pages 123 to 126 pypdf emits the folio doubled (`123123`). pypdf also emits the
  right column before the left on the index pages, 4 to 8. Part 1 leaves all of it in, along with
  the table of contents and the index, as part of the naive baseline.
- **Availability risk.** The download depends on the Internet Archive, which during setup
  intermittently returned its "Temporarily Offline" page instead of the file, roughly one request
  in three at one point. The script's fallback URL covered it. CI in part 4 should cache the file
  and retry once before treating a failed download as a failure.

## Decisions waiting for the owner

The setup session proposed these defaults. Do not start the task named in the last column until
the owner has confirmed or changed the decision. Record the outcome by editing this table.

| # | Decision | Proposed default | If the owner chooses differently | Blocks |
|---|----------|------------------|----------------------------------|--------|
| D1 | Generation provider and model | Anthropic, `claude-opus-5`, set by config | Swap one provider module and two config defaults | Task 7 |
| D2 | Embeddings. Anthropic has no embeddings endpoint, so "one provider" cannot cover both calls. | Local model `BAAI/bge-small-en-v1.5` (384 dimensions) run through `fastembed`. Retrieval, ingestion, and every retrieval-only eval then work with no API key and no cost. `fastembed` runs the model with onnxruntime and downloads the weights from Hugging Face on first use, so CI must cache them. | Voyage AI or OpenAI embeddings: one more key, a different vector dimension in `schema.sql`, nothing else | Task 4 |
| D3 | The brief says "BM25 via Postgres full-text search". Built-in Postgres ranking (`ts_rank_cd`) is not BM25; real BM25 needs an extension that the `pgvector/pgvector:pg16` image does not ship. | Use built-in full-text search in part 1 and call it "lexical (Postgres FTS)" everywhere. Make true BM25 one of the side-by-side experiments in part 2. | Switch the database image to one that bundles a BM25 extension and pgvector | Task 6 |
| D4 | Vector index | None in part 1. The corpus is a few hundred chunks, so an exact scan is fast and keeps evals deterministic. The HNSW statement stays in `schema.sql` as a comment with the reason. | Create the index, and set `hnsw.iterative_scan` because every query filters by chunk set | Task 1 |
| D5 | Golden-set fields beyond the brief's six plus `needs_review` | `type`, `evidence`, `alt_pages` (see "Golden set") | Drop the field from the file, the loader, and the verifier | Task 9 |
| D6 | Answer schema carries `found_in_handbook` in addition to answer, citations, confidence | Included, so abstention on unanswerable questions is measured without a judge | Remove the field and the abstention metric | Task 7 |
| D7 | Headline scoreboard columns for the whole series | The eleven columns under "Scoreboard format" | Edit the column list in `evals/scoreboard.py`; old results still render | Task 10 |
| D8 | Corpus edition. The brief pins 2026, and CMS has since published 2027. | Stay on 2026 as briefed; part 4 swaps 2025 for 2026 as planned. | Pin the 2027 file in the download script, re-verify the golden set against it, and make the part 4 swap 2026 to 2027 | Task 5, Task 9 |

## Architecture

```
handbook.pdf ──extract──▶ pages ──chunk──▶ chunks ──embed──▶ Postgres (tsvector + pgvector)

question ─┬─▶ embed ─────▶ vector top N ──┐
          └─▶ lexemes ───▶ lexical top N ─┴─▶ RRF ─▶ top k chunks ─▶ prompt ─▶ LLM ─▶ AnswerResponse
```

Everything is plain Python and plain SQL. There is no ORM and no RAG framework, so a reader can
see each step.

| Module (`src/fineprint/`) | Responsibility |
|-----------------------------|----------------|
| `config.py` | `Settings` (pydantic-settings), loaded from the environment and `.env` |
| `db.py` | psycopg 3 connection pool, pgvector type registration, `init_db()` that applies `schema.sql` |
| `schema.sql` | All DDL, idempotent |
| `pdf.py` | `extract_pages(path) -> list[Page]` |
| `chunking.py` | `Chunk` dataclass, the fixed-size chunker, a registry of chunkers by name |
| `providers/base.py` | `ChatModel` and `Embedder` protocols, `LLMResult` |
| `providers/anthropic_chat.py`, `providers/local_embedder.py`, `providers/factory.py` | The one default implementation of each protocol, chosen by config |
| `ingest.py` | Wires extract, chunk, embed, and store; idempotent per chunk set |
| `retrieval.py` | `search_lexical`, `search_vector`, `rrf_fuse`, `hybrid_search` |
| `answer.py` | Prompt assembly, the response models, citation validation, `answer_question()` |
| `api.py` | The FastAPI app |
| `cli.py` | `fineprint` command: `init-db`, `ingest`, `search`, `ask`, `serve` (argparse) |

| Module (`evals/`) | Responsibility |
|-------------------|----------------|
| `golden.py` | Load and validate the golden set (exists) |
| `verify_golden_set.py` | Check every evidence quote against the PDF (exists) |
| `run_golden_set.py` | Run one configuration over the golden set, write `evals/results/<run_id>.json` |
| `review.py` | Terminal tool to record a manual pass or fail per answer into a results file |
| `scoreboard.py` | Generate `evals/scoreboard.md` and the README block from the results files |

## Configuration

All settings come from environment variables, with `.env` loaded for local development. Task 1
adds the rows that are not yet in `.env.example`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://fineprint:fineprint@localhost:5432/fineprint` | Matches `docker-compose.yml` |
| `CORPUS_EDITION` | `2026` | Which handbook edition queries run against. Part 4 loads a second edition next to it. |
| `CHUNK_SET` | `fixed-220w` | Which set of chunks queries run against. Part 2 adds more sets side by side. |
| `LLM_PROVIDER` / `LLM_MODEL` | `anthropic` / `claude-opus-5` | D1 |
| `ANTHROPIC_API_KEY` | none | Secret. Only `/ask` and answer evals need it. |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `fastembed` / `BAAI/bge-small-en-v1.5` | D2 |
| `RETRIEVAL_CANDIDATES` | `20` | Rows taken from each retriever before fusion |
| `RETRIEVAL_TOP_K` | `5` | Chunks kept after fusion and sent to the LLM |
| `RRF_K` | `60` | The RRF constant |

## Data model

`schema.sql`, applied by `fineprint init-db`. Every statement is `IF NOT EXISTS`, so running it
twice is safe.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          serial PRIMARY KEY,
    edition     int  NOT NULL UNIQUE,          -- 2026
    title       text NOT NULL,
    source_url  text NOT NULL,
    sha256      text NOT NULL,
    page_count  int  NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    document_id int  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number int  NOT NULL,                 -- see "Corpus facts" for the numbering rule
    text        text NOT NULL,
    PRIMARY KEY (document_id, page_number)
);

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
    embedding   vector(384) NOT NULL,          -- must equal the embedding model's dimension (D2)
    UNIQUE (document_id, chunk_set, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- No approximate vector index in part 1 (D4): a few hundred rows scan exactly in milliseconds,
-- and exact results keep evals deterministic. When the table grows, add:
--   CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
-- and set hnsw.iterative_scan, because every query also filters by document and chunk set.
```

The `documents.edition` and `chunks.chunk_set` columns are what let part 2 compare chunkers and
part 4 compare editions without a schema change. Changing to an embedding model with a different
dimension means editing the `vector(384)` line, re-running `init-db` on a fresh database, and
re-ingesting; `ingest` must refuse to run when the embedder's dimension does not match the column.

## Tasks

Do them in order. Each task lands as one or more conventional commits, with its tests in the same
commit. Tests must not need an API key or network: use the fake embedder and fake chat model from
task 4. Tests that need Postgres carry the `integration` marker and skip with a clear reason when
`DATABASE_URL` is unreachable.

### Task 0: Preconditions

- The decisions above are resolved.
- `uv sync`, `docker compose up -d --wait`, and `uv run python scripts/download_handbook.py` all
  succeed.
- `uv run pytest` and `uv run python -m evals.verify_golden_set` pass.

### Task 1: Settings, database plumbing, schema

Add `pydantic`, `pydantic-settings`, `psycopg[binary,pool]`, and `pgvector`. Write `config.py`,
`db.py`, `schema.sql`, and the `init-db` subcommand (register the `fineprint` console script in
`pyproject.toml`). Complete `.env.example` from the configuration table. Register the console script
as `fineprint = "fineprint.cli:main"` and register the `integration` pytest marker in
`pyproject.toml`. `schema.sql` lives inside the package and is read with `importlib.resources`;
check that the built package includes it.

Accept when: `fineprint init-db` works on an empty database and again on an initialized one; an
integration test creates the schema and round-trips one row with a 384-dimension vector; settings
load from the environment with the documented defaults; no secret has a default value.

Commit: `feat(db): add settings, connection pool, and schema`

### Task 2: PDF extraction

`pdf.py` exposes `extract_pages(path) -> list[Page]`, where `Page` has `number` and `text`. Use
the library and the numbering rule recorded under "Corpus facts". Do no cleanup beyond what that
section calls for: part 1 is the naive baseline, and headers, the table of contents, and the index
stay in. Their effect on retrieval is something part 2 measures.

Accept when: the page count equals the number recorded under "Corpus facts"; a test asserts that
three known phrases appear on their known pages (take them from the `evidence` entries of the
golden set); the test skips with a clear reason when the PDF is absent.

Commit: `feat(ingest): extract handbook pages with page numbers`

### Task 3: Fixed-size chunking

`chunking.py` holds the `Chunk` dataclass (`ordinal`, `page_start`, `page_end`, `section`,
`text`) and `fixed_size_chunks(pages, words=220, overlap=40)`. Treat the document as one stream
of words in which each word remembers its page. A chunk is a window of `words` words, consecutive
windows share `overlap` words, and `page_start` and `page_end` come from the first and last word.
Register chunkers in a dict keyed by chunk-set name so part 2 can add one without touching
callers.

Accept when unit tests show: every word of the input appears in at least one chunk; page ranges
are correct for a chunk that crosses a page boundary; the overlap is exactly `overlap` words; a
document shorter than one window gives a single chunk; ordinals are consecutive from zero.

Commit: `feat(ingest): add fixed-size chunker with page provenance`

### Task 4: Provider abstraction and the embedder

`providers/base.py`:

```python
class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class LLMResult[T]:
    parsed: T
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ChatModel(Protocol):
    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]: ...
```

Documents and queries get separate methods because many embedding models treat them differently.
`LLMResult` carries tokens and latency now so part 3 can add cost and p95 latency without changing
any signature. `providers/factory.py` maps the provider names from config to classes with a plain
dict and raises a clear error for an unknown name. Add `fastembed` and write
`providers/local_embedder.py` (check the installed library's API rather than recalling it). Put a
deterministic `FakeEmbedder` (vectors derived from a hash of the text) and a `FakeChatModel`
(returns a canned object) in `tests/fakes.py`.

Accept when: the local embedder returns vectors of length `dimension` and the same text gives the
same vector twice; the factory test covers the known and unknown provider names; nothing imports a
provider SDK at module import time except the provider's own module.

Commit: `feat(providers): add chat and embedding protocols with local embedder`

### Task 5: Ingest command

`fineprint ingest [--edition 2026] [--chunk-set fixed-220w]` verifies the PDF hash against the
pinned value, extracts pages, chunks, embeds in batches, and writes `documents`, `pages`, and
`chunks` in one transaction. Re-running replaces that document's rows for that chunk set. It prints
the page count, chunk count, and the minimum, median, and maximum chunk length in words, and refuses
to run on a dimension mismatch.

The table of pinned editions (year, file name, sha256, source URLs) must live in one place that both
the package and the script can import. Move `EDITIONS` from `scripts/download_handbook.py` into
`src/fineprint/editions.py` and have the script import it. `documents.source_url` is the edition's
primary URL from that table.

Accept when: an integration test ingests a three-page fixture with the fake embedder, and a second
run leaves the same row counts; a real run on the handbook completes and its printed counts are
pasted into the commit body.

Commit: `feat(ingest): add idempotent ingest command`

### Task 6: Hybrid retrieval

`retrieval.py`. Both searches filter by document and chunk set, return `(chunk_id, rank, score)`,
and break ties by `id` so results are deterministic. Resolve `document_id` from `CORPUS_EDITION`
once at startup, and fail with a clear message when that edition has not been ingested.

Lexical search. A natural-language question must not be turned into an AND query, because then a
single missing word returns nothing. Build an OR query from the question's own lexemes, using
Postgres to produce them so the stemming matches the stored `tsv`. Reference query (verify it in
the integration tests):

```sql
WITH q AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
    FROM unnest(to_tsvector('english', %(question)s))
)
SELECT c.id, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks AS c, q
WHERE c.document_id = %(document_id)s AND c.chunk_set = %(chunk_set)s AND c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(limit)s;
```

The cast to `tsquery` takes the lexemes as they are, so they are not stemmed a second time. A
question made only of stop words yields a NULL query; return an empty list.

Vector search orders by cosine distance (`embedding <=> %(query_vector)s`) and reports
`1 - distance` as the score.

`rrf_fuse(rankings, k=60)` is a pure function: a chunk's fused score is the sum over the rankings
of `1 / (k + rank)`, with ranks starting at 1. It returns the fused order and keeps each chunk's
rank in every input ranking, so `/search` can show where a result came from.
`hybrid_search(question, mode)` supports `hybrid`, `vector`, and `lexical`.

Accept when: unit tests for `rrf_fuse` cover a chunk found by both retrievers outranking chunks
found by one, tie-breaking, and an empty ranking; integration tests show that a full question
returns lexical hits, that questions containing an apostrophe, a dollar sign, a colon, and a
hyphenated phone number do not raise, and that a stop-word-only question returns an empty list;
`fineprint search "<question>" --mode hybrid` prints ranks, pages, both source ranks, and the
first 100 characters of each chunk.

Commit: `feat(retrieval): add lexical, vector, and RRF hybrid search`

### Task 7: Answer generation

`answer.py`. The model returns a draft; the service validates it and adds the page numbers, so a
page number in a response always comes from the database, never from the model.

```python
class Citation(BaseModel):  # returned by the model
    chunk_id: int
    quote: str  # short verbatim span from that chunk


class DraftAnswer(BaseModel):  # returned by the model
    answer: str
    found_in_handbook: bool
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


class CitedChunk(BaseModel):  # returned by the API
    chunk_id: int
    page_start: int
    page_end: int
    quote: str
    quote_verified: bool  # the quote occurs in the chunk text, ignoring whitespace runs


class AnswerResponse(BaseModel):
    question: str
    answer: str
    found_in_handbook: bool
    citations: list[CitedChunk]
    dropped_citations: int  # citations that named a chunk which was not retrieved
    confidence: Literal["high", "medium", "low"]
    retrieved_chunk_ids: list[int]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
```

Add `anthropic`. The Anthropic implementation of `ChatModel` uses `client.messages.parse(...,
output_format=DraftAnswer)` and reads `response.parsed_output`. Load the `claude-api` skill before
writing it and follow its current guidance: check `stop_reason` before reading content, handle
`refusal`, and do not send `temperature` or other sampling parameters, which current Claude models
reject. The API's own citations feature cannot be combined with structured output, which is why
citations are chunk ids that the service checks itself.

Prompt rules, in the system prompt: answer only from the supplied excerpts of Medicare & You
2026; copy dollar amounts, dates, and phone numbers exactly; cite the chunk ids used, each with a
short quote; when the excerpts do not contain the answer, set `found_in_handbook` to false, say
so plainly, and point to the right resource named in the excerpts or one of Medicare.gov,
1-800-MEDICARE, the State Health Insurance Assistance Program, or Social Security; never give
personal medical advice. Confidence means: `high` when the cited text states the answer directly,
`medium` when the answer combines passages or needs a small inference, `low` when the support is
partial. Present each excerpt to the model as `<chunk id="123" pages="45-46">…</chunk>`.

Accept when unit tests with the fake chat model show: a citation naming a chunk that was not
retrieved is dropped and counted; page numbers come from the retrieved rows; `quote_verified` is
right for a matching quote, a quote that differs only in whitespace, and an invented quote; the
prompt contains every retrieved chunk once, in ranked order. One manual run of `fineprint ask`
on an answerable and an unanswerable question is pasted into the commit body.

Commit: `feat(answer): generate structured, cited answers`

### Task 8: API

Add `fastapi`, `uvicorn`, and `httpx` (dev, for the test client). `api.py` opens the connection
pool and loads the embedder once in the lifespan handler.

- `POST /search` with `{"query": str, "top_k": int = 5, "mode": "hybrid" | "vector" | "lexical" = "hybrid"}`
  returns the ranked chunks: id, pages, text, fused score, and the rank in each source ranking.
- `POST /ask` with `{"question": str, "top_k": int = 5}` returns `AnswerResponse`.
- `GET /healthz` checks the database connection.

Validate inputs (non-empty text, `top_k` between 1 and 20). A provider failure returns 502 with a
short message and no stack trace. `fineprint serve` runs uvicorn.

Accept when: test-client tests, using dependency overrides for the embedder and chat model, cover
a success and a validation error for each endpoint and the 502 path; `/docs` renders the schemas.

Commit: `feat(api): add /search, /ask, and /healthz`

### Task 9: Golden set to about 40 questions

Only after the owner has approved the first 15. Follow "Golden set" below and `evals/README.md`.
Keep the mix the brief asks for: easy lookups, answers that live in a table, answers that need two
sections, and a few the handbook does not answer.

Accept when: `uv run python -m evals.verify_golden_set` passes with every evidence quote found on
its page; ids are unique; any item that could not be verified says `needs_review: true`.

Commit: `feat(evals): extend golden set to 40 questions`

### Task 10: Eval runner, manual review, scoreboard

`python -m evals.run_golden_set --config hybrid [--retrieval-only]` runs every question and writes
`evals/results/<run_id>.json` (format below). `--retrieval-only` skips the LLM, needs no key, and is
deterministic. Configurations in part 1: `hybrid`, `vector-only`, and `lexical-only`. Run the last
two retrieval-only; they cost nothing and show whether fusion earns its place. The runner creates
`evals/results/` when it is missing.

`python -m evals.review evals/results/<run_id>.json` shows each question, the expected answer, and
the generated answer, and records `pass` or `fail` with an optional note. Unreviewed answers stay
`null`, and the scoreboard reports how many were reviewed. These labels are reused in part 4 to
calibrate the judge.

`python -m evals.scoreboard` regenerates `evals/scoreboard.md` and the block between
`<!-- scoreboard:start -->` and `<!-- scoreboard:end -->` in `README.md`. With `--check` it exits
non-zero when either is out of date, which is what stops hand edits.

Accept when: metric functions have unit tests with hand-computed cases; a fixture results file
renders to an exact expected Markdown table; the three configurations have been run on the real
corpus, the `hybrid` answers reviewed, and the results files committed.

Commits: `feat(evals): add golden-set runner and metrics`, `feat(evals): add manual review tool`,
`feat(evals): generate scoreboard from results`

### Task 11: README, tag

Write the part 1 section of `README.md`: what was built, how to run it end to end, and what the
numbers say, including the failures. Every number comes from the generated block. Then tag
`part-1`.

### Task 12: Video script

Write `docs/scripts/01-rag-service.md` from "What the video must show".

## API contract notes

Status codes: 200 on success, 422 on validation errors (FastAPI's default), 502 when the LLM or
embedding provider fails, 503 from `/healthz` when the database is unreachable. `/ask` always
answers with 200 when it abstains: an abstention is a valid answer with `found_in_handbook: false`.

## Golden set

One JSON object per line in `evals/golden_set.jsonl`.

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | `q001`, `q002`, … Never reused or renumbered. |
| `question` | string | In the voice of a beneficiary or their adult child |
| `expected_answer` | string | What a correct answer must say, including exact amounts. For unanswerable questions: that the handbook does not say, and the resource to point to. |
| `expected_pages` | list of int | Every page a correct answer needs. Empty for unanswerable questions. |
| `category` | string | `costs`, `coverage`, `enrollment`, `original_vs_advantage`, `part_d`, `medigap` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `type` | string | `lookup`, `table`, `multi_section`, `unanswerable`. Lets the scoreboard detail show where retrieval fails. |
| `evidence` | list of `{page, quote}` | Verbatim spans that support the expected answer. `verify_golden_set` checks each quote against the extracted text of its page, so "verified pages" is a reproducible claim. Every expected page has at least one quote. For `unanswerable` items the quotes come from the page where the handbook sends the reader elsewhere; that page is not an expected page. |
| `alt_pages` | object, optional | Maps an expected page to other pages that state the same fact, for example `{"30": [23]}`. Retrieving an alternate counts as retrieving the expected page. |
| `needs_review` | bool | True when a page or the answer could not be verified |

## Eval design

**Metrics in part 1.** A retrieved chunk covers a page when that page lies within the chunk's
`page_start` to `page_end`. An expected page counts as found when a top-5 chunk covers it or one of
its alternates in `alt_pages`. Retrieval metrics are computed over answerable questions at k = 5.

- `page_hit@5`: share of questions with at least one expected page found. **Headline.**
- `page_recall@5`: mean share of a question's expected pages found. This is the number that exposes
  multi-section questions.
- `mrr`: mean reciprocal rank of the first chunk that finds an expected page, counting 0 when there
  is none.
- `cited_page_hit`: share of answered questions whose citations find an expected page.
- `abstention_accuracy`: share of all questions where `found_in_handbook` matches whether the
  question is answerable.
- `manual_pass`: passes divided by reviewed answers, shown with the reviewed count. **Headline.**

**Results file.** `evals/results/<run_id>.json`, where `run_id` is
`<UTC timestamp>_<config name>`. It stores the git commit and whether the tree was dirty; the
configuration (name, mode, chunk set, k, candidates, RRF k, embedding model, LLM model or null);
the corpus edition and sha256; the golden-set sha256 and count; and, per question, the retrieved
chunk ids with pages and ranks, the per-question metrics, the full `AnswerResponse` or null, and
the manual verdict. The scoreboard recomputes every aggregate from the per-question data. Results
files that back a published number are committed.

**Scoreboard format.** One row per configuration (the latest results file for each
configuration name). The headline columns are fixed for the whole series, so the same table
closes every video with more cells filled in:

| Column | Filled in |
|--------|-----------|
| Configuration | part 1 |
| Page hit@5, Manual pass | part 1 |
| Context recall, Context precision, Faithfulness, Answer relevance | part 2 |
| Cost per query, p95 latency | part 3 |
| Judge pass, Adversarial pass | part 4 |

A cell with no measurement renders as `—`, and a legend under the table says so. A dash is the
absence of a measurement, never a stand-in value. Below the headline table, a detail section per
part carries the secondary metrics and the breakdown by question `type`. The header states the
golden-set size, the corpus edition and hash, and the generation time.

## Seams for later parts

- `chunks.chunk_set` plus the chunker registry: part 2 adds the structure-aware chunker and
  compares sets side by side.
- `hybrid_search` returns candidates before the final cut: part 2 inserts the re-ranker there.
- `LLMResult` and `AnswerResponse` already carry tokens and latency: part 3 adds cost and p95.
- `answer_question()` is the single entry point for a query: part 3 wraps it with tracing and puts
  the agent above it.
- `documents.edition` and `CORPUS_EDITION`: part 4 loads another edition and diffs the answers.
- `--retrieval-only` and `scoreboard --check`: part 4 runs them in CI without secrets.

## What the video must show

Seven to ten minutes, in this order.

1. The finished scoreboard first, with most cells empty: this is what the series measures, and
   part 1 fills the first columns.
2. The corpus: the handbook, why it is a good test (tables, page cross-references, a new edition
   every year), and the download script with its pinned hash.
3. Ingestion, live: run `fineprint ingest`, then look at one chunk row in `psql` with its pages,
   `tsv`, and embedding.
4. Hybrid retrieval: `/search` in all three modes on two questions taken from the eval results,
   one that lexical finds and vector misses and one the other way round, then the RRF ranks that
   fix both. If the results contain no such pair, say so on camera instead of staging one.
5. `/ask`: the structured response, then the PDF opened at a cited page to prove the citation.
   Then an unanswerable question, and the abstention with its referral.
6. The golden set: one lookup, one table, and one unanswerable entry with their evidence quotes;
   the runner; two manual reviews; scoreboard generation.
7. The numbers, honestly: hybrid against the single retrievers, and which question types fail.
   Those failures are the opening of part 2.

## Definition of done

- [ ] From a clean clone: `uv sync`, `docker compose up -d --wait`, download, `init-db`, `ingest`,
      `serve`, and the three eval commands all work by following the README alone.
- [ ] `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest` pass; integration
      tests pass with the database up.
- [ ] The golden set has about 40 verified questions, and the owner has approved it.
- [ ] `evals/scoreboard.md` and the README block are generated, `scoreboard --check` passes, and
      every number in the README comes from that block.
- [ ] The part 1 README section, `docs/scripts/01-rag-service.md`, and the `part-1` tag exist.
- [ ] The owner has recorded the 7 to 10 minute walkthrough from the script, and the README links to
      it.
