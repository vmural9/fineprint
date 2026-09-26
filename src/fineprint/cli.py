"""The `fineprint` command.

One subcommand per thing the service can do: `init-db` and `ingest` prepare the database,
`search` shows what retrieval finds, `ask` answers a question from it, and `serve` puts the
same two operations behind HTTP.

Each subcommand reads `Settings` itself, so a flag that is left off falls back to the
configured value and the environment stays the single place configuration lives.

`search` and `ask` print for a person reading a terminal, not for a program: the text is
wrapped, the columns line up, and a quote the service could not find in its chunk says so
in words. Anything that wants the data should call the API instead.
"""

import argparse
import shutil
import sys
import textwrap
import time
from collections.abc import Sequence

import psycopg
import uvicorn
from psycopg_pool import PoolTimeout

from fineprint.answer import AnswerResponse, answer_question
from fineprint.chunking import CHUNKERS
from fineprint.config import Settings
from fineprint.db import connection_pool, describe_database, init_db
from fineprint.editions import EDITIONS
from fineprint.ingest import IngestError, ingest
from fineprint.providers.factory import get_chat_model, get_embedder, get_reranker
from fineprint.retrieval import (
    SEARCH_MODES,
    EditionNotIngestedError,
    RetrievedChunk,
    Retriever,
)

# The default address `fineprint serve` listens on. Localhost, because this service has no
# authentication and reads a database; putting it on a network is a deployment decision.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# How much of a chunk `search` shows, and how wide the prose it prints may run.
PREVIEW_CHARACTERS = 100
MAX_WIDTH = 96


def run_init_db(args: argparse.Namespace) -> int:
    """Create the tables and indexes in the configured database."""
    settings = Settings()
    init_db(settings.database_url)
    print(f"schema applied to {describe_database(settings.database_url)}")
    return 0


def run_ingest(args: argparse.Namespace) -> int:
    """Read the handbook into the database and report what was written."""
    settings = Settings()
    year = settings.corpus_edition if args.edition is None else args.edition
    chunk_set = settings.chunk_set if args.chunk_set is None else args.chunk_set

    edition = EDITIONS.get(year)
    if edition is None:
        print(
            f"ERROR: there is no pinned {year} edition of the handbook. "
            f"Pinned editions: {', '.join(str(pinned) for pinned in sorted(EDITIONS))}.",
            file=sys.stderr,
        )
        return 1

    started = time.perf_counter()
    try:
        embedder = get_embedder(settings)
        with connection_pool(settings.database_url) as pool:
            summary = ingest(pool, embedder, edition, chunk_set)
    except IngestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    words = summary.words
    print(
        f"ingested {edition.title} into {describe_database(settings.database_url)}\n"
        f"  chunk set    {summary.chunk_set}\n"
        f"  pages        {summary.page_count}\n"
        f"  chunks       {summary.chunk_count}\n"
        f"  chunk words  min {words.minimum}, median {words.median:g}, max {words.maximum}\n"
        f"  elapsed      {elapsed:.1f}s"
    )
    return 0


def text_width() -> int:
    """How wide to wrap prose: the terminal, but never wider than is comfortable to read."""
    return min(shutil.get_terminal_size(fallback=(80, 24)).columns, MAX_WIDTH)


def format_rank(rank: int | None) -> str:
    """A retriever's rank, or a dash when that retriever did not return this chunk."""
    return "-" if rank is None else str(rank)


def format_pages(page_start: int, page_end: int) -> str:
    """`23` for one page, `42-43` for a chunk that crosses a boundary."""
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"


def preview(text: str) -> str:
    """The first `PREVIEW_CHARACTERS` characters of a chunk, with an ellipsis if cut."""
    if len(text) <= PREVIEW_CHARACTERS:
        return text
    return text[:PREVIEW_CHARACTERS] + "…"


def print_hits(chunks: Sequence[RetrievedChunk]) -> None:
    """One block per retrieved chunk: where it ranked, where it is, and how it starts.

    A re-ranked chunk also shows its fused rank, where it stood before the re-ranker read it,
    and the re-ranker's score in place of the fused one.
    """
    if not chunks:
        print("no chunks matched this question")
        return
    for chunk in chunks:
        pages = format_pages(chunk.page_start, chunk.page_end)
        if chunk.rerank_score is None:
            scores = f"score {chunk.score:.4f}"
        else:
            scores = f"fused {format_rank(chunk.fused_rank):<3} rerank {chunk.rerank_score:.4f}"
        print(
            f"{chunk.rank:>3}. pages {pages:<6} "
            f"lexical {format_rank(chunk.lexical_rank):<3} "
            f"vector {format_rank(chunk.vector_rank):<3} "
            f"{scores}  chunk {chunk.chunk_id}"
        )
        print(f"     {preview(chunk.text)}")
        print()


def print_answer(answered: AnswerResponse) -> None:
    """The answer, what it rests on, and what the call cost."""
    width = text_width()
    print()
    for paragraph in answered.answer.splitlines():
        print(textwrap.fill(paragraph, width=width) if paragraph.strip() else "")
    print()
    print(
        f"found in the handbook: {'yes' if answered.found_in_handbook else 'no'}"
        f"    confidence: {answered.confidence}"
    )
    print()

    if answered.citations:
        print(f"citations ({len(answered.citations)}):")
        for number, cited in enumerate(answered.citations, start=1):
            pages = format_pages(cited.page_start, cited.page_end)
            label = "page" if cited.page_start == cited.page_end else "pages"
            verdict = "quote verified" if cited.quote_verified else "QUOTE NOT VERIFIED"
            print(f" {number:>2}. {label} {pages:<6} {verdict}")
            for line in textwrap.wrap(f'"{cited.quote}"', width=width - 5):
                print(f"     {line}")
    else:
        print("citations: none")
    if answered.dropped_citations:
        print(
            f" ! {answered.dropped_citations} citation(s) dropped: the model named a chunk "
            f"that was not retrieved"
        )
    print()
    print(
        f"{answered.model}  ·  {answered.input_tokens:,} in / {answered.output_tokens:,} out "
        f"tokens  ·  {answered.latency_ms / 1000:.1f}s"
    )


def open_retriever(settings: Settings, pool) -> Retriever:
    """A retriever on this pool, with the configured embedder and re-ranker behind it."""
    return Retriever(pool, get_embedder(settings), settings, reranker=get_reranker(settings))


def report_failure(error: Exception, settings: Settings) -> int:
    """Print one clear line about a failure the caller can do something about.

    A traceback is noise to someone who mistyped a question or forgot to start Postgres, so
    the exception's own first line is printed instead — with its class name, so that a real
    bug is still identifiable from the output.
    """
    if isinstance(error, EditionNotIngestedError):
        print(f"ERROR: {error}", file=sys.stderr)
    elif isinstance(error, psycopg.Error | PoolTimeout):
        print(
            f"ERROR: cannot query {describe_database(settings.database_url)}: "
            f"{str(error).splitlines()[0]}\n"
            "Start it with: docker compose up -d --wait, then: fineprint init-db && "
            "fineprint ingest",
            file=sys.stderr,
        )
    else:
        print(f"ERROR: {type(error).__name__}: {str(error).splitlines()[0]}", file=sys.stderr)
    return 1


def run_search(args: argparse.Namespace) -> int:
    """Show the chunks a question retrieves, and where each one came from."""
    settings = Settings()
    top_k = settings.retrieval_top_k if args.top_k is None else args.top_k

    try:
        with connection_pool(settings.database_url) as pool:
            chunks = open_retriever(settings, pool).search(
                args.question, mode=args.mode, top_k=top_k
            )
    except Exception as error:
        return report_failure(error, settings)

    print(f'search: "{args.question}"')
    print(f"mode:   {args.mode}    top-k: {top_k}    edition: {settings.corpus_edition}")
    print()
    print_hits(chunks)
    return 0


def run_ask(args: argparse.Namespace) -> int:
    """Answer a question from the handbook and show what the answer rests on."""
    settings = Settings()
    top_k = settings.retrieval_top_k if args.top_k is None else args.top_k

    try:
        with connection_pool(settings.database_url) as pool:
            answered = answer_question(
                args.question,
                open_retriever(settings, pool),
                get_chat_model(settings),
                top_k=top_k,
            )
    except Exception as error:
        return report_failure(error, settings)

    print(f'question: "{args.question}"')
    print_answer(answered)
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Serve the API with uvicorn until interrupted."""
    settings = Settings()
    print(
        f"fineprint serving on http://{args.host}:{args.port}  "
        f"(docs at http://{args.host}:{args.port}/docs)\n"
        f"edition {settings.corpus_edition}, chunk set {settings.chunk_set}, "
        f"database {describe_database(settings.database_url)}"
    )
    uvicorn.run("fineprint.api:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Describe the command and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="fineprint",
        description='Answer Medicare questions from the official "Medicare & You 2026" handbook.',
    )
    # No metavar: the usage line should name the commands, so `fineprint` on its own tells the
    # reader what there is to run.
    subcommands = parser.add_subparsers(dest="command", required=True)

    init_db_command = subcommands.add_parser(
        "init-db",
        help="create the tables and indexes in DATABASE_URL; safe to run more than once",
        description=(
            "Apply the schema to the database named by DATABASE_URL. Running it on a database "
            "that already has the schema changes nothing."
        ),
    )
    init_db_command.set_defaults(run=run_init_db)

    ingest_command = subcommands.add_parser(
        "ingest",
        help="read the handbook into the database; safe to run more than once",
        description=(
            "Verify the pinned handbook PDF, cut it into chunks, embed them, and write the "
            "pages and chunks to the database. Running it again for the same edition and chunk "
            "set replaces those rows instead of adding a second copy. Embedding calls AWS "
            "Bedrock, so this command needs AWS credentials."
        ),
    )
    ingest_command.add_argument(
        "--edition",
        type=int,
        choices=sorted(EDITIONS),
        help="handbook edition year (default: CORPUS_EDITION)",
    )
    ingest_command.add_argument(
        "--chunk-set",
        choices=sorted(CHUNKERS),
        help="which chunker to cut the pages with, and the name its chunks are stored under "
        "(default: CHUNK_SET)",
    )
    ingest_command.set_defaults(run=run_ingest)

    search_command = subcommands.add_parser(
        "search",
        help="show the handbook passages a question retrieves, without asking a model",
        description=(
            "Search the ingested handbook and print the chunks that come back: their rank, "
            "their pages, where each one stood in the lexical and the vector list, and how "
            "each one starts. Searching by meaning embeds the question, which calls AWS "
            "Bedrock; --mode lexical calls no model at all. With RERANKER_PROVIDER set, a "
            "hybrid search also has a re-ranker put the candidates in a new order, one more "
            "Bedrock call, and prints each chunk's fused rank and re-rank score."
        ),
    )
    search_command.add_argument("question", help="what to search for, in quotes")
    search_command.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        default="hybrid",
        help="hybrid fuses both retrievers, vector searches by meaning, lexical by words "
        "(default: hybrid)",
    )
    search_command.add_argument(
        "--top-k",
        type=int,
        help="how many chunks to keep (default: RETRIEVAL_TOP_K)",
    )
    search_command.set_defaults(run=run_search)

    ask_command = subcommands.add_parser(
        "ask",
        help="answer a Medicare question from the handbook, with citations",
        description=(
            "Retrieve the passages a question is about, ask Claude to answer from them "
            "alone, then check every citation against what was retrieved. Page numbers come "
            "from the database, never from the model. This calls AWS Bedrock twice: once to "
            "embed the question and once to answer it, and once more to re-rank the passages "
            "when RERANKER_PROVIDER is set."
        ),
    )
    ask_command.add_argument("question", help="the question, in quotes")
    ask_command.add_argument(
        "--top-k",
        type=int,
        help="how many chunks to answer from (default: RETRIEVAL_TOP_K)",
    )
    ask_command.set_defaults(run=run_ask)

    serve_command = subcommands.add_parser(
        "serve",
        help="run the HTTP API",
        description=(
            "Serve /search, /ask and /healthz with uvicorn. The connection pool and the "
            "models are built once at startup and shared by every request."
        ),
    )
    serve_command.add_argument(
        "--host", default=DEFAULT_HOST, help=f"address to listen on (default: {DEFAULT_HOST})"
    )
    serve_command.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port to listen on (default: {DEFAULT_PORT})",
    )
    serve_command.set_defaults(run=run_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return the exit code for the process."""
    args = build_parser().parse_args(argv)
    return args.run(args)
