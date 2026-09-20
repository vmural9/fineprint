"""The `fineprint` command.

One subcommand per thing the service can do. Part 1 adds `init-db` and `ingest` here first;
`search`, `ask`, and `serve` join them as the later tasks build them.

Each subcommand reads `Settings` itself, so a flag that is left off falls back to the
configured value and the environment stays the single place configuration lives.
"""

import argparse
import sys
import time
from collections.abc import Sequence

from fineprint.chunking import CHUNKERS
from fineprint.config import Settings
from fineprint.db import connection_pool, describe_database, init_db
from fineprint.editions import EDITIONS
from fineprint.ingest import IngestError, ingest
from fineprint.providers.factory import get_embedder


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return the exit code for the process."""
    args = build_parser().parse_args(argv)
    return args.run(args)
