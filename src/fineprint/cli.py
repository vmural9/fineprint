"""The `fineprint` command.

One subcommand per thing the service can do. Part 1 adds `init-db` here first; `ingest`,
`search`, `ask`, and `serve` join it as the later tasks build them.
"""

import argparse
from collections.abc import Sequence

from fineprint.config import Settings
from fineprint.db import describe_database, init_db


def run_init_db(args: argparse.Namespace) -> int:
    """Create the tables and indexes in the configured database."""
    settings = Settings()
    init_db(settings.database_url)
    print(f"schema applied to {describe_database(settings.database_url)}")
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return the exit code for the process."""
    args = build_parser().parse_args(argv)
    return args.run(args)
