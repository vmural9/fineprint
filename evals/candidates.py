"""The candidate-rank diagnostic: where would a re-ranker have to find its answer?

    python -m evals.candidates [--chunk-set NAME] [--mode hybrid|vector|lexical]

A re-ranker can only promote a chunk that retrieval already put in its pool (hypothesis H2). For
every answerable golden question, this script calls `Retriever.candidates()` — the fused list
before the top-5 cut and before any re-ranking — and finds the 1-based pool rank of the first
chunk that covers an expected page (or a page recorded as stating the same fact under it in
`alt_pages`). The rank is bucketed: `1-5` is already where `search()` would return it; `6-20` and
`21+` are inside the pool but past the top 5, exactly what a re-ranker could still promote;
`absent` means no chunk in the pool covers the page at all, which no re-ranker can fix.

The pool depends on only two things: the chunk set and the retrieval mode. Re-ranking and the
top-k cut both happen after `candidates()` returns, so neither plays any part here — which is
why this script takes `--chunk-set` and `--mode` directly instead of an `evals/configs.py` name.

`--mode lexical` calls no model at all; `hybrid` and `vector` each embed every question once,
which is one Bedrock call per question either way. No answer model ever runs.

This reuses `evals.run_golden_set`'s `build_retriever`, `corpus_from_database` and `RunError`,
and `evals.results`'s provenance helpers, rather than rebuilding a second copy of "how a run
turns `Settings` into a database connection and a results file". `save()` and `results_path()`
only need a dataclass, so they work for this script's own schema too, even though it is not a
`RunResult`.

Its results land in `evals/results/candidates/`, a subdirectory of the run results rather than
beside them: `evals.results.load_all` globs only the top level of `evals/results/` and treats
every file it finds there as a `RunResult`, which a `CandidatesResult` — it has no `config`
field — is not, so the scoreboard would refuse it with "not a results file".
"""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psycopg
from psycopg_pool import PoolTimeout

from evals.golden import DEFAULT_GOLDEN_SET, GoldenItem, GoldenSetError, load_golden_set
from evals.metrics import RankedPageRange, covers
from evals.results import (
    RESULTS_DIR,
    Corpus,
    GitState,
    GoldenSetInfo,
    git_state,
    make_run_id,
    now_utc,
    results_path,
    save,
    sha256_of,
)
from evals.run_golden_set import RunError, build_retriever, corpus_from_database
from fineprint.config import Settings
from fineprint.db import connection_pool, describe_database

# Mirrors `fineprint.retrieval.SEARCH_MODES`, named here so this module needs no import of
# `fineprint.retrieval` (and everything behind it) just to describe the command line.
MODES = ("hybrid", "vector", "lexical")

# Where this diagnostic writes its output: its own subdirectory of the run results, not beside
# them. A `CandidatesResult` carries no `config` field, so `evals.results.load_all`'s glob — the
# one the scoreboard reads through — would refuse one of these files with "not a results file"
# if it ever landed at the top level of `evals/results/` alongside an actual run.
CANDIDATES_DIR = RESULTS_DIR / "candidates"

# The three question types the retrieval metrics apply to, in the order the scoreboard uses.
# `unanswerable` questions have no expected page to look for, so they never appear in a row here.
QUESTION_TYPES = ("lookup", "table", "multi_section")

BUCKETS = ("1-5", "6-20", "21+", "absent")

EXIT_OK = 0
EXIT_FAILED = 1


class Retriever(Protocol):
    """The part of `fineprint.retrieval.Retriever` this diagnostic uses."""

    def candidates(self, question: str, mode: str = "hybrid") -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class QuestionRank:
    """One answerable question's row: where its first covering chunk sat in the pool."""

    id: str
    type: str
    expected_pages: list[int]
    first_hit_rank: int | None
    bucket: str


@dataclass(frozen=True, slots=True)
class CandidatesResult:
    """One run of the diagnostic, written to
    `evals/results/<UTC ts>_candidates-<chunk_set>-<mode>.json`."""

    run_id: str
    created_at: str
    git: GitState
    chunk_set: str
    mode: str
    pool_size: int
    corpus: Corpus
    golden_set: GoldenSetInfo
    questions: list[QuestionRank]


def first_hit_rank(item: GoldenItem, candidates: Sequence[RankedPageRange]) -> int | None:
    """The 1-based pool rank of the first chunk covering an expected page or its alternate.

    "Covering" is `metrics.covers` applied to every page `pages_that_satisfy` accepts for each
    expected page — the same rule `metrics.found_expected_pages` uses to decide whether a page
    counts as found at all. Taking the minimum rank among the matching chunks, rather than the
    first match in list order, is what makes this correct regardless of how the pool is ordered.
    """
    satisfying_pages = {
        page for expected in item.expected_pages for page in item.pages_that_satisfy(expected)
    }
    ranks = [
        chunk.rank for chunk in candidates if any(covers(chunk, page) for page in satisfying_pages)
    ]
    return min(ranks) if ranks else None


def bucket_for(rank: int | None) -> str:
    """Which of the four buckets a pool rank falls into.

    `21+` is open-ended on purpose: folding every deep rank into a `21-40` bucket would only be
    true while `RETRIEVAL_CANDIDATES` stays at its default of 20, and this tool exists to show
    where a page actually sits. A rank of 90 is `21+` here, never mislabelled `absent` — the
    exact number is still in `first_hit_rank` for anyone who needs it.
    """
    if rank is None:
        return "absent"
    if rank <= 5:
        return "1-5"
    if rank <= 20:
        return "6-20"
    return "21+"


def rank_question(item: GoldenItem, candidates: Sequence[RankedPageRange]) -> QuestionRank:
    """Score one answerable question against its retrieved candidate pool."""
    rank = first_hit_rank(item, candidates)
    return QuestionRank(
        id=item.id,
        type=item.type,
        expected_pages=list(item.expected_pages),
        first_hit_rank=rank,
        bucket=bucket_for(rank),
    )


def rank_questions(
    items: Sequence[GoldenItem], retriever: Retriever, *, mode: str
) -> list[QuestionRank]:
    """Rank every answerable question's first covering chunk in its candidate pool.

    Unanswerable questions have no expected pages to look for, so they are skipped outright, the
    same way the retrieval metrics skip them (see `metrics.page_hit`).
    """
    return [
        rank_question(item, retriever.candidates(item.question, mode=mode))
        for item in items
        if item.answerable
    ]


def render_table(rows: Sequence[QuestionRank]) -> list[str]:
    """One line per question type plus an `all` row, each with its four bucket counts."""
    header = f"{'type':<13} " + " ".join(f"{bucket:>6}" for bucket in BUCKETS) + "   n"
    lines = [header]
    for question_type in (*QUESTION_TYPES, "all"):
        subset = (
            rows if question_type == "all" else [row for row in rows if row.type == question_type]
        )
        counts = Counter(row.bucket for row in subset)
        cells = " ".join(f"{counts[bucket]:>6}" for bucket in BUCKETS)
        lines.append(f"{question_type:<13} {cells}   {len(subset)}")
    return lines


def render_outliers(rows: Sequence[QuestionRank]) -> list[str]:
    """One line per question whose first covering chunk fell outside the top 5, or missed the
    pool entirely — the questions a re-ranker could help, or could never help."""
    lines = []
    for row in rows:
        if row.bucket == "1-5":
            continue
        rank_text = "absent" if row.first_hit_rank is None else str(row.first_hit_rank)
        lines.append(f"{row.id}  {row.type:<13} rank {rank_text}  expected {row.expected_pages}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    """Describe the command."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.candidates",
        description=(
            "For every answerable golden question, find where the first chunk covering an "
            "expected page sits in the retriever's candidate pool, before any cut or "
            "re-ranking (tests the premise of H2: a re-ranker can only lift a page that is "
            "already in the pool). Writes into its own subdirectory, apart from run results, "
            "since its files are not runs and evals.results.load_all would refuse them as one."
        ),
    )
    parser.add_argument(
        "--chunk-set",
        default=None,
        help="which chunk set to search (default: Settings().chunk_set)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="hybrid",
        help="which retrieval mode's pool to inspect (default: hybrid)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=CANDIDATES_DIR,
        help=f"where to write the results file (default: {CANDIDATES_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic once and write
    `evals/results/candidates/<UTC ts>_candidates-<chunk_set>-<mode>.json`."""
    args = build_parser().parse_args(argv)
    settings = Settings()
    chunk_set = args.chunk_set or settings.chunk_set
    if chunk_set != settings.chunk_set:
        settings = settings.model_copy(update={"chunk_set": chunk_set})
    mode = args.mode

    try:
        items = load_golden_set(DEFAULT_GOLDEN_SET)
    except GoldenSetError as error:
        print(f"ERROR: {DEFAULT_GOLDEN_SET} failed validation:\n{error}", file=sys.stderr)
        return EXIT_FAILED

    answerable = [item for item in items if item.answerable]
    print(
        f"candidates: {len(answerable)} answerable questions, mode {mode}, chunk set "
        f"{chunk_set}, pool size {settings.retrieval_candidates}, database "
        f"{describe_database(settings.database_url)}"
    )

    try:
        with connection_pool(settings.database_url) as pool:
            corpus = corpus_from_database(pool, settings.corpus_edition)
            retriever = build_retriever(settings, pool)
            rows = rank_questions(answerable, retriever, mode=mode)
    except RunError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED
    except (psycopg.Error, PoolTimeout) as error:
        # The database being down or unprepared is an ordinary mistake, not a crash: say which
        # command fixes it instead of printing a traceback.
        print(
            f"ERROR: cannot query {describe_database(settings.database_url)}: "
            f"{str(error).splitlines()[0]}\n"
            "Start it with: docker compose up -d --wait, then: fineprint init-db && "
            "fineprint ingest",
            file=sys.stderr,
        )
        return EXIT_FAILED

    result = CandidatesResult(
        run_id=make_run_id(f"candidates-{chunk_set}-{mode}"),
        created_at=now_utc(),
        git=git_state(),
        chunk_set=chunk_set,
        mode=mode,
        pool_size=settings.retrieval_candidates,
        corpus=corpus,
        golden_set=GoldenSetInfo(sha256=sha256_of(DEFAULT_GOLDEN_SET), count=len(items)),
        questions=rows,
    )
    path = save(result, results_path(result.run_id, args.results_dir))

    print()
    for line in render_table(rows):
        print(line)
    outliers = render_outliers(rows)
    if outliers:
        print("\noutside the top 5:")
        for line in outliers:
            print(line)
    print(f"\nwrote {path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
