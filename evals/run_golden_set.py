"""Run one configuration over the golden set and write a results file.

    python -m evals.run_golden_set --config hybrid
    python -m evals.run_golden_set --config lexical-only --retrieval-only

Part 1 has three configurations: `hybrid` fuses both retrievers, `vector-only` searches by
meaning alone, and `lexical-only` searches by words alone. Running the last two with
`--retrieval-only` is how the scoreboard shows whether fusion earns its place.

`--retrieval-only` skips the answer model, which is the expensive call, but a run is neither
free nor offline: every mode that searches by meaning embeds each question, and that is one
Bedrock call per question. Only `lexical-only --retrieval-only` calls no model at all.

A failure on one question is recorded on that question and the run carries on, so one bad
question cannot cost a whole run. The metrics of a failed question are left empty rather than
counted as zero, and the summary says how many failed.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from evals import metrics
from evals.golden import DEFAULT_GOLDEN_SET, GoldenItem, GoldenSetError, load_golden_set
from evals.results import (
    RESULTS_DIR,
    Corpus,
    GoldenSetInfo,
    QuestionResult,
    RetrievedChunk,
    RunConfig,
    RunResult,
    git_state,
    make_run_id,
    now_utc,
    results_path,
    save,
    sha256_of,
)
from fineprint.config import Settings
from fineprint.db import connection_pool, describe_database

# The configurations of part 1, and the retrieval mode each one runs in.
CONFIGS = {"hybrid": "hybrid", "vector-only": "vector", "lexical-only": "lexical"}

EXIT_OK = 0
EXIT_FAILED = 1


class RunError(RuntimeError):
    """Raised when the run cannot start at all, as opposed to one question failing."""


class Retriever(Protocol):
    """The part of `fineprint.retrieval.Retriever` this runner uses."""

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> Sequence[Any]: ...


# Given a question, produce an `AnswerResponse`. The runner binds the retriever and the chat
# model into it, so the loop only has to hand it a question.
AnswerFunction = Callable[[str], Any]


def run_questions(
    items: Sequence[GoldenItem],
    retriever: Retriever,
    answer_fn: AnswerFunction | None = None,
    *,
    mode: str,
    top_k: int,
    report: Callable[[str], None] = print,
) -> list[QuestionResult]:
    """Search for every question, optionally answer it, and score what came back.

    The retriever and the answer function are arguments rather than globals so that the tests
    can drive this loop with fakes and no database, no AWS, and no network.
    """
    rows: list[QuestionResult] = []
    for item in items:
        row = _run_one(item, retriever, answer_fn, mode=mode, top_k=top_k)
        rows.append(row)
        report(progress_line(row))
    return rows


def _run_one(
    item: GoldenItem,
    retriever: Retriever,
    answer_fn: AnswerFunction | None,
    *,
    mode: str,
    top_k: int,
) -> QuestionResult:
    """One question: retrieve, answer, score. Any failure lands in the row's `error`."""
    row = QuestionResult(
        id=item.id,
        question=item.question,
        type=item.type,
        category=item.category,
        difficulty=item.difficulty,
        answerable=item.answerable,
        expected_pages=list(item.expected_pages),
        retrieved=[],
        metrics=metrics.QuestionMetrics(),
    )

    try:
        chunks = list(retriever.search(item.question, mode=mode, top_k=top_k))
    except Exception as error:  # one question's failure is not the run's failure
        row.error = f"retrieval failed: {error}"
        return row
    row.retrieved = [_retrieved(chunk) for chunk in chunks]

    answer = None
    if answer_fn is not None:
        try:
            answer = answer_fn(item.question)
        except Exception as error:
            row.error = f"answering failed: {error}"

    if answer is not None:
        row.answer = _as_json(answer)
        citations = list(answer.citations)
        found_in_handbook = answer.found_in_handbook
    else:
        citations = None
        found_in_handbook = None

    row.metrics = metrics.score_question(
        item, chunks, citations=citations, found_in_handbook=found_in_handbook
    )
    return row


def _retrieved(chunk: Any) -> RetrievedChunk:
    """Keep the parts of a retrieved chunk a results file stores: where it is and how it ranked."""
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        rank=chunk.rank,
        score=getattr(chunk, "score", None),
        lexical_rank=getattr(chunk, "lexical_rank", None),
        vector_rank=getattr(chunk, "vector_rank", None),
    )


def _as_json(answer: Any) -> dict[str, Any]:
    """The whole answer as plain JSON.

    `AnswerResponse` is a Pydantic model; the tests hand the loop a dataclass instead.
    """
    if hasattr(answer, "model_dump"):
        return answer.model_dump(mode="json")
    if is_dataclass(answer) and not isinstance(answer, type):
        return asdict(answer)
    if isinstance(answer, dict):
        return dict(answer)
    raise TypeError(f"cannot store an answer of type {type(answer).__name__}")


def progress_line(row: QuestionResult) -> str:
    """One line per question, so a long run shows its work as it goes."""
    if row.error and not row.retrieved:
        return f"{row.id}  {row.type:<13} {row.error}"
    scored = row.metrics
    if scored.abstention_correct is None:
        abstention = metrics.NOT_MEASURED
    else:
        abstention = "ok" if scored.abstention_correct else "WRONG"
    line = (
        f"{row.id}  {row.type:<13} "
        f"hit {_flag(scored.page_hit)}  "
        f"recall {metrics.as_score(scored.page_recall):>4}  "
        f"rr {metrics.as_score(scored.reciprocal_rank):>4}  "
        f"cited {_flag(scored.cited_page_hit)}  "
        f"abstain {abstention:<5} "
        f"{len(row.retrieved)} chunks"
    )
    return f"{line}  [{row.error}]" if row.error else line


def _flag(value: float | None) -> str:
    """A yes-or-no metric as 1, 0, or the dash that means it does not apply here."""
    return metrics.NOT_MEASURED if value is None else str(int(value))


def summary_lines(result: RunResult) -> list[str]:
    """The aggregate numbers, recomputed from the rows that were just written."""
    totals = result.aggregates
    failed = len(result.errors)
    manual = totals.manual
    return [
        f"{totals.questions} questions, {totals.answerable} answerable, "
        f"{totals.answered} answered, {failed} failed",
        f"  page_hit@5           {metrics.as_percentage(totals.page_hit)}",
        f"  page_recall@5        {metrics.as_percentage(totals.page_recall)}",
        f"  mrr                  {metrics.as_score(totals.mrr)}",
        f"  cited_page_hit       {metrics.as_percentage(totals.cited_page_hit)}",
        f"  abstention_accuracy  {metrics.as_percentage(totals.abstention_accuracy)}",
        f"  manual_pass          {metrics.as_percentage(manual.rate)} ({manual.reviewed} reviewed)",
    ]


def corpus_from_database(pool: Any, edition: int) -> Corpus:
    """Read which handbook was ingested, so the results file pins the corpus it searched."""
    with pool.connection() as connection:
        row = connection.execute(
            "SELECT sha256 FROM documents WHERE edition = %s", (edition,)
        ).fetchone()
    if row is None:
        raise RunError(
            f"edition {edition} has not been ingested; run: fineprint ingest --edition {edition}"
        )
    return Corpus(edition=edition, sha256=row[0])


def build_retriever(settings: Settings, pool: Any) -> Retriever:
    """The real retriever, with the real embedder behind it.

    `fineprint.retrieval` and `fineprint.providers.factory` are imported here rather than at the
    top of the module so that the metrics, the results file and this loop can be imported — and
    tested — without them.
    """
    from fineprint.providers.factory import get_embedder
    from fineprint.retrieval import Retriever as RealRetriever

    return RealRetriever(pool, get_embedder(settings), settings)


def build_answer_function(settings: Settings, retriever: Retriever) -> AnswerFunction:
    """Bind the retriever and the chat model into something that answers one question."""
    from fineprint.answer import answer_question
    from fineprint.providers.factory import get_chat_model

    chat = get_chat_model(settings)

    def answer(question: str) -> Any:
        return answer_question(question, retriever, chat, top_k=settings.retrieval_top_k)

    return answer


def build_parser() -> argparse.ArgumentParser:
    """Describe the command."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_golden_set",
        description="Run one retrieval configuration over the golden set and record the results.",
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(CONFIGS),
        help="which configuration to run: hybrid fuses both retrievers, the others use one each",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip the answer model; searching by meaning still embeds every question",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"where to write the results file (default: {RESULTS_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the golden set once and write `evals/results/<run_id>.json`."""
    args = build_parser().parse_args(argv)
    settings = Settings()
    mode = CONFIGS[args.config]

    try:
        items = load_golden_set(DEFAULT_GOLDEN_SET)
    except GoldenSetError as error:
        print(f"ERROR: {DEFAULT_GOLDEN_SET} failed validation:\n{error}", file=sys.stderr)
        return EXIT_FAILED

    print(
        f"{args.config}: {len(items)} questions, mode {mode}, k {settings.retrieval_top_k}, "
        f"chunk set {settings.chunk_set}, database {describe_database(settings.database_url)}"
    )

    try:
        with connection_pool(settings.database_url) as pool:
            corpus = corpus_from_database(pool, settings.corpus_edition)
            retriever = build_retriever(settings, pool)
            answer_fn = None if args.retrieval_only else build_answer_function(settings, retriever)
            rows = run_questions(
                items,
                retriever,
                answer_fn,
                mode=mode,
                top_k=settings.retrieval_top_k,
            )
    except RunError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED

    result = RunResult(
        run_id=make_run_id(args.config),
        created_at=now_utc(),
        git=git_state(),
        config=RunConfig(
            name=args.config,
            mode=mode,
            chunk_set=settings.chunk_set,
            top_k=settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
            embedding_model=settings.embedding_model,
            llm_model=None if args.retrieval_only else settings.llm_model,
        ),
        corpus=corpus,
        golden_set=GoldenSetInfo(sha256=sha256_of(DEFAULT_GOLDEN_SET), count=len(items)),
        questions=rows,
    )
    path = save(result, results_path(result.run_id, args.results_dir))

    print()
    for line in summary_lines(result):
        print(line)
    print(f"\nwrote {path}")
    if not args.retrieval_only:
        print(f"next: python -m evals.review {path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
