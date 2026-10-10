"""Run one configuration over the golden set and write a results file.

    python -m evals.run_golden_set --config hybrid
    python -m evals.run_golden_set --config sections+rerank
    python -m evals.run_golden_set --config lexical-only --retrieval-only

`evals/configs.py` registers seven configurations. Part 1 left three: `hybrid` fuses both
retrievers, `vector-only` searches by meaning alone, and `lexical-only` searches by words alone.
Part 2 adds `hybrid+rerank`, where Cohere Rerank 3.5 re-orders the fused list before the top-k
cut, and `sections` / `sections+rerank`, which run hybrid retrieval — and its re-ranked variant —
over the `sections` chunk set instead of `fixed-220w`. Part 3 adds `agent`, which retrieves like
`sections+rerank` and then lets the model call two tools before it answers. `--config` names one
of the seven; every other setting for the run comes from `settings_for(config, Settings())`.

Every question runs inside a trace (`fineprint.tracing`): the session id is the run id, the tags
are `eval` and the configuration's name, and the question id is in the metadata. The trace's
`TraceRecord` (its latency, its cost and the calls inside it) is stored on the question's row, and
the summary reports the run's cost, its p95 latency and any question the fallback model answered.
With `TRACING_PROVIDER=langfuse` the same traces are also sent to Langfuse; with the default,
"none", nothing leaves the process and cost and latency are still measured.

`--retrieval-only` skips the answer model, which is the expensive call, but a run is neither free
nor offline: every mode that searches by meaning embeds each question, and a configured
re-ranker is one more Bedrock call per question on top of that. Only `lexical-only
--retrieval-only` calls no model at all.

A failure on one question is recorded on that question and the run carries on, so one bad
question cannot cost a whole run. The metrics of a failed question are left empty rather than
counted as zero, and the summary says how many failed. The results file is written either way;
the process exit code is what tells a script or CI that a run was not clean.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import psycopg
from psycopg_pool import PoolTimeout

from evals import metrics
from evals.configs import CONFIGS, EvalConfig, settings_for
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
from fineprint.pricing import PRICES
from fineprint.providers.factory import get_reranker, get_tracer
from fineprint.tracing import Tracer

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2


class RunError(RuntimeError):
    """Raised when the run cannot start at all, as opposed to one question failing."""


class Retriever(Protocol):
    """The part of `fineprint.retrieval.Retriever` this runner uses."""

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> Sequence[Any]: ...


# Given a question and the chunks already retrieved for it, produce an `AnswerResponse`. The
# runner binds the chat model into it, so the loop only has to hand it a question and the chunks
# it already retrieved — retrieval happens once, in the loop below, never again inside the answer.
AnswerFunction = Callable[[str, Sequence[Any]], Any]


def run_questions(
    items: Sequence[GoldenItem],
    retriever: Retriever,
    answer_fn: AnswerFunction | None = None,
    *,
    mode: str,
    top_k: int,
    report: Callable[[str], None] = print,
    tracer: Tracer | None = None,
    run_id: str | None = None,
    config_name: str | None = None,
) -> list[QuestionResult]:
    """Search for every question, optionally answer it, and score what came back.

    The retriever and the answer function are arguments rather than globals so that the tests
    can drive this loop with fakes and no database, no AWS, and no network. Given a `tracer`,
    each question is one trace, with `run_id` as its session and `config_name` among its tags, and
    its `TraceRecord` is stored on the row; the retriever and the answer function should have
    been built with the same tracer so their calls land inside it.
    """
    rows: list[QuestionResult] = []
    for item in items:
        row = _run_one(
            item,
            retriever,
            answer_fn,
            mode=mode,
            top_k=top_k,
            tracer=tracer,
            run_id=run_id,
            config_name=config_name,
        )
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
    tracer: Tracer | None,
    run_id: str | None,
    config_name: str | None,
) -> QuestionResult:
    """One question inside its trace, if there is a tracer; the trace's record goes on the row."""
    if tracer is None:
        return _answer_one(item, retriever, answer_fn, mode=mode, top_k=top_k)
    with tracer.trace(
        "eval-question",
        input=item.question,
        session_id=run_id,
        tags=["eval", config_name] if config_name else ["eval"],
        metadata={"question_id": item.id},
    ) as trace:
        row = _answer_one(item, retriever, answer_fn, mode=mode, top_k=top_k)
        if row.answer is not None:
            trace.set_output(row.answer.get("answer"))
    row.trace = trace.record
    return row


def _answer_one(
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
            answer = answer_fn(item.question, chunks)
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
    """Keep the parts of a retrieved chunk a results file stores: its text, where it is, and how
    it ranked (decision D5 — from part 2 on, the text is stored too, since the scoring pass
    reads contexts only from this file)."""
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        rank=chunk.rank,
        score=getattr(chunk, "score", None),
        lexical_rank=getattr(chunk, "lexical_rank", None),
        vector_rank=getattr(chunk, "vector_rank", None),
        ordinal=getattr(chunk, "ordinal", None),
        section=getattr(chunk, "section", None),
        text=getattr(chunk, "text", None),
        fused_rank=getattr(chunk, "fused_rank", None),
        rerank_score=getattr(chunk, "rerank_score", None),
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


def fallback_questions(result: RunResult) -> list[str]:
    """The ids of the questions whose answer came from the fallback model."""
    return [
        row.id
        for row in result.questions
        if row.trace is not None
        and any(call.kind == "generation" and call.fallback for call in row.trace.calls)
    ]


def summary_lines(result: RunResult) -> list[str]:
    """The aggregate numbers, recomputed from the rows that were just written.

    Cost, latency and fallbacks come from the same aggregates the scoreboard prints, so they are
    dashes unless every question has a trace with a cost.
    """
    totals = result.aggregates
    failed = len(result.errors)
    manual = totals.manual
    total_cost = (
        None if totals.cost_per_query_usd is None else totals.cost_per_query_usd * totals.questions
    )
    fallbacks = metrics.NOT_MEASURED if totals.fallbacks is None else totals.fallbacks
    lines = [
        f"{totals.questions} questions, {totals.answerable} answerable, "
        f"{totals.answered} answered, {failed} failed",
        f"  page_hit@5           {metrics.as_percentage(totals.page_hit)}",
        f"  page_recall@5        {metrics.as_percentage(totals.page_recall)}",
        f"  mrr                  {metrics.as_score(totals.mrr)}",
        f"  cited_page_hit       {metrics.as_percentage(totals.cited_page_hit)}",
        f"  abstention_accuracy  {metrics.as_percentage(totals.abstention_accuracy)}",
        f"  manual_pass          {metrics.as_percentage(manual.rate)} ({manual.reviewed} reviewed)",
        f"  cost                 {metrics.as_money(total_cost)} in all, "
        f"{metrics.as_money(totals.cost_per_query_usd)} a question",
        f"  p95 latency          {metrics.as_seconds(totals.p95_latency_ms)}",
        f"  fallbacks            {fallbacks}",
    ]
    answered_by_fallback = fallback_questions(result)
    if answered_by_fallback:
        model = result.config.fallback_model or "the fallback model"
        lines.append(f"  answered by {model}: {', '.join(answered_by_fallback)}")
    return lines


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


def build_retriever(settings: Settings, pool: Any, tracer: Tracer | None = None) -> Retriever:
    """The real retriever, with the real embedder — and a re-ranker, when this configuration
    turned one on — behind it. All three share `tracer`, so their calls land in the trace of the
    question being run.

    `fineprint.retrieval` is imported here rather than at the top of the module so that the
    metrics, the results file and this loop can be imported — and tested — without it.
    `get_reranker` is imported at the top instead, precisely so a test can replace it, the same
    way the other module-level imports below are replaced.
    """
    from fineprint.providers.factory import get_embedder
    from fineprint.retrieval import Retriever as RealRetriever

    return RealRetriever(
        pool,
        get_embedder(settings, tracer),
        settings,
        get_reranker(settings, tracer),
        tracer,
    )


def build_answer_function(
    settings: Settings,
    retriever: Retriever,
    *,
    tools: bool = False,
    today: date | None = None,
    tracer: Tracer | None = None,
) -> AnswerFunction:
    """Bind the chat model into something that answers one question from its own chunks.

    `retriever` is threaded through only because `answer_question` still requires one; passing
    `chunks` means it never calls it, so the loop's own single retrieval is the only search each
    question causes. The chat model, fallback included, is traced with `tracer`. With `tools` the
    answer comes from the agent, which may call the two tools first and treats `today` as the
    date when the model gives none.
    """
    from fineprint.answer import answer_question
    from fineprint.providers.factory import get_chat_model

    chat = get_chat_model(settings, tracer)

    if tools:
        from fineprint.agent import answer_with_tools

        def answer_with_the_tools(question: str, chunks: Sequence[Any]) -> Any:
            return answer_with_tools(
                question,
                retriever,
                chat,
                chunks=chunks,
                today=today,
                max_tool_turns=settings.agent_max_tool_turns,
                tracer=tracer,
            )

        return answer_with_the_tools

    def answer(question: str, chunks: Sequence[Any]) -> Any:
        return answer_question(question, retriever, chat, chunks=chunks, tracer=tracer)

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
        help="which configuration to run; see evals/configs.py and evals/README.md for what "
        "each one is",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip the answer model; searching by meaning still embeds every question",
    )
    parser.add_argument(
        "--only",
        metavar="IDS",
        help="run only these golden-set questions, comma-separated (for example q001,q006), as a "
        "smoke test; needs a --results-dir other than the default, and the file it writes never "
        "becomes a scoreboard row",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"where to write the results file (default: {RESULTS_DIR})",
    )
    return parser


def preflight_problem(settings: Settings, *, retrieval_only: bool) -> str | None:
    """Why this run would be spoiled before it starts, or None when it is fit to pay for.

    A measured run must not inject faults: with `FAULT_INJECTION` left set, answers could be
    written by the fallback model or fail outright, and the file would still say otherwise. And
    every model the run calls needs a price in `PRICES`, or the cost and p95 columns of its row
    come out blank after the money is spent.
    """
    if settings.fault_injection:
        return (
            f"FAULT_INJECTION={settings.fault_injection} is set; a measured run must not inject "
            "faults. Unset it and start again."
        )
    models = {"EMBEDDING_MODEL": settings.embedding_model}
    if settings.reranker_provider != "none":
        models["RERANKER_MODEL"] = settings.reranker_model
    if not retrieval_only:
        models["LLM_MODEL"] = settings.llm_model
        if settings.fallback_llm_model:
            models["FALLBACK_LLM_MODEL"] = settings.fallback_llm_model
    for variable, model in models.items():
        if model not in PRICES:
            return (
                f"{variable}={model} has no entry in PRICES (src/fineprint/pricing.py), so the "
                "run's cost would be blank. Add its price or use a priced model."
            )
    return None


def select_questions(items: Sequence[GoldenItem], only: str) -> tuple[list[GoldenItem], list[str]]:
    """The golden-set questions named in `only` (comma-separated ids), in golden-set order.

    Raises `RunError` for an id the golden set does not have, or for no id at all.
    """
    wanted = [part.strip() for part in only.split(",") if part.strip()]
    if not wanted:
        raise RunError("--only needs at least one question id, for example q001,q006")
    known = {item.id for item in items}
    unknown = [question_id for question_id in wanted if question_id not in known]
    if unknown:
        raise RunError(f"--only names {', '.join(unknown)}, which the golden set does not have")
    chosen = [item for item in items if item.id in set(wanted)]
    return chosen, [item.id for item in chosen]


def run_date() -> date:
    """The day of the run: what the enrollment tool treats as today, and the file records."""
    return date.today()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the golden set once and write `evals/results/<run_id>.json`."""
    args = build_parser().parse_args(argv)
    eval_config: EvalConfig = CONFIGS[args.config]
    settings = settings_for(eval_config, Settings())
    mode = eval_config.mode

    problem = preflight_problem(settings, retrieval_only=args.retrieval_only)
    if problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        items = load_golden_set(DEFAULT_GOLDEN_SET)
    except GoldenSetError as error:
        print(f"ERROR: {DEFAULT_GOLDEN_SET} failed validation:\n{error}", file=sys.stderr)
        return EXIT_FAILED

    golden_count = len(items)
    only: list[str] | None = None
    if args.only is not None:
        try:
            items, only = select_questions(items, args.only)
        except RunError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return EXIT_FAILED
        if args.results_dir.resolve() == RESULTS_DIR.resolve():
            print(
                "ERROR: --only needs a --results-dir other than evals/results/, because the "
                "scoreboard reads the latest file there and a partial run must not become a row",
                file=sys.stderr,
            )
            return EXIT_FAILED

    reranker_note = f", reranker {eval_config.reranker}" if eval_config.reranker else ""
    print(
        f"{args.config}: {len(items)} questions, mode {mode}, k {settings.retrieval_top_k}, "
        f"chunk set {settings.chunk_set}{reranker_note}, "
        f"database {describe_database(settings.database_url)}"
    )

    run_id = make_run_id(args.config)
    today = run_date()
    tracer = get_tracer(settings)
    try:
        with connection_pool(settings.database_url) as pool:
            corpus = corpus_from_database(pool, settings.corpus_edition)
            retriever = build_retriever(settings, pool, tracer)
            answer_fn = (
                None
                if args.retrieval_only
                else build_answer_function(
                    settings, retriever, tools=eval_config.tools, today=today, tracer=tracer
                )
            )
            rows = run_questions(
                items,
                retriever,
                answer_fn,
                mode=mode,
                top_k=settings.retrieval_top_k,
                tracer=tracer,
                run_id=run_id,
                config_name=args.config,
            )
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
    finally:
        tracer.flush()
        tracer.shutdown()

    answering = not args.retrieval_only
    result = RunResult(
        run_id=run_id,
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
            llm_model=settings.llm_model if answering else None,
            reranker=eval_config.reranker,
            rerank_candidates=settings.rerank_candidates if eval_config.reranker else None,
            tools=eval_config.tools,
            tools_today=today.isoformat() if eval_config.tools else None,
            agent_max_tool_turns=settings.agent_max_tool_turns if eval_config.tools else None,
            fallback_model=(settings.fallback_llm_model or None) if answering else None,
            tracing=settings.tracing_provider,
            langfuse_base_url=(
                settings.langfuse_base_url if settings.tracing_provider == "langfuse" else None
            ),
            only=only,
        ),
        corpus=corpus,
        golden_set=GoldenSetInfo(sha256=sha256_of(DEFAULT_GOLDEN_SET), count=golden_count),
        questions=rows,
    )
    path = save(result, results_path(result.run_id, args.results_dir))

    print()
    for line in summary_lines(result):
        print(line)
    print(f"\nwrote {path}")
    if not args.retrieval_only:
        print(f"next: python -m evals.review {path}")
    return EXIT_FAILED if result.errors else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
