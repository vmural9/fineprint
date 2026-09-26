"""Score a results file on the four RAGAS metrics, in a second pass over a finished run.

    python -m evals.score evals/results/<run_id>.json
    python -m evals.score evals/results/<run_id>.json --explain q001
    python -m evals.score evals/results/<run_id>.json --rescore

`run_golden_set` searches, answers and records what came back. This pass reads that file and has
a judge model score every question on context recall, context precision, faithfulness and answer
relevance, the four metrics `evals/ragas_metrics.py` defines. The scores go into the same file,
beside the page metrics; the judge's verdicts behind them go under each question's
`scoring_detail`; and the file's `scoring` block records which judge, which embedding model and
which prompts produced them, and how many judge tokens they took.

What a question is scored on depends on what the file holds for it. An unanswerable question gets
none of the four: its reference answer says the handbook does not give this, which leaves no
facts for retrieval to find and no claims to check. A question with no answer, as in a
`--retrieval-only` run, gets context recall and context precision, which judge the passages
against the reference answer. Every other question gets all four.

The judge reads the passages from the results file, never from the database, whose chunks may
have been re-ingested since the run; a file written before results carried chunk text is refused
before any model is called. The file is saved after every question, so an interrupted pass loses
nothing: the same command carries on where it stopped. A judge failure is recorded on its
question, whose scores stay empty, and the pass goes on; the next pass tries that question again.

`--explain QID` prints the working behind one question's four numbers and writes nothing.

Exit codes: 0 when every question was scored or skipped, 1 when the judge failed on at least one,
2 when the file cannot be scored as it stands, in which case nothing is called and nothing written.
"""

import argparse
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evals import metrics
from evals.golden import DEFAULT_GOLDEN_SET, GoldenItem, GoldenSetError, load_golden_set
from evals.ragas_metrics import (
    PROMPTS_SHA256,
    MetricResult,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)
from evals.results import (
    QuestionResult,
    ResultsError,
    RetrievedChunk,
    RunResult,
    Scoring,
    load,
    now_utc,
    sha256_of,
)
from evals.results import save as save_results
from fineprint.config import Settings
from fineprint.providers.base import ChatModel, Embedder
from fineprint.providers.factory import get_embedder

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

# The four metrics, in the order they are computed and printed. Each name is a field of
# `QuestionMetrics` and a key of a scored question's `scoring_detail`.
METRICS = ("context_recall", "context_precision", "faithfulness", "answer_relevance")

# Why a question has no scores, as `scoring_detail["skipped"]` records it, and what --explain
# says about it.
UNANSWERABLE = "unanswerable"
RETRIEVAL_FAILED = "retrieval failed"
WHY_SKIPPED = {
    UNANSWERABLE: (
        "The handbook does not answer this question, so none of the four metrics applies: there "
        "are no facts for retrieval to find and no claims to check. Abstention accuracy is what "
        "measures it."
    ),
    RETRIEVAL_FAILED: (
        "Retrieval failed on this question when the run was made, so there are no passages to "
        "judge."
    ),
}
NO_ANSWER = "The run has no answer for this question, so there is nothing to judge here."

# --explain prints at the review screen's width, so it reads the same in a terminal and on video.
# Each judged item is laid out as: indent, its number, its verdict, then its text and reason.
WIDTH = 88
INDENT = "   "
NUMBER_WIDTH = 4  # "1.  " up to "99. "
VERDICT_WIDTH = 15  # "NOT supported" and two spaces
TEXT_WIDTH = WIDTH - len(INDENT) - NUMBER_WIDTH - VERDICT_WIDTH
LABEL_WIDTH = 21  # the runner's summary lines line their numbers up at this column


@dataclass(frozen=True)
class Scored:
    """One question's four scores and the working behind them.

    `values` holds all four metrics, `None` where one does not apply or the judge failed;
    `detail` is what the results file keeps as the question's `scoring_detail`.
    """

    values: dict[str, float | None]
    detail: dict[str, Any]


def build_judge(settings: Settings, model: str) -> ChatModel:
    """The judge: `model` on Bedrock, through the same chat model class that writes the answers.

    Imported here rather than at the top so that refusing a file, or printing `--help`, never
    loads the Anthropic SDK. The tests replace this function outright, and `build_embedder`.
    """
    from fineprint.providers.bedrock_chat import BedrockChatModel

    return BedrockChatModel(model=model, region=settings.aws_region)


def build_embedder(settings: Settings) -> Embedder:
    """The embedder answer relevance compares questions with: the one retrieval searches with."""
    return get_embedder(settings)


def ranked(row: QuestionResult) -> list[RetrievedChunk]:
    """The chunks retrieved for a question, best-ranked first."""
    return sorted(row.retrieved, key=lambda chunk: chunk.rank)


def passages(row: QuestionResult) -> list[str]:
    """What the judge reads: the retrieved chunks' own text from the results file, best first."""
    texts = [chunk.text for chunk in ranked(row) if chunk.text is not None]
    if len(texts) != len(row.retrieved):
        raise ValueError(f"{row.id}: a retrieved chunk carries no text to judge")
    return texts


def score_question(
    row: QuestionResult, item: GoldenItem, judge: ChatModel, embedder: Embedder
) -> Scored:
    """Score one question on every metric that applies to it.

    Whatever the judge or the embedder raises is left to the caller, which records it.
    """
    if not item.answerable:
        return skipped(UNANSWERABLE)
    if row.error is not None and not row.retrieved:
        return skipped(RETRIEVAL_FAILED)
    texts = passages(row)
    results: dict[str, MetricResult] = {
        "context_recall": context_recall(row.question, item.expected_answer, texts, judge),
        "context_precision": context_precision(row.question, item.expected_answer, texts, judge),
    }
    if row.answer is not None:
        answer = row.answer["answer"]
        results["faithfulness"] = faithfulness(row.question, answer, texts, judge)
        results["answer_relevance"] = answer_relevance(row.question, answer, judge, embedder)
    detail: dict[str, Any] = {name: result.detail for name, result in results.items()}
    detail["judge_input_tokens"] = sum(result.input_tokens for result in results.values())
    detail["judge_output_tokens"] = sum(result.output_tokens for result in results.values())
    values = {name: results[name].value if name in results else None for name in METRICS}
    return Scored(values=values, detail=detail)


def skipped(reason: str) -> Scored:
    """A question none of the four metrics applies to, and why."""
    return Scored(values=dict.fromkeys(METRICS), detail={"skipped": reason})


def failed(error: Exception) -> Scored:
    """A question the judge failed on: no scores at all, and the error that stopped it."""
    return Scored(
        values=dict.fromkeys(METRICS),
        detail={"error": {"type": type(error).__name__, "message": str(error)}},
    )


def already_scored(row: QuestionResult) -> bool:
    """True when a pass scored this question or recorded why it has no scores.

    A question the judge failed on is not: the next pass tries it again.
    """
    return row.scoring_detail is not None and "error" not in row.scoring_detail


def record(row: QuestionResult, scored: Scored) -> None:
    """Write one question's scores into its row, and the working behind them beside it."""
    row.metrics = replace(row.metrics, **scored.values)
    row.scoring_detail = scored.detail


def clear_scores(result: RunResult) -> None:
    """Forget every score in the run, before `--rescore` scores it all again.

    Clearing first means a rescore that is interrupted leaves questions unscored, which the next
    pass picks up, and never leaves scores from two passes side by side in one file.
    """
    for row in result.questions:
        row.metrics = replace(row.metrics, **dict.fromkeys(METRICS))
        row.scoring_detail = None
    result.scoring = None


def token_totals(result: RunResult) -> tuple[int, int]:
    """The judge's input and output tokens behind every score in the file."""
    details = [row.scoring_detail or {} for row in result.questions]
    return (
        sum(detail.get("judge_input_tokens", 0) for detail in details),
        sum(detail.get("judge_output_tokens", 0) for detail in details),
    )


def scoring_block(result: RunResult, judge_model: str, embedding_model: str) -> Scoring:
    """The `scoring` block as it stands after the latest question, with the totals recounted."""
    input_tokens, output_tokens = token_totals(result)
    return Scoring(
        judge_model=judge_model,
        embedding_model=embedding_model,
        prompts_sha256=PROMPTS_SHA256,
        scored_at=now_utc(),
        judge_input_tokens=input_tokens,
        judge_output_tokens=output_tokens,
    )


def score_run(
    result: RunResult,
    golden: dict[str, GoldenItem],
    judge: ChatModel,
    embedder: Embedder,
    *,
    judge_model: str,
    embedding_model: str,
    save: Callable[[], None],
    report: Callable[[str], None] = print,
) -> int:
    """Score every question that has no scores yet, in file order, saving after each one.

    A failure on one question is recorded on it and the pass moves on, so one bad reply cannot
    cost a whole pass. Returns how many questions this pass took on.
    """
    taken = 0
    for row in result.questions:
        if already_scored(row):
            continue
        item = golden[row.id]
        try:
            scored = score_question(row, item, judge, embedder)
        except Exception as error:  # one question's failure is not the pass's failure
            scored = failed(error)
        record(row, scored)
        result.scoring = scoring_block(result, judge_model, embedding_model)
        save()
        report(progress_line(row))
        taken += 1
    return taken


def refusal(result: RunResult, path: Path) -> str | None:
    """Why this file cannot be scored as it stands, or None when it can.

    Checked before any model is called: the passages must be in the file, and the golden set the
    run was made against must be the one whose reference answers the judge will be shown.
    """
    how_to_fix = (
        f"Make a file to score with:\n  python -m evals.run_golden_set --config "
        f"{result.config.name}"
    )
    without_text = [
        row.id for row in result.questions if any(chunk.text is None for chunk in row.retrieved)
    ]
    if without_text:
        return (
            f"{path} predates chunk text in results files: the chunks retrieved for "
            f"{len(without_text)} of its {len(result.questions)} questions carry no text, and "
            "the judge reads passages from the results file only, never from the database. "
            + how_to_fix
        )
    golden_sha256 = sha256_of(DEFAULT_GOLDEN_SET)
    if result.golden_set.sha256 != golden_sha256:
        return (
            f"{path} was run against a different golden set (sha256 "
            f"{result.golden_set.sha256[:12]}…) from the one in {DEFAULT_GOLDEN_SET.name} now "
            f"({golden_sha256[:12]}…), whose reference answers may differ. " + how_to_fix
        )
    return None


def scorer_change(result: RunResult, judge_model: str, embedding_model: str) -> str | None:
    """What differs between the scorer behind the file's scores and this pass's, if anything.

    Scores from two judges, two embedders or two sets of prompts cannot share one file: its
    `scoring` block names only one of each.
    """
    before = result.scoring
    if before is None:
        return None
    changes = []
    if before.judge_model != judge_model:
        changes.append(f"the judge was {before.judge_model}, now {judge_model}")
    if before.embedding_model != embedding_model:
        changes.append(f"the embedding model was {before.embedding_model}, now {embedding_model}")
    if before.prompts_sha256 != PROMPTS_SHA256:
        changes.append(
            f"the prompts' sha256 was {before.prompts_sha256[:12]}…, now {PROMPTS_SHA256[:12]}…"
        )
    return "; ".join(changes) or None


def failed_questions(result: RunResult) -> list[str]:
    """The ids of the questions the judge failed on."""
    return [row.id for row in result.questions if "error" in (row.scoring_detail or {})]


def progress_line(row: QuestionResult) -> str:
    """One line per question as the pass goes, laid out like the runner's."""
    detail = row.scoring_detail or {}
    head = f"{row.id}  {row.type:<13} "
    if "skipped" in detail:
        return f"{head}skipped: {detail['skipped']}"
    if "error" in detail:
        error = detail["error"]
        first_line = next(iter(error["message"].splitlines()), "")
        return f"{head}FAILED  {error['type']}: {first_line}"
    scores = row.metrics
    return (
        f"{head}recall {metrics.as_score(scores.context_recall):>4}  "
        f"precision {metrics.as_score(scores.context_precision):>4}  "
        f"faithfulness {metrics.as_score(scores.faithfulness):>4}  "
        f"relevance {metrics.as_score(scores.answer_relevance):>4}"
    )


def summary_lines(result: RunResult) -> list[str]:
    """The file's scores as they now stand, each averaged over the questions it applies to."""
    skipped_count = sum("skipped" in (row.scoring_detail or {}) for row in result.questions)
    scored_count = sum(already_scored(row) for row in result.questions) - skipped_count
    failures = failed_questions(result)
    totals = result.aggregates
    input_tokens, output_tokens = token_totals(result)
    lines = [
        f"{len(result.questions)} questions, {scored_count} scored, {skipped_count} skipped, "
        f"{len(failures)} failed",
        *(f"  {name:<{LABEL_WIDTH}}{metrics.as_score(getattr(totals, name))}" for name in METRICS),
        f"  {'judge tokens':<{LABEL_WIDTH}}{input_tokens:,} in, {output_tokens:,} out",
    ]
    if failures:
        lines.append(f"failed: {', '.join(failures)} (run the same command again to retry)")
    return lines


# --- --explain ---------------------------------------------------------------------


def explain(
    result: RunResult,
    golden: dict[str, GoldenItem],
    question_id: str,
    *,
    path: Path,
    settings: Settings,
    judge_model: str,
    ask_again: bool,
) -> int:
    """Print the working behind one question's four numbers. Writes nothing.

    A question the file already has scores for is explained from them, with no model call. Any
    other, or any at all with `ask_again`, is scored afresh for the explanation only.
    """
    row = next((row for row in result.questions if row.id == question_id), None)
    if row is None:
        print(f"ERROR: {path} has no question {question_id}", file=sys.stderr)
        return EXIT_REFUSED

    if already_scored(row) and not ask_again:
        stored = row.scoring_detail or {}
        scored = Scored(
            values={name: getattr(row.metrics, name) for name in METRICS}, detail=stored
        )
        judge_name = result.scoring.judge_model if result.scoring else "an unrecorded judge"
        source = f"Scored by {judge_name}; read from {path.name}."
    else:
        judge = build_judge(settings, judge_model)
        embedder = build_embedder(settings)
        try:
            scored = score_question(row, golden[row.id], judge, embedder)
        except Exception as error:
            print(
                f"ERROR: the judge failed on {row.id}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return EXIT_FAILED
        source = f"Scored just now by {judge_model}; nothing is written to {path.name}."

    print("\n".join(explanation(row, scored, source)))
    return EXIT_OK


def explanation(row: QuestionResult, scored: Scored, source: str) -> list[str]:
    """One question's scores and every verdict behind them, as lines of plain text."""
    detail = scored.detail
    lines = [
        f"{row.id} · {row.type} · {row.category} · {row.difficulty}",
        source,
        "",
        "QUESTION",
        _wrap(row.question),
    ]
    if row.answer is not None:
        lines += ["", "ANSWER", _wrap(str(row.answer.get("answer", "")))]
    if "skipped" in detail:
        reason = detail["skipped"]
        lines += ["", _wrap(WHY_SKIPPED.get(reason, f"Skipped: {reason}."), indent="")]
    else:
        values = scored.values
        lines += _recall_section(detail["context_recall"], values["context_recall"])
        lines += _precision_section(
            detail["context_precision"], values["context_precision"], ranked(row)
        )
        lines += _faithfulness_section(detail.get("faithfulness"), values["faithfulness"])
        lines += _relevance_section(detail.get("answer_relevance"), values["answer_relevance"])
    return lines + _scores_section(scored)


def _recall_section(entry: dict[str, Any], value: float | None) -> list[str]:
    """Each sentence of the reference answer, and whether the passages support it."""
    lines = _heading(
        "CONTEXT RECALL", value, "Is each sentence of the expected answer in the passages?"
    )
    for number, sentence in enumerate(entry["sentences"], start=1):
        verdict = "supported" if sentence["supported"] else "NOT supported"
        lines += _judged(number, verdict, sentence["sentence"], sentence["reason"])
    return lines


def _precision_section(
    entry: dict[str, Any], value: float | None, chunks: list[RetrievedChunk]
) -> list[str]:
    """Each passage in rank order, where it sits, and whether it helps reach the answer."""
    lines = _heading(
        "CONTEXT PRECISION",
        value,
        "Does each passage, in rank order, help reach the expected answer?",
    )
    if not entry["passages"]:
        lines.append(f"{INDENT}No passages were retrieved, which scores 0.")
    for passage in entry["passages"]:
        verdict = "useful" if passage["useful"] else "NOT useful"
        where = _passage_line(chunks[passage["rank"] - 1])
        lines += _judged(passage["rank"], verdict, where, passage["reason"])
    return lines


def _faithfulness_section(entry: dict[str, Any] | None, value: float | None) -> list[str]:
    """Each claim the answer makes, and whether the passages support it."""
    lines = _heading("FAITHFULNESS", value, "Is each claim the answer makes in the passages?")
    if entry is None:
        return [*lines, f"{INDENT}{NO_ANSWER}"]
    if entry["no_claims"]:
        return [
            *lines,
            f"{INDENT}The answer makes no factual claims, so there is nothing to check.",
        ]
    for number, claim in enumerate(entry["claims"], start=1):
        verdict = "supported" if claim["supported"] else "NOT supported"
        lines += _judged(number, verdict, claim["claim"], claim["reason"])
    return lines


def _relevance_section(entry: dict[str, Any] | None, value: float | None) -> list[str]:
    """The questions written back from the answer, each with its similarity to the real one."""
    lines = _heading(
        "ANSWER RELEVANCE",
        value,
        "Questions written back from the answer alone, and how close each is to the one asked:",
    )
    if entry is None:
        return [*lines, f"{INDENT}{NO_ANSWER}"]
    for question in entry["questions"]:
        lines += _item(f"{INDENT}{question['similarity']:.2f}  ", question["question"])
    if entry["noncommittal"]:
        lines.append(
            f"{INDENT}The judge found the answer noncommittal, which scores 0 whatever the "
            "similarities."
        )
    return lines


def _scores_section(scored: Scored) -> list[str]:
    """The four numbers, and the judge tokens they took."""
    lines = ["", "SCORES"]
    lines += [
        f"{INDENT}{name:<{LABEL_WIDTH}}{metrics.as_score(scored.values[name])}" for name in METRICS
    ]
    if "judge_input_tokens" in scored.detail:
        tokens_in = scored.detail["judge_input_tokens"]
        tokens_out = scored.detail["judge_output_tokens"]
        lines.append(f"{INDENT}{'judge tokens':<{LABEL_WIDTH}}{tokens_in:,} in, {tokens_out:,} out")
    return lines


def _heading(title: str, value: float | None, asks: str) -> list[str]:
    """A section's title with its score, and the question the judge was asked."""
    return ["", f"{title}  {metrics.as_score(value)}", asks]


def _judged(number: int, verdict: str, text: str, reason: str) -> list[str]:
    """One judged item: its number, verdict and text, then the judge's reason under the text."""
    head = f"{INDENT}{f'{number}.':<{NUMBER_WIDTH}}{verdict:<{VERDICT_WIDTH}}"
    return _item(head, text) + _item(" " * len(head), f"reason: {reason}")


def _item(head: str, text: str) -> list[str]:
    """`text` wrapped to the screen, starting after `head` and hanging under its first line."""
    lines = _wrapped(text, WIDTH - len(head)) or [""]
    return [head + lines[0], *(" " * len(head) + line for line in lines[1:])]


def _passage_line(chunk: RetrievedChunk) -> str:
    """Where a passage sits in the handbook, and how it begins, on one line."""
    first, last = chunk.page_start, chunk.page_end
    pages = f"p.{first}" if first == last else f"pp.{first}-{last}"
    room = TEXT_WIDTH - len(pages) - 3  # a space and two quote marks
    return f'{pages} "{textwrap.shorten(chunk.text or "", width=room, placeholder=" …")}"'


def _wrap(text: str, indent: str = INDENT) -> str:
    """Indent and wrap a paragraph to the screen's width."""
    return "\n".join(indent + line for line in _wrapped(text, WIDTH - len(indent)))


def _wrapped(text: str, width: int) -> list[str]:
    """Wrap text to `width`, never breaking a word, a web address or a hyphenated term."""
    return textwrap.wrap(
        " ".join(text.split()), width=width, break_long_words=False, break_on_hyphens=False
    )


# --- the command -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Describe the command."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.score",
        description="Score a results file on the four RAGAS metrics, writing the scores into it.",
    )
    parser.add_argument("results", type=Path, help="the results file to score, in place")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="score every question again, not only those without scores; with --explain, ask "
        "the judge afresh instead of reading the file",
    )
    parser.add_argument(
        "--explain",
        metavar="QID",
        help="print the judge's working behind one question's scores, and write nothing",
    )
    parser.add_argument(
        "--judge-model",
        metavar="ID",
        help="Bedrock model id of the judge (default: the JUDGE_MODEL setting)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score one results file in place, or explain one question's scores."""
    args = build_parser().parse_args(argv)
    path: Path = args.results

    try:
        result = load(path)
        golden = {item.id: item for item in load_golden_set(DEFAULT_GOLDEN_SET)}
    except (ResultsError, GoldenSetError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_REFUSED
    problem = refusal(result, path)
    if problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return EXIT_REFUSED

    settings = Settings()
    judge_model = args.judge_model or settings.judge_model
    embedding_model = settings.embedding_model
    if args.explain is not None:
        return explain(
            result,
            golden,
            args.explain,
            path=path,
            settings=settings,
            judge_model=judge_model,
            ask_again=args.rescore,
        )

    if args.rescore:
        clear_scores(result)
    else:
        change = scorer_change(result, judge_model, embedding_model)
        if change:
            print(
                f"ERROR: {path} holds scores from a different scorer: {change}. One file cannot "
                "hold scores from both; pass --rescore to score every question again.",
                file=sys.stderr,
            )
            return EXIT_REFUSED

    print(
        f"{result.run_id}: {len(result.questions)} questions, judge {judge_model}, "
        f"embeddings {embedding_model}"
    )
    kept = sum(already_scored(row) for row in result.questions)
    if kept:
        print(f"{kept} already scored and kept; --rescore scores every question again")
    taken = score_run(
        result,
        golden,
        build_judge(settings, judge_model),
        build_embedder(settings),
        judge_model=judge_model,
        embedding_model=embedding_model,
        save=lambda: save_results(result, path),
    )

    print()
    for line in summary_lines(result):
        print(line)
    print(f"\nwrote {path}" if taken else f"\nnothing left to score; {path} is unchanged")
    return EXIT_FAILED if failed_questions(result) else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
