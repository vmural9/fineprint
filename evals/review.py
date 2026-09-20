"""Read the generated answers one by one and record a human pass or fail on each.

    python -m evals.review evals/results/20260921T031500Z_hybrid.json
    python -m evals.review evals/results/20260921T031500Z_hybrid.json --only-unreviewed

The automatic metrics say whether the right pages came back and whether the answer cited them.
They cannot say whether the answer is *correct*: the retrieved page can be right while the
sentence built from it is wrong. That judgement is the reviewer's, and it is recorded next to
the numbers instead of replacing them. Part 4 reuses these labels to calibrate the LLM judge,
which is why the note, the reviewer and the time are stored with every verdict.

The file is saved after every single verdict, so a session interrupted halfway loses nothing.
"""

import argparse
import os
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from evals.golden import DEFAULT_GOLDEN_SET, GoldenSetError, load_golden_set
from evals.metrics import FAIL, PASS
from evals.results import QuestionResult, ResultsError, Review, RunResult, load, now_utc
from evals.results import save as save_results

EXIT_OK = 0
EXIT_FAILED = 1

# The screen is a fixed width so the layout reads the same in a terminal and on video.
WIDTH = 88
INDENT = "   "
# A quote is shortened rather than wrapped, so every citation stays two lines tall.
QUOTE_LIMIT = WIDTH - 10

SKIP = "skip"
QUIT = "quit"
CHOICES = {
    "p": PASS,
    "pass": PASS,
    "f": FAIL,
    "fail": FAIL,
    "s": SKIP,
    "skip": SKIP,
    "q": QUIT,
    "quit": QUIT,
}
PROMPT = "  [p]ass  [f]ail  [s]kip  [q]uit and save > "
NOTE_PROMPT = "  note (optional) > "


def review_answers(
    result: RunResult,
    expected: Mapping[str, str],
    *,
    reviewer: str | None,
    input_stream: TextIO,
    output: TextIO,
    save: Callable[[], None],
    only_unreviewed: bool = False,
) -> int:
    """Show every answered question and record the verdicts. Returns how many were recorded.

    `save` is called after each verdict rather than at the end, and the streams are arguments,
    so the tests can drive a whole session with no terminal.
    """
    answered = [row for row in result.questions if row.answered]
    todo = [row for row in answered if not (only_unreviewed and row.verdict is not None)]

    _write(output, f"Reviewing {result.run_id}  ({result.config.name}, k {result.config.top_k})")
    unanswered = len(result.questions) - len(answered)
    if unanswered:
        verb = "has" if unanswered == 1 else "have"
        _write(
            output,
            f"{unanswered} of the {len(result.questions)} questions {verb} no answer "
            "and cannot be reviewed.",
        )
    if not todo:
        _write(output, "Nothing left to review.")
        return 0
    _write(output, f"{len(todo)} to review. The file is saved after every verdict.")

    recorded = 0
    for position, row in enumerate(todo, start=1):
        _write(output, render_question(row, expected.get(row.id), position, len(todo)))
        choice = _ask(input_stream, output, PROMPT, CHOICES)
        if choice is None or choice == QUIT:
            break
        if choice == SKIP:
            continue
        note = _read_line(input_stream, output, NOTE_PROMPT) or ""
        row.review = Review(
            verdict=choice, note=note.strip(), reviewer=reviewer, reviewed_at=now_utc()
        )
        save()
        recorded += 1

    reviewed = sum(row.verdict is not None for row in answered)
    _write(
        output,
        f"\nRecorded {recorded} {'verdict' if recorded == 1 else 'verdicts'}. "
        f"{reviewed} of {len(answered)} answers reviewed.",
    )
    return recorded


def render_question(
    row: QuestionResult, expected_answer: str | None, position: int, total: int
) -> str:
    """One screen: the question, what a correct answer must say, and what the model said."""
    lines = [
        "",
        "=" * WIDTH,
        f" {row.id}   {position} of {total}   {row.category} · {row.type} · {row.difficulty}",
        "=" * WIDTH,
        "",
        " QUESTION",
        _wrap(row.question),
        "",
        f" EXPECTED  ({_expected_pages(row)})",
        _wrap(expected_answer or "(this question is not in the current golden set)"),
        "",
        f" ANSWER  ({_answer_header(row.answer)})",
        _wrap(_text(row.answer, "answer", "(no answer text)")),
    ]
    citations = _citations(row.answer)
    lines.append("")
    lines.append(" CITED")
    lines.extend(_citation_lines(citations) if citations else [f"{INDENT}(nothing cited)"])
    dropped = (row.answer or {}).get("dropped_citations") or 0
    if dropped:
        noun = "citation" if dropped == 1 else "citations"
        lines.append(f"{INDENT}{dropped} {noun} named a chunk that was not retrieved")
    if row.error:
        lines.append(f"{INDENT}note: {row.error}")
    lines.append("")
    return "\n".join(lines)


def _expected_pages(row: QuestionResult) -> str:
    """How the expected pages are announced above the expected answer."""
    if not row.answerable:
        return "the handbook does not answer this"
    if not row.expected_pages:
        return "no expected pages"
    label = "page" if len(row.expected_pages) == 1 else "pages"
    return f"{label} {', '.join(str(page) for page in row.expected_pages)}"


def _answer_header(answer: dict[str, Any] | None) -> str:
    """The one-line verdict the model gave itself: did it find the answer, and how sure is it."""
    found = (answer or {}).get("found_in_handbook")
    found_text = {True: "found in handbook", False: "NOT found in handbook"}.get(found, "unknown")
    return f"{found_text} · confidence {_text(answer, 'confidence', 'unknown')}"


def _citation_lines(citations: Sequence[dict[str, Any]]) -> list[str]:
    """One pair of lines per citation: where it points, and the quote it stands on."""
    labels = [_page_label(citation) for citation in citations]
    width = max(len(label) for label in labels)
    lines = []
    for citation, label in zip(citations, labels, strict=True):
        verified = "quote verified" if citation.get("quote_verified") else "QUOTE NOT VERIFIED"
        lines.append(
            f"{INDENT}{label:<{width}}  chunk {citation.get('chunk_id', '?')}   {verified}"
        )
        lines.append(f'{INDENT}  "{_shorten(str(citation.get("quote", "")))}"')
    return lines


def _page_label(citation: Mapping[str, Any]) -> str:
    """`p.23` for one page, `pp.27-28` for a chunk that spans a page break."""
    start, end = citation.get("page_start"), citation.get("page_end")
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


def _citations(answer: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The answer's citations, or an empty list when it cited nothing."""
    citations = (answer or {}).get("citations") or []
    return [citation for citation in citations if isinstance(citation, dict)]


def _text(source: Mapping[str, Any] | None, key: str, fallback: str) -> str:
    """Read one string out of an answer, tolerating an answer that is missing it."""
    value = (source or {}).get(key)
    return str(value) if value else fallback


def _shorten(text: str) -> str:
    """Keep a quote to one readable span."""
    return textwrap.shorten(text, width=QUOTE_LIMIT, placeholder=" …")


def _wrap(text: str) -> str:
    """Indent and wrap a paragraph to the review screen's width."""
    return textwrap.fill(
        " ".join(text.split()),
        width=WIDTH - len(INDENT),
        initial_indent=INDENT,
        subsequent_indent=INDENT,
    )


def _write(output: TextIO, text: str) -> None:
    """Write one block and flush, so a prompt never waits behind a buffer."""
    output.write(text + "\n")
    output.flush()


def _read_line(input_stream: TextIO, output: TextIO, prompt: str) -> str | None:
    """Show a prompt and read one line. None means the input ended."""
    output.write(prompt)
    output.flush()
    line = input_stream.readline()
    return None if line == "" else line.rstrip("\n")


def _ask(
    input_stream: TextIO, output: TextIO, prompt: str, choices: Mapping[str, str]
) -> str | None:
    """Ask until the answer is one we understand. None means the input ended."""
    while True:
        answer = _read_line(input_stream, output, prompt)
        if answer is None:
            return None
        choice = choices.get(answer.strip().lower())
        if choice is not None:
            return choice
        _write(output, "  please answer p, f, s or q")


def expected_answers() -> Mapping[str, str]:
    """What a correct answer must say, per question id, from the golden set.

    The results file does not carry the expected answer, because the golden set is the one
    place it is written down and verified against the PDF.
    """
    try:
        return {item.id: item.expected_answer for item in load_golden_set(DEFAULT_GOLDEN_SET)}
    except GoldenSetError:
        return {}


def build_parser() -> argparse.ArgumentParser:
    """Describe the command."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.review",
        description="Record a human pass or fail on each generated answer in a results file.",
    )
    parser.add_argument("results", type=Path, help="the results file to review, in place")
    parser.add_argument(
        "--only-unreviewed",
        action="store_true",
        help="show only the answers nobody has judged yet",
    )
    parser.add_argument(
        "--reviewer",
        default=os.environ.get("USER"),
        help="who is reviewing; stored with every verdict (default: $USER)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    input_stream: TextIO | None = None,
    output: TextIO | None = None,
) -> int:
    """Review one results file, writing each verdict back into it as it is given."""
    args = build_parser().parse_args(argv)
    reading = input_stream or sys.stdin
    writing = output or sys.stdout

    try:
        result = load(args.results)
    except ResultsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED

    _write(writing, f"\n{args.results}")
    review_answers(
        result,
        expected_answers(),
        reviewer=args.reviewer,
        input_stream=reading,
        output=writing,
        save=lambda: save_results(result, args.results),
        only_unreviewed=args.only_unreviewed,
    )
    _write(writing, f"Saved to {args.results}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
