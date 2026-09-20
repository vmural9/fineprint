"""Tests for the manual review tool, driven through injected input and output streams."""

import io
import json
from pathlib import Path

from evals.metrics import QuestionMetrics
from evals.results import (
    Corpus,
    GitState,
    GoldenSetInfo,
    QuestionResult,
    RetrievedChunk,
    Review,
    RunConfig,
    RunResult,
    load,
    results_path,
    save,
)
from evals.review import main, review_answers

HYBRID = RunConfig(
    name="hybrid",
    mode="hybrid",
    chunk_set="fixed-220w",
    top_k=5,
    candidates=20,
    rrf_k=60,
    embedding_model="amazon.titan-embed-text-v2:0",
    llm_model="us.anthropic.claude-opus-5",
)

ANSWER = {
    "question": "How much is the Part B premium?",
    "answer": "The standard Part B premium in 2026 is $202.90 a month.",
    "found_in_handbook": True,
    "citations": [
        {
            "chunk_id": 141,
            "page_start": 23,
            "page_end": 23,
            "quote": "The standard Part B premium amount in 2026 is $202.90.",
            "quote_verified": True,
        },
        {
            "chunk_id": 147,
            "page_start": 27,
            "page_end": 28,
            "quote": "a quote the model made up",
            "quote_verified": False,
        },
    ],
    "dropped_citations": 0,
    "confidence": "high",
}

EXPECTED = {"q001": "The standard Part B premium in 2026 is $202.90 a month."}


def question(
    item_id: str = "q001",
    *,
    answer: dict | None = None,
    verdict: str | None = None,
    answerable: bool = True,
) -> QuestionResult:
    """One row of a results file, answered unless `answer` is left out."""
    return QuestionResult(
        id=item_id,
        question="How much is the Part B premium?",
        type="lookup" if answerable else "unanswerable",
        category="costs",
        difficulty="easy",
        answerable=answerable,
        expected_pages=[23] if answerable else [],
        retrieved=[RetrievedChunk(chunk_id=141, page_start=23, page_end=23, rank=1)],
        metrics=QuestionMetrics(1.0, 1.0, 1.0, 1.0, 1.0),
        answer=answer,
        review=Review(verdict=verdict),
    )


def run(*questions: QuestionResult) -> RunResult:
    """A results file in memory, holding the given rows."""
    return RunResult(
        run_id="20260921T031500Z_hybrid",
        created_at="2026-09-21T03:15:00Z",
        git=GitState(commit="0123456", dirty=False),
        config=HYBRID,
        corpus=Corpus(edition=2026, sha256="d7a341bc3d2d"),
        golden_set=GoldenSetInfo(sha256="ab" * 32, count=len(questions)),
        questions=list(questions),
    )


def review(result: RunResult, typed: str, *, only_unreviewed: bool = False):
    """Run a whole review session against typed-in input. Returns the output and the saves."""
    output = io.StringIO()
    saves: list[int] = []
    recorded = review_answers(
        result,
        EXPECTED,
        reviewer="saint",
        input_stream=io.StringIO(typed),
        output=output,
        save=lambda: saves.append(1),
        only_unreviewed=only_unreviewed,
    )
    return recorded, output.getvalue(), len(saves)


# --- recording a verdict ----------------------------------------------------------


def test_p_records_a_pass_with_the_note_the_reviewer_typed():
    result = run(question("q001", answer=ANSWER))

    recorded, _, _ = review(result, "p\ngot the premium and the IRMAA rule right\n")

    review_row = result.questions[0].review
    assert recorded == 1
    assert review_row.verdict == "pass"
    assert review_row.note == "got the premium and the IRMAA rule right"
    assert review_row.reviewer == "saint"
    assert review_row.reviewed_at.endswith("Z")


def test_f_records_a_fail_and_an_empty_note_is_allowed():
    result = run(question("q001", answer=ANSWER))

    recorded, _, _ = review(result, "f\n\n")

    assert recorded == 1
    assert result.questions[0].review.verdict == "fail"
    assert result.questions[0].review.note == ""


def test_s_skips_the_question_and_leaves_it_unreviewed():
    result = run(question("q001", answer=ANSWER), question("q002", answer=ANSWER))

    recorded, _, saves = review(result, "s\np\n\n")

    assert [row.review.verdict for row in result.questions] == [None, "pass"]
    assert (recorded, saves) == (1, 1)


def test_the_file_is_saved_after_every_verdict():
    result = run(*(question(f"q00{n}", answer=ANSWER) for n in (1, 2, 3)))

    recorded, _, saves = review(result, "p\n\nf\n\np\n\n")

    assert (recorded, saves) == (3, 3)


def test_q_saves_and_quits_without_touching_the_questions_after_it():
    result = run(question("q001", answer=ANSWER), question("q002", answer=ANSWER))

    recorded, output, saves = review(result, "p\n\nq\n")

    assert [row.review.verdict for row in result.questions] == ["pass", None]
    assert (recorded, saves) == (1, 1)
    assert "1 verdict" in output


def test_the_end_of_the_input_ends_the_session_like_q():
    result = run(question("q001", answer=ANSWER), question("q002", answer=ANSWER))

    recorded, _, _ = review(result, "")

    assert recorded == 0
    assert [row.review.verdict for row in result.questions] == [None, None]


def test_an_answer_the_tool_does_not_understand_is_asked_again():
    result = run(question("q001", answer=ANSWER))

    recorded, output, _ = review(result, "yes\np\n\n")

    assert recorded == 1
    assert output.count("[p]ass") == 2
    assert "p, f, s or q" in output


# --- which questions are offered --------------------------------------------------


def test_a_question_with_no_answer_is_not_offered_for_review():
    result = run(question("q001", answer=None), question("q002", answer=ANSWER))

    recorded, output, _ = review(result, "p\n\n")

    assert recorded == 1
    assert result.questions[0].review.verdict is None
    assert result.questions[1].review.verdict == "pass"
    assert "1 of the 2 questions has no answer" in output


def test_only_unreviewed_skips_the_answers_already_judged():
    result = run(
        question("q001", answer=ANSWER, verdict="pass"),
        question("q002", answer=ANSWER),
    )

    recorded, output, _ = review(result, "f\n\n", only_unreviewed=True)

    assert recorded == 1
    assert [row.review.verdict for row in result.questions] == ["pass", "fail"]
    assert "q002" in output and "q001" not in output


def test_nothing_left_to_review_says_so_and_asks_nothing():
    result = run(question("q001", answer=ANSWER, verdict="pass"))

    recorded, output, saves = review(result, "", only_unreviewed=True)

    assert (recorded, saves) == (0, 0)
    assert "nothing left to review" in output.lower()


# --- what the reviewer sees -------------------------------------------------------


def test_the_screen_shows_the_question_the_expected_answer_and_the_generated_one():
    result = run(question("q001", answer=ANSWER))

    _, output, _ = review(result, "p\n\n")

    assert "How much is the Part B premium?" in output
    assert "EXPECTED" in output and "$202.90 a month" in output
    assert "ANSWER" in output and "confidence high" in output


def test_the_screen_shows_each_cited_page_and_whether_its_quote_was_verified():
    result = run(question("q001", answer=ANSWER))

    _, output, _ = review(result, "p\n\n")

    assert "p.23" in output and "quote verified" in output
    assert "pp.27-28" in output and "QUOTE NOT VERIFIED" in output


def test_an_unanswerable_question_says_so_where_the_expected_answer_goes():
    unanswerable = question("q003", answer=ANSWER, answerable=False)

    _, output, _ = review(run(unanswerable), "p\n\n")

    assert "the handbook does not answer this" in output


# --- the command ------------------------------------------------------------------


def test_the_command_writes_the_verdicts_back_into_the_results_file(tmp_path: Path):
    result = run(question("q001", answer=ANSWER))
    path = save(result, results_path(result.run_id, tmp_path))

    exit_code = main(
        [str(path), "--reviewer", "saint"],
        input_stream=io.StringIO("p\nclean answer, right page\n"),
        output=io.StringIO(),
    )

    assert exit_code == 0
    reloaded = load(path)
    assert reloaded.questions[0].review.verdict == "pass"
    assert reloaded.questions[0].review.note == "clean answer, right page"
    assert reloaded.questions[0].review.reviewer == "saint"


def test_the_command_refuses_a_file_that_is_not_there(tmp_path: Path, capsys):
    exit_code = main(
        [str(tmp_path / "missing.json")], input_stream=io.StringIO(""), output=io.StringIO()
    )

    assert exit_code == 1
    assert "missing.json" in capsys.readouterr().err


def test_the_command_changes_nothing_in_the_file_but_the_review(tmp_path: Path):
    result = run(question("q001", answer=ANSWER))
    path = save(result, results_path(result.run_id, tmp_path))
    before = json.loads(path.read_text(encoding="utf-8"))

    main([str(path)], input_stream=io.StringIO("f\nmissed the IRMAA rule\n"), output=io.StringIO())

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["questions"][0].pop("review")["verdict"] == "fail"
    before["questions"][0].pop("review")
    assert after == before
