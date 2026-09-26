"""Tests for the part 1 eval metrics, with every expected number worked out by hand."""

from dataclasses import dataclass

import pytest

from evals.golden import Evidence, GoldenItem
from evals.metrics import (
    Aggregates,
    ManualPass,
    QuestionMetrics,
    abstention_correct,
    aggregate,
    cited_page_hit,
    covers,
    found_expected_pages,
    mean,
    page_hit,
    page_recall,
    reciprocal_rank,
    score_question,
)


@dataclass(frozen=True)
class Chunk:
    """A stand-in for a retrieved chunk: the pages it covers and where it ranked."""

    rank: int
    page_start: int
    page_end: int


@dataclass(frozen=True)
class Citation:
    """A stand-in for a cited chunk in an answer."""

    page_start: int
    page_end: int


@dataclass(frozen=True)
class Scored:
    """A stand-in for one question's row in a results file."""

    answerable: bool
    answered: bool
    metrics: QuestionMetrics
    verdict: str | None = None


def golden(
    item_id: str = "q001",
    *,
    item_type: str = "lookup",
    expected_pages: tuple[int, ...] = (23,),
    alt_pages: dict[int, tuple[int, ...]] | None = None,
) -> GoldenItem:
    """A golden item with only the fields the metrics read spelled out."""
    return GoldenItem(
        id=item_id,
        question="How much is the Part B premium?",
        expected_answer="$202.90 a month in 2026.",
        expected_pages=expected_pages,
        category="costs",
        difficulty="easy",
        type=item_type,
        evidence=(Evidence(page=23, quote="The standard Part B premium amount in 2026"),),
        alt_pages=alt_pages or {},
    )


UNANSWERABLE = golden("q003", item_type="unanswerable", expected_pages=())


# --- covering a page --------------------------------------------------------------


@pytest.mark.parametrize(
    ("page", "expected"),
    [(26, False), (27, True), (28, True), (29, True), (30, False)],
)
def test_a_chunk_covers_every_page_from_its_start_to_its_end(page: int, expected: bool):
    assert covers(Chunk(rank=1, page_start=27, page_end=29), page) is expected


def test_a_single_page_chunk_covers_only_that_page():
    chunk = Chunk(rank=1, page_start=23, page_end=23)

    assert covers(chunk, 23) is True
    assert covers(chunk, 24) is False


# --- which expected pages were found ----------------------------------------------


def test_an_expected_page_is_found_when_a_chunk_covers_it():
    item = golden(expected_pages=(23, 27))
    chunks = [Chunk(rank=1, page_start=22, page_end=23), Chunk(rank=2, page_start=40, page_end=41)]

    assert found_expected_pages(item, chunks) == (23,)


def test_an_alternate_page_stands_in_for_the_expected_page_it_is_listed_under():
    # q004-style: page 42 is expected, page 55 states the same fact.
    item = golden(expected_pages=(42,), alt_pages={42: (55,)})
    chunks = [Chunk(rank=1, page_start=55, page_end=56)]

    assert found_expected_pages(item, chunks) == (42,)
    assert page_hit(item, chunks) == 1.0


def test_an_alternate_of_another_expected_page_does_not_stand_in():
    # The alternate is tied to page 30, so finding it says nothing about page 27.
    item = golden(expected_pages=(27, 30), alt_pages={30: (23,)})
    chunks = [Chunk(rank=1, page_start=23, page_end=23)]

    assert found_expected_pages(item, chunks) == (30,)


# --- page_hit@5 and page_recall@5 -------------------------------------------------


def test_page_hit_is_one_when_any_expected_page_is_found_and_zero_otherwise():
    item = golden(expected_pages=(27, 29))

    assert page_hit(item, [Chunk(rank=1, page_start=29, page_end=30)]) == 1.0
    assert page_hit(item, [Chunk(rank=1, page_start=40, page_end=41)]) == 0.0
    assert page_hit(item, []) == 0.0


def test_page_recall_is_the_share_of_expected_pages_found():
    # q002-style: two expected pages, the retriever found one of them.
    item = golden("q002", item_type="multi_section", expected_pages=(27, 29))
    half = [Chunk(rank=1, page_start=26, page_end=27), Chunk(rank=2, page_start=90, page_end=91)]

    assert page_recall(item, half) == 0.5
    assert page_hit(item, half) == 1.0


def test_page_recall_is_one_when_every_expected_page_is_found():
    item = golden(expected_pages=(27, 29))
    both = [Chunk(rank=1, page_start=27, page_end=29)]

    assert page_recall(item, both) == 1.0


# --- mrr --------------------------------------------------------------------------


def test_reciprocal_rank_uses_the_rank_of_the_first_chunk_that_finds_an_expected_page():
    item = golden(expected_pages=(23,))
    chunks = [
        Chunk(rank=1, page_start=90, page_end=91),
        Chunk(rank=2, page_start=40, page_end=41),
        Chunk(rank=3, page_start=23, page_end=24),
        Chunk(rank=4, page_start=23, page_end=23),
    ]

    assert reciprocal_rank(item, chunks) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_no_chunk_finds_an_expected_page():
    assert reciprocal_rank(golden(), [Chunk(rank=1, page_start=90, page_end=91)]) == 0.0


def test_reciprocal_rank_reads_the_rank_rather_than_the_list_order():
    item = golden(expected_pages=(23,))
    out_of_order = [
        Chunk(rank=5, page_start=23, page_end=23),
        Chunk(rank=1, page_start=9, page_end=9),
    ]

    assert reciprocal_rank(item, out_of_order) == 0.2


# --- retrieval metrics skip unanswerable questions --------------------------------


def test_retrieval_metrics_are_not_applicable_to_an_unanswerable_question():
    chunks = [Chunk(rank=1, page_start=92, page_end=92)]

    assert page_hit(UNANSWERABLE, chunks) is None
    assert page_recall(UNANSWERABLE, chunks) is None
    assert reciprocal_rank(UNANSWERABLE, chunks) is None
    assert cited_page_hit(UNANSWERABLE, [Citation(page_start=92, page_end=92)]) is None


# --- cited_page_hit and abstention_accuracy ---------------------------------------


def test_cited_page_hit_asks_whether_the_citations_find_an_expected_page():
    item = golden(expected_pages=(23,))

    assert cited_page_hit(item, [Citation(page_start=22, page_end=23)]) == 1.0
    assert cited_page_hit(item, [Citation(page_start=40, page_end=41)]) == 0.0
    assert cited_page_hit(item, []) == 0.0


def test_cited_page_hit_is_not_applicable_when_no_answer_was_generated():
    assert cited_page_hit(golden(), None) is None


def test_abstention_is_correct_when_found_in_handbook_matches_the_question():
    assert abstention_correct(golden(), found_in_handbook=True) == 1.0
    assert abstention_correct(golden(), found_in_handbook=False) == 0.0
    assert abstention_correct(UNANSWERABLE, found_in_handbook=False) == 1.0
    assert abstention_correct(UNANSWERABLE, found_in_handbook=True) == 0.0
    assert abstention_correct(golden(), found_in_handbook=None) is None


# --- scoring one question ---------------------------------------------------------


def test_score_question_fills_every_metric_for_an_answered_answerable_question():
    item = golden(expected_pages=(27, 29))
    chunks = [Chunk(rank=1, page_start=26, page_end=27), Chunk(rank=2, page_start=90, page_end=91)]

    scored = score_question(
        item, chunks, citations=[Citation(page_start=27, page_end=27)], found_in_handbook=True
    )

    assert scored == QuestionMetrics(
        page_hit=1.0,
        page_recall=0.5,
        reciprocal_rank=1.0,
        cited_page_hit=1.0,
        abstention_correct=1.0,
    )


def test_score_question_leaves_the_answer_metrics_empty_on_a_retrieval_only_run():
    scored = score_question(golden(), [Chunk(rank=1, page_start=23, page_end=23)])

    assert scored == QuestionMetrics(
        page_hit=1.0,
        page_recall=1.0,
        reciprocal_rank=1.0,
        cited_page_hit=None,
        abstention_correct=None,
    )


# --- aggregates -------------------------------------------------------------------


def test_mean_ignores_metrics_that_do_not_apply_and_is_empty_when_none_apply():
    assert mean([1.0, 0.0, None, 0.5]) == 0.5
    assert mean([None, None]) is None
    assert mean([]) is None


def test_manual_pass_is_passes_over_reviewed_and_reports_the_reviewed_count():
    reviewed = ManualPass(passes=3, reviewed=4)

    assert reviewed.rate == 0.75
    assert ManualPass(passes=0, reviewed=0).rate is None


def test_aggregate_averages_over_the_questions_each_metric_applies_to():
    # Three answerable questions and one unanswerable one, all answered.
    # page_hit: (1 + 1 + 0) / 3, page_recall: (1 + 0.5 + 0) / 3, mrr: (1 + 0.5 + 0) / 3,
    # cited_page_hit: (1 + 0 + 0) / 3, abstention: (1 + 1 + 1 + 0) / 4, manual: 1 pass of 2.
    rows = [
        Scored(
            answerable=True,
            answered=True,
            metrics=QuestionMetrics(1.0, 1.0, 1.0, 1.0, 1.0),
            verdict="pass",
        ),
        Scored(
            answerable=True,
            answered=True,
            metrics=QuestionMetrics(1.0, 0.5, 0.5, 0.0, 1.0),
            verdict="fail",
        ),
        Scored(
            answerable=True,
            answered=True,
            metrics=QuestionMetrics(0.0, 0.0, 0.0, 0.0, 1.0),
            verdict=None,
        ),
        Scored(
            answerable=False,
            answered=True,
            metrics=QuestionMetrics(None, None, None, None, 0.0),
            verdict=None,
        ),
    ]

    assert aggregate(rows) == Aggregates(
        questions=4,
        answerable=3,
        answered=4,
        page_hit=pytest.approx(2 / 3),
        page_recall=0.5,
        mrr=0.5,
        cited_page_hit=pytest.approx(1 / 3),
        abstention_accuracy=0.75,
        manual=ManualPass(passes=1, reviewed=2),
        context_recall=None,
        context_precision=None,
        faithfulness=None,
        answer_relevance=None,
    )


def test_aggregate_of_nothing_measures_nothing():
    assert aggregate([]) == Aggregates(
        questions=0,
        answerable=0,
        answered=0,
        page_hit=None,
        page_recall=None,
        mrr=None,
        cited_page_hit=None,
        abstention_accuracy=None,
        manual=ManualPass(passes=0, reviewed=0),
        context_recall=None,
        context_precision=None,
        faithfulness=None,
        answer_relevance=None,
    )


# --- the four RAGAS-style metrics: nothing computes them yet, but aggregate() folds them in ---


def test_aggregate_means_the_four_ragas_metrics_when_they_are_present():
    """Nothing in this module computes these yet — a later scoring pass will. This only proves
    `aggregate` folds them in with the same None-ignoring mean as every other metric here, once
    they are filled: one answerable question scores on all four, one unanswerable question (D7)
    leaves them `None` and is skipped rather than counted as zero."""
    rows = [
        Scored(
            answerable=True,
            answered=True,
            metrics=QuestionMetrics(
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                context_recall=1.0,
                context_precision=0.8,
                faithfulness=1.0,
                answer_relevance=0.9,
            ),
        ),
        Scored(
            answerable=False,
            answered=True,
            metrics=QuestionMetrics(None, None, None, None, 0.0),
        ),
    ]

    totals = aggregate(rows)

    assert totals.context_recall == 1.0
    assert totals.context_precision == 0.8
    assert totals.faithfulness == 1.0
    assert totals.answer_relevance == 0.9


def test_aggregate_leaves_the_four_ragas_metrics_none_when_none_are_present():
    metrics = QuestionMetrics(1.0, 1.0, 1.0, 1.0, 1.0)
    rows = [Scored(answerable=True, answered=True, metrics=metrics)]

    totals = aggregate(rows)

    assert (
        totals.context_recall,
        totals.context_precision,
        totals.faithfulness,
        totals.answer_relevance,
    ) == (None, None, None, None)
