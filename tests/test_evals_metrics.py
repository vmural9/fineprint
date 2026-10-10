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
    as_money,
    as_seconds,
    cited_page_hit,
    covers,
    found_expected_pages,
    mean,
    page_hit,
    page_recall,
    percentile,
    reciprocal_rank,
    score_question,
    step_times,
)
from fineprint.tracing import CallRecord, TraceRecord


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
    trace: TraceRecord | None = None


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


# --- part 3: percentiles, cost and latency ----------------------------------------


def call(kind: str, *, fallback: bool = False) -> CallRecord:
    """One finished call of the given kind; the aggregates only read its kind and fallback flag."""
    return CallRecord(
        kind=kind,
        name=kind,
        model=None,
        input_tokens=None,
        output_tokens=None,
        units=None,
        latency_ms=1.0,
        cost_usd=None,
        error=None,
        fallback=fallback,
    )


def trace(latency_ms: float, cost_usd: float | None, *calls: CallRecord) -> TraceRecord:
    return TraceRecord(
        trace_id=None, name="ask", latency_ms=latency_ms, cost_usd=cost_usd, calls=list(calls)
    )


def traced(*traces: TraceRecord | None) -> list[Scored]:
    """One answered question per trace."""
    return [
        Scored(answerable=True, answered=True, metrics=QuestionMetrics(), trace=item)
        for item in traces
    ]


def test_the_95th_percentile_of_forty_values_is_the_38th_smallest():
    # Rank = ceil(95 / 100 * 40) = 38, so of 100, 200, ..., 4000 (shuffled by reversing) it is 3800.
    values = [float(step * 100) for step in range(40, 0, -1)]

    assert percentile(values, 95) == 3800.0


def test_the_median_of_forty_values_is_the_20th_smallest_not_the_mean_of_two():
    values = [float(step * 100) for step in range(1, 41)]

    assert percentile(values, 50) == 2000.0


def test_a_percentile_is_always_one_of_the_values_and_the_ends_are_the_smallest_and_largest():
    values = [30.0, 10.0, 20.0]

    assert percentile(values, 100) == 30.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 95) == 30.0  # rank ceil(2.85) = 3


def test_the_percentile_of_a_single_value_is_that_value():
    assert percentile([7.5], 95) == 7.5
    assert percentile([7.5], 50) == 7.5


def test_the_two_slowest_of_forty_calls_do_not_move_the_95th_percentile_but_three_do():
    # Ranks 39 and 40 sit above the 38th value; a third slow call pushes into rank 38.
    two_slow = [1000.0] * 38 + [90000.0, 90000.0]
    three_slow = [1000.0] * 37 + [90000.0] * 3

    assert percentile(two_slow, 95) == 1000.0
    assert percentile(three_slow, 95) == 90000.0


def test_a_percentile_of_nothing_is_refused_and_so_is_a_p_outside_zero_to_a_hundred():
    with pytest.raises(ValueError):
        percentile([], 95)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_the_cost_and_latency_aggregates_are_worked_out_from_the_traces():
    # Costs 0.01, 0.02, 0.03, 0.04 average 0.025. Latencies 1000, 2000, 3000, 10000 ms: p50 is
    # rank 2 (2000), p95 is rank ceil(3.8) = 4 (10000), max 10000. Generations 1, 2, 1, 2 average
    # 1.5 (embedding and rerank calls are not generations). Tool calls 0, 1, 0, 3 average 1.0.
    # Fallbacks: the second question has one fallback generation, the fourth has two; two
    # questions, not three calls.
    rows = traced(
        trace(1000.0, 0.01, call("embedding"), call("generation")),
        trace(2000.0, 0.02, call("generation", fallback=True), call("generation"), call("tool")),
        trace(3000.0, 0.03, call("rerank"), call("generation")),
        trace(
            10000.0,
            0.04,
            call("generation", fallback=True),
            call("generation", fallback=True),
            call("tool"),
            call("tool"),
            call("tool"),
        ),
    )

    totals = aggregate(rows)

    assert totals.cost_per_query_usd == pytest.approx(0.025)
    assert totals.p50_latency_ms == 2000.0
    assert totals.p95_latency_ms == 10000.0
    assert totals.max_latency_ms == 10000.0
    assert totals.generations_per_query == 1.5
    assert totals.tool_calls_per_query == 1.0
    assert totals.fallbacks == 2


def test_unanswerable_questions_count_towards_the_cost_per_query():
    rows = traced(trace(1000.0, 0.03), trace(1000.0, 0.01))
    rows[1] = Scored(
        answerable=False,
        answered=True,
        metrics=QuestionMetrics(),
        trace=trace(1000.0, 0.01),
    )

    assert aggregate(rows).cost_per_query_usd == pytest.approx(0.02)


def test_one_question_without_a_trace_makes_every_cost_and_latency_aggregate_none():
    totals = aggregate(traced(trace(1000.0, 0.01), None, trace(3000.0, 0.03)))

    assert totals.cost_per_query_usd is None
    assert totals.p50_latency_ms is None
    assert totals.p95_latency_ms is None
    assert totals.max_latency_ms is None
    assert totals.generations_per_query is None
    assert totals.tool_calls_per_query is None
    assert totals.fallbacks is None


def test_one_trace_without_a_cost_makes_every_cost_and_latency_aggregate_none():
    totals = aggregate(traced(trace(1000.0, 0.01), trace(2000.0, None), trace(3000.0, 0.03)))

    assert totals.cost_per_query_usd is None
    assert totals.p95_latency_ms is None
    assert totals.fallbacks is None


def test_a_trace_costing_nothing_is_a_cost_not_a_missing_one():
    totals = aggregate(traced(trace(1000.0, 0.0)))

    assert totals.cost_per_query_usd == 0.0
    assert totals.fallbacks == 0


def test_a_fallback_flag_on_a_call_that_is_not_a_generation_is_not_a_fallback():
    # The column says a generation ran on the fallback model; a re-rank that degraded carries the
    # flag too, and must not be counted.
    rows = traced(
        trace(1000.0, 0.01, call("rerank", fallback=True), call("generation")),
        trace(1000.0, 0.01, call("generation", fallback=True), call("rerank", fallback=True)),
    )

    assert aggregate(rows).fallbacks == 1


def test_money_is_four_decimals_and_seconds_one_and_a_missing_value_is_a_dash():
    assert as_money(0.0464) == "$0.0464"
    assert as_money(0.0) == "$0.0000"
    assert as_money(0.12345) == "$0.1235"
    assert as_money(None) == "—"
    assert as_seconds(12400.0) == "12.4 s"
    assert as_seconds(None) == "—"


def test_display_rounding_is_half_up_on_the_decimal_value_not_on_the_float():
    # 0.95 and 0.00005 are a hair under in binary; a person reading them expects the round up.
    assert as_seconds(950.0) == "1.0 s"
    assert as_seconds(1049.9) == "1.0 s"
    assert as_seconds(1050.0) == "1.1 s"
    assert as_seconds(949.9) == "0.9 s"
    assert as_money(0.00005) == "$0.0001"
    assert as_money(0.00004999) == "$0.0000"
    assert as_money(0.00015) == "$0.0002"


# --- where the time went ------------------------------------------------------------


def timed_call(kind: str, latency_ms: float) -> CallRecord:
    return CallRecord(
        kind=kind,
        name=kind,
        model=None,
        input_tokens=None,
        output_tokens=None,
        units=None,
        latency_ms=latency_ms,
        cost_usd=None,
        error=None,
    )


def test_step_times_sum_each_kind_per_question_then_summarise_over_the_questions():
    # Question 1: 1 s search, generations 2 s + 3 s, a 0.5 s tool, of 10 s in all.
    # Question 2: 3 s search, one 6 s generation, no tool, of 10 s in all.
    # Question 3: 2 s search, one 1 s generation, of 4 s in all.
    traces = [
        trace(
            10_000,
            None,
            timed_call("retriever", 1000),
            timed_call("generation", 2000),
            timed_call("generation", 3000),
            timed_call("tool", 500),
        ),
        trace(10_000, None, timed_call("retriever", 3000), timed_call("generation", 6000)),
        trace(4_000, None, timed_call("retriever", 2000), timed_call("generation", 1000)),
    ]

    steps = step_times(traces)

    assert (steps.retriever_p50_ms, steps.retriever_p95_ms) == (2000.0, 3000.0)
    assert (steps.generation_p50_ms, steps.generation_p95_ms) == (5000.0, 6000.0)
    # Tools per question are 500, 0, 0: the question with no tool counts as 0 s.
    assert (steps.tools_p50_ms, steps.tools_p95_ms) == (0.0, 500.0)
    # Shares 0.5, 0.6, 0.25: the median is the second smallest, 0.5.
    assert steps.generation_share_p50 == 0.5


def test_a_question_with_no_time_is_left_out_of_the_share():
    steps = step_times([trace(0, None), trace(2_000, None, timed_call("generation", 1000))])

    assert steps.generation_share_p50 == 0.5
    assert step_times([trace(0, None)]).generation_share_p50 is None


def test_step_times_of_no_questions_is_an_error():
    with pytest.raises(ValueError):
        step_times([])
