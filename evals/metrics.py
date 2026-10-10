"""The part 1 eval metrics: what "good retrieval" means, written out as pure functions.

Every function here takes plain data and returns a number. Nothing in this module touches the
database, the network, or a results file, so the definitions can be read and tested on their own.

Two conventions run through the module:

- **A metric that does not apply is `None`, never zero.** Retrieval metrics do not apply to an
  unanswerable question (it has no expected pages), and the answer metrics do not apply to a
  question no answer was generated for. `None` keeps those questions out of the average instead
  of dragging it down, and it is what the scoreboard renders as a dash.
- **A metric that does apply is 1.0 or 0.0 when it is a yes-or-no**, so that averaging a list of
  per-question values gives the published rate with no special cases.

The spec names the retrieval metrics `page_hit@5` and `page_recall@5`. The `5` is the run's
configured top-k, which the results file records next to the numbers; part 1 runs every
configuration at k = 5. The functions here score whatever list of chunks they are handed, so the
caller is the one that cuts the list to k.

Part 2 adds four more fields to `QuestionMetrics` and `Aggregates`: `context_recall`,
`context_precision`, `faithfulness` and `answer_relevance`, the four RAGAS-style metrics judged
by a language model rather than computed from page numbers. This module only carries their
fields and folds them into a run's aggregates with the same `None`-ignoring `mean` as everything
else here; `evals/ragas_metrics.py` defines what each one measures, and `python -m evals.score`
computes them in a second pass over a results file, from the retrieved text the file records.

Part 3 adds cost and latency. They are read from the `TraceRecord` stored with each question and
are all-or-nothing per run: if any question has no trace, or a trace with no cost, the whole set
is `None`, because a mean over some of the questions would read as the cost of all of them.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from evals.golden import GoldenItem
from fineprint.tracing import TraceRecord

PASS = "pass"
FAIL = "fail"
VERDICTS = frozenset({PASS, FAIL})

# What a missing measurement looks like everywhere it is printed. A dash is the absence of a
# measurement, never a stand-in value, so it is defined once and imported.
NOT_MEASURED = "—"


class PageRange(Protocol):
    """Anything that sits on a run of handbook pages: a retrieved chunk, or a citation."""

    page_start: int
    page_end: int


class RankedPageRange(PageRange, Protocol):
    """A page range that also knows its 1-based rank in the retrieved list."""

    rank: int


class QuestionScore(Protocol):
    """One question's row in a results file, as far as aggregation is concerned."""

    answerable: bool
    answered: bool
    metrics: "QuestionMetrics"
    verdict: str | None
    trace: TraceRecord | None


@dataclass(frozen=True, slots=True)
class QuestionMetrics:
    """What one question scored. `None` means the metric does not apply to this question."""

    page_hit: float | None = None
    page_recall: float | None = None
    reciprocal_rank: float | None = None
    cited_page_hit: float | None = None
    abstention_correct: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None


@dataclass(frozen=True, slots=True)
class ManualPass:
    """The human verdicts: how many answers passed, out of how many a person looked at."""

    passes: int
    reviewed: int

    @property
    def rate(self) -> float | None:
        """Passes divided by reviewed answers, or None when nobody has reviewed anything."""
        return self.passes / self.reviewed if self.reviewed else None


@dataclass(frozen=True, slots=True)
class Aggregates:
    """The run-level numbers, every one of them averaged from the per-question metrics."""

    questions: int
    answerable: int
    answered: int
    page_hit: float | None
    page_recall: float | None
    mrr: float | None
    cited_page_hit: float | None
    abstention_accuracy: float | None
    manual: ManualPass
    context_recall: float | None
    context_precision: float | None
    faithfulness: float | None
    answer_relevance: float | None
    # Part 3: read from the trace stored with each question. All None unless every question of
    # the run has a trace with a cost (see `_serving`).
    cost_per_query_usd: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    max_latency_ms: float | None = None
    generations_per_query: float | None = None
    tool_calls_per_query: float | None = None
    fallbacks: int | None = None


@dataclass(frozen=True, slots=True)
class Serving:
    """What serving the run's questions cost and how long it took, from their traces."""

    cost_per_query_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    generations_per_query: float
    tool_calls_per_query: float
    fallbacks: int


def covers(page_range: PageRange, page: int) -> bool:
    """True when the page lies within this chunk's first and last page, inclusive."""
    return page_range.page_start <= page <= page_range.page_end


def found_expected_pages(item: GoldenItem, chunks: Sequence[PageRange]) -> tuple[int, ...]:
    """The expected pages the chunks found, in ascending order.

    An expected page counts as found when a chunk covers it or covers one of the pages listed
    under it in `alt_pages` — a page that states the same fact, so retrieving it is just as good.
    """
    return tuple(
        page
        for page in sorted(item.expected_pages)
        if any(
            covers(chunk, satisfying)
            for satisfying in item.pages_that_satisfy(page)
            for chunk in chunks
        )
    )


def page_hit(item: GoldenItem, chunks: Sequence[RankedPageRange]) -> float | None:
    """1.0 when the retrieved chunks found at least one expected page. The headline metric."""
    if not item.answerable:
        return None
    return float(bool(found_expected_pages(item, chunks)))


def page_recall(item: GoldenItem, chunks: Sequence[RankedPageRange]) -> float | None:
    """The share of the question's expected pages the retrieved chunks found.

    This is the metric that exposes a multi-section question: finding one of the two pages an
    answer needs scores 1.0 on `page_hit` but only 0.5 here.
    """
    if not item.answerable:
        return None
    return len(found_expected_pages(item, chunks)) / len(item.expected_pages)


def reciprocal_rank(item: GoldenItem, chunks: Sequence[RankedPageRange]) -> float | None:
    """1 / the rank of the first chunk that finds an expected page, or 0.0 when none does."""
    if not item.answerable:
        return None
    ranks = [
        chunk.rank
        for chunk in chunks
        if any(
            covers(chunk, satisfying)
            for page in item.expected_pages
            for satisfying in item.pages_that_satisfy(page)
        )
    ]
    return 1 / min(ranks) if ranks else 0.0


def cited_page_hit(item: GoldenItem, citations: Sequence[PageRange] | None) -> float | None:
    """1.0 when the answer's own citations land on an expected page.

    `page_hit` asks whether retrieval put the right page in front of the model; this asks whether
    the model then used it. It does not apply to an unanswerable question, which has no expected
    pages, nor to a question that was never answered.
    """
    if not item.answerable or citations is None:
        return None
    return float(bool(found_expected_pages(item, citations)))


def abstention_correct(item: GoldenItem, found_in_handbook: bool | None) -> float | None:
    """1.0 when the answer's `found_in_handbook` matches whether the handbook answers at all.

    This is the one answer metric that applies to every question, answerable or not: saying "the
    handbook does not give this" about a question it does answer is just as wrong as the reverse.
    """
    if found_in_handbook is None:
        return None
    return float(found_in_handbook is item.answerable)


def score_question(
    item: GoldenItem,
    chunks: Sequence[RankedPageRange],
    *,
    citations: Sequence[PageRange] | None = None,
    found_in_handbook: bool | None = None,
) -> QuestionMetrics:
    """Score one question. Leave `citations` and `found_in_handbook` out on a retrieval-only run.

    This fills only the five part 1 metrics. The four RAGAS metrics are left at their default
    `None` here; `python -m evals.score` fills them in afterwards, with a judge model to ask.
    """
    return QuestionMetrics(
        page_hit=page_hit(item, chunks),
        page_recall=page_recall(item, chunks),
        reciprocal_rank=reciprocal_rank(item, chunks),
        cited_page_hit=cited_page_hit(item, citations),
        abstention_correct=abstention_correct(item, found_in_handbook),
    )


def mean(values: Iterable[float | None]) -> float | None:
    """The mean of the values the metric applied to, or None when it applied to nothing."""
    applicable = [value for value in values if value is not None]
    return sum(applicable) / len(applicable) if applicable else None


def manual_pass(verdicts: Iterable[str | None]) -> ManualPass:
    """Count the human verdicts. An unreviewed answer is neither a pass nor a fail."""
    reviewed = [verdict for verdict in verdicts if verdict is not None]
    return ManualPass(passes=sum(verdict == PASS for verdict in reviewed), reviewed=len(reviewed))


def percentile(values: Sequence[float], p: float) -> float:
    """The nearest-rank percentile: sort ascending and take the value at 1-based rank
    ceil(p / 100 * n).

    Of 40 values the 95th percentile is therefore the 38th smallest, one actual measurement and
    never an interpolation between two. Raises `ValueError` when there are no values.
    """
    if not values:
        raise ValueError("a percentile needs at least one value")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be between 0 and 100, got {p}")
    rank = max(1, math.ceil(p * len(values) / 100))  # multiply before dividing: 95 * 40 is exact
    return sorted(values)[rank - 1]


def as_percentage(value: float | None) -> str:
    """A share printed as a percentage, or the dash that means nobody has measured it."""
    return NOT_MEASURED if value is None else f"{value * 100:.1f}%"


def as_score(value: float | None) -> str:
    """A plain number, such as mrr, printed to two places, or the dash for no measurement."""
    return NOT_MEASURED if value is None else f"{value:.2f}"


def as_money(value: float | None) -> str:
    """Dollars to four places, such as `$0.0464`, or the dash for no measurement."""
    return NOT_MEASURED if value is None else f"${value:.4f}"


def as_seconds(value_ms: float | None) -> str:
    """Milliseconds printed as seconds to one place, such as `12.4 s`, or the dash."""
    return NOT_MEASURED if value_ms is None else f"{value_ms / 1000:.1f} s"


def _serving(rows: Sequence[QuestionScore]) -> Serving | None:
    """Cost and latency over every question, or None unless each has a trace with a cost.

    Unanswerable questions count: the service pays to say "the handbook does not say" too.
    """
    traces = [row.trace for row in rows]
    complete = [trace for trace in traces if trace is not None and trace.cost_usd is not None]
    if not traces or len(complete) != len(traces):
        return None
    latencies = [trace.latency_ms for trace in complete]
    count = len(complete)
    return Serving(
        cost_per_query_usd=sum(trace.cost_usd for trace in complete) / count,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        max_latency_ms=max(latencies),
        generations_per_query=_calls(complete, "generation") / count,
        tool_calls_per_query=_calls(complete, "tool") / count,
        fallbacks=sum(any(call.fallback for call in trace.calls) for trace in complete),
    )


def _calls(traces: Sequence[TraceRecord], kind: str) -> int:
    """How many calls of this kind the traces hold in all."""
    return sum(call.kind == kind for trace in traces for call in trace.calls)


def aggregate(scores: Iterable[QuestionScore]) -> Aggregates:
    """Roll the per-question metrics up into the run's numbers.

    The scoreboard calls this on the rows of a results file rather than reading a stored summary,
    so a published number can always be traced back to the questions behind it.
    """
    rows = list(scores)
    serving = _serving(rows)
    return Aggregates(
        questions=len(rows),
        answerable=sum(row.answerable for row in rows),
        answered=sum(row.answered for row in rows),
        page_hit=mean(row.metrics.page_hit for row in rows),
        page_recall=mean(row.metrics.page_recall for row in rows),
        mrr=mean(row.metrics.reciprocal_rank for row in rows),
        cited_page_hit=mean(row.metrics.cited_page_hit for row in rows),
        abstention_accuracy=mean(row.metrics.abstention_correct for row in rows),
        manual=manual_pass(row.verdict for row in rows),
        context_recall=mean(row.metrics.context_recall for row in rows),
        context_precision=mean(row.metrics.context_precision for row in rows),
        faithfulness=mean(row.metrics.faithfulness for row in rows),
        answer_relevance=mean(row.metrics.answer_relevance for row in rows),
        cost_per_query_usd=serving.cost_per_query_usd if serving else None,
        p50_latency_ms=serving.p50_latency_ms if serving else None,
        p95_latency_ms=serving.p95_latency_ms if serving else None,
        max_latency_ms=serving.max_latency_ms if serving else None,
        generations_per_query=serving.generations_per_query if serving else None,
        tool_calls_per_query=serving.tool_calls_per_query if serving else None,
        fallbacks=serving.fallbacks if serving else None,
    )
