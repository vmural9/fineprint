"""Check a results file's tokens and cost against Langfuse's copy of the same traces.

    python -m evals.traces verify evals/results/<run_id>.json

A traced run stores, with every question, a `TraceRecord`: the calls the question made, the tokens
each used and what each cost. The same calls were sent to Langfuse as observations. This command
reads those observations back and compares them with the file, one question at a time. Input and
output tokens, summed over the generation and embedding calls, must match exactly. The total
cost, summed over the same observations, must match within a ten-thousandth of a dollar. (A
re-rank is stored as a generation in Langfuse, so its cost is in that total too.)

Langfuse ingests spans in the background, so a trace can take a few seconds to appear after a run
ends. A trace that is missing or does not match yet is asked for again, for up to `--wait`
seconds, before it counts as a problem.

For a run traced to Langfuse, a question with no trace id stored is a problem too.

Exit codes: 0 when every trace matched, 1 when one did not match or never appeared, or when the
three Langfuse settings are missing, or when the file holds no trace id to look for, 2 when the
file cannot be read.
"""

import argparse
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langfuse import Langfuse

from evals.results import ResultsError, RunResult, load
from fineprint.config import Settings
from fineprint.tracing import TraceRecord

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_UNREADABLE = 2

COST_TOLERANCE_USD = 0.0001
DEFAULT_WAIT_SECONDS = 30
RETRY_INTERVAL_SECONDS = 5.0
PAGE_SIZE = 1000
# The field groups the check needs: the type and trace id, and the usage, cost and total cost.
FIELDS = "core,usage"

# The observation types that carry tokens and a cost, as Langfuse spells them.
PRICED_TYPES = frozenset({"GENERATION", "EMBEDDING"})
# The call kinds whose tokens are compared. A re-rank has search units instead of tokens.
TOKEN_KINDS = frozenset({"generation", "embedding"})


@dataclass(frozen=True)
class Totals:
    """What one trace used: input tokens, output tokens, and dollars."""

    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def stored_totals(record: TraceRecord) -> Totals:
    """The totals the results file holds for one question.

    Tokens count only generation and embedding calls. The cost is the trace's, which is the sum
    over every call that has one, or None when a priced call has no price.
    """
    calls = [call for call in record.calls if call.kind in TOKEN_KINDS]
    return Totals(
        input_tokens=sum(call.input_tokens or 0 for call in calls),
        output_tokens=sum(call.output_tokens or 0 for call in calls),
        cost_usd=record.cost_usd,
    )


def langfuse_totals(observations: Sequence[Any]) -> Totals:
    """The same totals, summed over one trace's generation and embedding observations."""
    priced = [obs for obs in observations if str(obs.type).upper() in PRICED_TYPES]
    return Totals(
        input_tokens=sum((obs.usage_details or {}).get("input", 0) for obs in priced),
        output_tokens=sum((obs.usage_details or {}).get("output", 0) for obs in priced),
        cost_usd=sum(_cost_of(obs) for obs in priced),
    )


def _cost_of(observation: Any) -> float:
    """What Langfuse says one observation cost: its total, else its `total` cost detail."""
    if observation.total_cost is not None:
        return observation.total_cost
    return (observation.cost_details or {}).get("total", 0.0)


def fetch_observations(api: Any, trace_id: str) -> list[Any]:
    """Every observation of a trace, following the cursor until there is no next page."""
    observations: list[Any] = []
    cursor: str | None = None
    while True:
        page = api.observations.get_many(
            trace_id=trace_id, fields=FIELDS, limit=PAGE_SIZE, cursor=cursor
        )
        observations.extend(page.data)
        cursor = page.meta.cursor
        if not cursor:
            return observations


def differences(record: TraceRecord, observations: Sequence[Any]) -> list[str]:
    """How Langfuse's copy of one trace differs from the stored one; empty when they agree."""
    if not observations:
        return ["Langfuse has no such trace"]
    stored = stored_totals(record)
    seen = langfuse_totals(observations)
    found: list[str] = []
    if stored.input_tokens != seen.input_tokens:
        found.append(f"input tokens: stored {stored.input_tokens}, Langfuse {seen.input_tokens}")
    if stored.output_tokens != seen.output_tokens:
        found.append(f"output tokens: stored {stored.output_tokens}, Langfuse {seen.output_tokens}")
    if stored.cost_usd is not None and abs(stored.cost_usd - seen.cost_usd) > COST_TOLERANCE_USD:
        found.append(f"cost: stored ${stored.cost_usd:.6f}, Langfuse ${seen.cost_usd:.6f}")
    return found


@dataclass(frozen=True)
class Verification:
    """The outcome of checking one results file."""

    checked: int
    problems: list[str]  # one line per question that did not match
    without_cost: int  # questions whose stored cost is unknown, so only their tokens were compared


def verify(
    result: RunResult,
    api: Any,
    *,
    attempts: int = 1,
    interval: float = RETRY_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Verification:
    """Compare every traced question with Langfuse, asking again for those that do not match yet.

    A question counts as matching the first time it does. Each further round waits `interval`
    seconds and asks only about the questions still not matching, so one that arrives late costs
    nothing and one that never arrives costs `attempts - 1` waits in all.
    """
    traced = {
        row.id: row.trace
        for row in result.questions
        if row.trace is not None and row.trace.trace_id is not None
    }
    # A run traced to Langfuse has an id for every question; one without it means the sink
    # failed to open that trace, which "all match" over fewer traces would hide.
    untraced = (
        [row.id for row in result.questions if row.id not in traced]
        if result.config.tracing == "langfuse"
        else []
    )
    pending = dict(traced)
    latest: dict[str, list[str]] = {}
    for attempt in range(attempts):
        if attempt:
            sleep(interval)
        for question_id, record in list(pending.items()):
            assert record.trace_id is not None
            found = differences(record, fetch_observations(api, record.trace_id))
            latest[question_id] = found
            if not found:
                del pending[question_id]
        if not pending:
            break
    problems = [
        f"{question_id}  no trace id was stored, so Langfuse cannot be asked about it"
        for question_id in untraced
    ]
    problems += [
        f"{question_id}  trace {traced[question_id].trace_id}  {problem}"
        for question_id in pending
        for problem in latest[question_id]
    ]
    return Verification(
        checked=len(traced),
        problems=problems,
        without_cost=sum(record.cost_usd is None for record in traced.values()),
    )


def build_client(settings: Settings) -> Langfuse:
    """The Langfuse client, built from the three Langfuse settings.

    Raises `ValueError` naming the ones that are not set.
    """
    values = {
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_BASE_URL": settings.langfuse_base_url,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"verifying needs {', '.join(missing)} to be set (environment or .env)")
    assert settings.langfuse_secret_key is not None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )


def summary(outcome: Verification, run_id: str) -> str:
    """One line: how many traces were checked and how many did not match."""
    wrong = len(outcome.problems)
    line = f"{run_id}: {outcome.checked} traces checked against Langfuse, "
    line += "all match" if not wrong else f"{wrong} do not match"
    if outcome.without_cost:
        line += f" ({outcome.without_cost} with an unknown stored cost, compared on tokens only)"
    return line


def build_parser() -> argparse.ArgumentParser:
    """Describe the command."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.traces",
        description="Check a results file against the traces Langfuse holds.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "verify", help="compare each question's stored tokens and cost with Langfuse's"
    )
    check.add_argument("results", type=Path, help="the results file to check")
    check.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        metavar="SECONDS",
        help="how long to keep asking for a trace that is missing or does not match yet "
        f"(default: {DEFAULT_WAIT_SECONDS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Check one results file against Langfuse."""
    args = build_parser().parse_args(argv)
    try:
        result = load(args.results)
    except ResultsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_UNREADABLE
    if not any(row.trace is not None and row.trace.trace_id for row in result.questions):
        print(
            f"ERROR: {args.results} holds no trace id to look for; it was not made with "
            "TRACING_PROVIDER=langfuse",
            file=sys.stderr,
        )
        return EXIT_MISMATCH
    try:
        client = build_client(Settings())
    except ValueError as error:  # includes pydantic's validation error for a half-set .env
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_MISMATCH

    attempts = 1 + math.ceil(max(args.wait, 0) / RETRY_INTERVAL_SECONDS)
    try:
        outcome = verify(result, client.api, attempts=attempts)
    finally:
        client.shutdown()
    for problem in outcome.problems:
        print(problem)
    print(summary(outcome, result.run_id))
    return EXIT_MISMATCH if outcome.problems else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
