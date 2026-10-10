"""The two tools the agent can call, and `run_tool`, which executes a call the model made.

`lookup_cost_figure` returns one of the 2026 dollar amounts the handbook prints, with the sentence
it comes from; `check_enrollment_window` works out enrollment dates from the handbook's rules.
Both are local code: nothing here calls a model, the network or the database. A call the tools
cannot answer raises `ToolError`, whose message goes back to the model as the tool's result.
"""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from fineprint.providers.base import ToolSpec
from fineprint.tools.cost_figures import COST_FIGURES, lookup_cost_figure
from fineprint.tools.enrollment import ENROLLMENT_PAGES, SOURCE_TEXT, check_enrollment_window
from fineprint.tools.errors import ToolError

__all__ = ["TOOL_SPECS", "ToolError", "ToolOutput", "run_tool"]

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class ToolOutput:
    """What a tool call produced: the JSON the model receives, and where it came from."""

    output: dict[str, Any]
    pages: list[int]
    source_text: str | None


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="lookup_cost_figure",
        description=(
            "Look up a 2026 Medicare dollar amount exactly as the handbook prints it: the Part B "
            "premium or deductible, the Part A deductible and hospital coinsurance, skilled "
            "nursing facility coinsurance, the Part D out-of-pocket cap, or the insulin cap. "
            "Call it whenever an answer needs one of these amounts, or arithmetic on one, "
            "instead of relying on memory. The amount comes back as an exact decimal string, "
            "with the page and the handbook sentence it comes from. If the amount asked about "
            "is not in the list, the handbook does not print it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "figure": {
                    "type": "string",
                    "enum": list(COST_FIGURES),
                    "description": "Which amount to look up.",
                }
            },
            "required": ["figure"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="check_enrollment_window",
        description=(
            "Work out from a birth date when someone can sign up for Medicare and when their "
            "coverage starts: their Initial Enrollment Period around their 65th birthday, "
            "whether a given date is before, inside or after it, and the General Enrollment, "
            "Open Enrollment and Medicare Advantage Open Enrollment Periods. Call it for any "
            "question about when someone can sign up, join or switch plans, or when coverage "
            "starts. Give as_of when the question says what today's date is; otherwise leave it "
            "out. If the tool says the handbook does not cover a case, say so in the answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "birth_date": {
                    "type": "string",
                    "description": "The person's date of birth, as YYYY-MM-DD.",
                },
                "as_of": {
                    "type": "string",
                    "description": (
                        "The date to check against, as YYYY-MM-DD, when the question gives one. "
                        "Defaults to today."
                    ),
                },
            },
            "required": ["birth_date"],
            "additionalProperties": False,
        },
    ),
]


def _parse_date(arguments: Mapping[str, Any], field: str) -> date:
    """The argument `field` as a date, or a `ToolError` the model can act on."""
    value = arguments.get(field)
    if isinstance(value, str) and ISO_DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ToolError(f"{field} must be a real date written as YYYY-MM-DD; got {value!r}.")


def _lookup_cost_figure(arguments: Mapping[str, Any]) -> ToolOutput:
    key = arguments.get("figure")
    try:
        figure = lookup_cost_figure(key)
    except (KeyError, TypeError):
        known = ", ".join(COST_FIGURES)
        raise ToolError(
            f"The handbook figure table has no {key!r}; the figures it holds are: {known}."
        ) from None
    output = asdict(figure)
    output["amount"] = str(figure.amount)
    return ToolOutput(output=output, pages=[figure.page], source_text=figure.source_text)


def _check_enrollment_window(arguments: Mapping[str, Any], today: date) -> ToolOutput:
    birth_date = _parse_date(arguments, "birth_date")
    as_of = _parse_date(arguments, "as_of") if arguments.get("as_of") is not None else today
    output = check_enrollment_window(birth_date, as_of)
    return ToolOutput(output=output, pages=list(ENROLLMENT_PAGES), source_text=SOURCE_TEXT)


def run_tool(name: str, arguments: Mapping[str, Any], *, today: date) -> ToolOutput:
    """Execute the tool `name` with the model's `arguments`. Raises `ToolError` on a bad call.

    `today` is the date `check_enrollment_window` uses when the model gives no `as_of`; the
    service passes the real date and the eval runner a pinned one, so a run is reproducible.
    """
    if name == "lookup_cost_figure":
        return _lookup_cost_figure(arguments)
    if name == "check_enrollment_window":
        return _check_enrollment_window(arguments, today)
    known = ", ".join(tool.name for tool in TOOL_SPECS)
    raise ToolError(f"unknown tool {name!r}; the tools are: {known}.")
