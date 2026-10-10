"""Tests for the two tool definitions and `run_tool`, the one place a tool call is executed.

Everything a model can send is checked here: each tool with good arguments, an unknown tool, an
unknown figure, a malformed date, and `as_of` left out so `today` is used.
"""

import json
from datetime import date

import pytest

from fineprint.providers.base import ToolSpec
from fineprint.tools import TOOL_SPECS, ToolError, ToolOutput, run_tool
from fineprint.tools.cost_figures import COST_FIGURES
from fineprint.tools.enrollment import ENROLLMENT_PAGES, SOURCE_TEXT

TODAY = date(2026, 1, 10)


def spec(name: str) -> ToolSpec:
    return next(tool for tool in TOOL_SPECS if tool.name == name)


def test_there_are_exactly_the_two_tools():
    assert [tool.name for tool in TOOL_SPECS] == ["lookup_cost_figure", "check_enrollment_window"]
    assert all(isinstance(tool, ToolSpec) and tool.description for tool in TOOL_SPECS)


def test_the_figure_enum_is_exactly_the_table_keys():
    """The model cannot ask for a figure the handbook does not print."""
    schema = spec("lookup_cost_figure").input_schema

    assert schema["properties"]["figure"]["enum"] == list(COST_FIGURES)
    assert schema["required"] == ["figure"]


def test_the_enrollment_schema_requires_only_the_birth_date():
    schema = spec("check_enrollment_window").input_schema

    assert set(schema["properties"]) == {"birth_date", "as_of"}
    assert schema["required"] == ["birth_date"]


def test_a_cost_figure_comes_back_with_its_amount_as_an_exact_string():
    result = run_tool("lookup_cost_figure", {"figure": "part_b_standard_premium"}, today=TODAY)
    figure = COST_FIGURES["part_b_standard_premium"]

    assert isinstance(result, ToolOutput)
    assert result.output["amount"] == "202.90"
    assert result.output["key"] == "part_b_standard_premium"
    assert result.output["page"] == 23
    assert result.pages == [23]
    assert result.source_text == figure.source_text
    json.dumps(result.output)  # what the model receives must serialise


def test_the_enrollment_check_returns_the_window_and_its_sources():
    result = run_tool(
        "check_enrollment_window", {"birth_date": "1961-03-12", "as_of": "2026-01-10"}, today=TODAY
    )

    assert result.output["initial_enrollment_period"]["starts"] == "2025-12-01"
    assert result.output["initial_enrollment_period"]["ends"] == "2026-06-30"
    assert result.output["coverage_start_if_signed_up_on_as_of"] == "2026-03-01"
    assert result.pages == list(ENROLLMENT_PAGES)
    assert result.source_text == SOURCE_TEXT
    json.dumps(result.output)


def test_as_of_defaults_to_today():
    result = run_tool(
        "check_enrollment_window", {"birth_date": "1961-06-02"}, today=date(2026, 9, 30)
    )

    assert result.output["as_of"] == "2026-09-30"
    assert result.output["initial_enrollment_period"]["as_of_is"] == "inside"


def test_an_unknown_tool_is_a_tool_error():
    with pytest.raises(ToolError, match="unknown tool"):
        run_tool("lookup_premium", {}, today=TODAY)


def test_an_unknown_figure_is_a_tool_error_that_lists_the_known_ones():
    with pytest.raises(ToolError, match="part_b_deductible"):
        run_tool("lookup_cost_figure", {"figure": "part_d_deductible"}, today=TODAY)
    with pytest.raises(ToolError):
        run_tool("lookup_cost_figure", {}, today=TODAY)


@pytest.mark.parametrize(
    "arguments",
    [
        {"birth_date": "03/12/1961"},
        {"birth_date": "19610312"},
        {"birth_date": "1961-02-30"},
        {"birth_date": 19610312},
        {"birth_date": "1961-03-12", "as_of": "next January"},
        {},
    ],
)
def test_a_malformed_or_missing_date_is_a_tool_error(arguments: dict):
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        run_tool("check_enrollment_window", arguments, today=TODAY)


def test_errors_from_the_rules_come_through_as_tool_errors():
    with pytest.raises(ToolError, match="handbook does not say"):
        run_tool("check_enrollment_window", {"birth_date": "1960-02-29"}, today=TODAY)
    with pytest.raises(ToolError, match="before the birth date"):
        run_tool("check_enrollment_window", {"birth_date": "2027-01-01"}, today=TODAY)
