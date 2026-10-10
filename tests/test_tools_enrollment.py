"""Tests for the enrollment-window rules of pages 17, 18, 71, 72 and 80.

The cases come from the handbook: page 17's two worked examples (turning 65 on June 2 gives March
to September, on June 1 gives February to August) and golden question q006 (turning 65 on March 12
gives December to June). Every window is checked on its first and last day and the days either
side. The `corpus` test checks the rule sentences the module quotes against the page text.
"""

from datetime import date

import pytest

from fineprint.tools import ToolError
from fineprint.tools.enrollment import (
    ENROLLMENT_PAGES,
    RULE_QUOTES,
    check_enrollment_window,
    general_enrollment_period,
    initial_enrollment_period,
    medicare_advantage_open_enrollment,
    open_enrollment_period,
    part_b_coverage_start,
    turns_65_on,
)
from tests.test_tools_cost_figures import collapse, handbook_pages

# Born so that they turn 65 in 2026 on the dates the handbook and q006 use.
JUNE_2 = date(1961, 6, 2)
JUNE_1 = date(1961, 6, 1)
MARCH_12 = date(1961, 3, 12)


def test_turning_65_is_the_65th_birthday():
    assert turns_65_on(JUNE_2) == date(2026, 6, 2)
    assert turns_65_on(date(1960, 12, 31)) == date(2025, 12, 31)


def test_a_leap_day_birth_date_is_refused_because_the_handbook_does_not_cover_it():
    with pytest.raises(ToolError, match="handbook does not say"):
        turns_65_on(date(1960, 2, 29))
    with pytest.raises(ToolError, match="handbook does not say"):
        check_enrollment_window(date(1960, 2, 29), date(2025, 1, 15))


# The Initial Enrollment Period


def test_page_17_example_turning_65_on_june_2_runs_march_to_september():
    window = initial_enrollment_period(JUNE_2)

    assert (window.starts, window.ends) == (date(2026, 3, 1), date(2026, 9, 30))
    assert window.page == 17


def test_page_17_example_turning_65_on_june_1_runs_february_to_august():
    window = initial_enrollment_period(JUNE_1)

    assert (window.starts, window.ends) == (date(2026, 2, 1), date(2026, 8, 31))


def test_q006_turning_65_on_march_12_runs_december_to_june():
    window = initial_enrollment_period(MARCH_12)

    assert (window.starts, window.ends) == (date(2025, 12, 1), date(2026, 6, 30))


def test_the_window_crosses_the_year_end_and_ends_on_the_last_day_of_a_leap_february():
    window = initial_enrollment_period(date(1962, 11, 15))  # turns 65 on 2027-11-15

    assert (window.starts, window.ends) == (date(2027, 8, 1), date(2028, 2, 29))


def test_a_january_1_birthday_shifts_the_window_back_into_the_previous_year():
    window = initial_enrollment_period(date(1962, 1, 1))  # turns 65 on 2027-01-01

    assert (window.starts, window.ends) == (date(2026, 9, 1), date(2027, 3, 31))


@pytest.mark.parametrize(
    ("as_of", "position"),
    [
        (date(2026, 2, 28), "before"),  # the day before the window
        (date(2026, 3, 1), "inside"),  # the first day
        (date(2026, 9, 30), "inside"),  # the last day
        (date(2026, 10, 1), "after"),  # the day after
    ],
)
def test_as_of_is_placed_before_inside_or_after_the_june_2_window(as_of: date, position: str):
    window = check_enrollment_window(JUNE_2, as_of)["initial_enrollment_period"]

    assert window["as_of_is"] == position


@pytest.mark.parametrize(
    ("as_of", "position"),
    [
        (date(2026, 1, 31), "before"),
        (date(2026, 2, 1), "inside"),
        (date(2026, 8, 31), "inside"),
        (date(2026, 9, 1), "after"),
    ],
)
def test_as_of_is_placed_before_inside_or_after_the_june_1_window(as_of: date, position: str):
    window = check_enrollment_window(JUNE_1, as_of)["initial_enrollment_period"]

    assert window["as_of_is"] == position


@pytest.mark.parametrize(
    ("as_of", "position"),
    [
        (date(2025, 11, 30), "before"),
        (date(2025, 12, 1), "inside"),
        (date(2026, 6, 30), "inside"),
        (date(2026, 7, 1), "after"),
    ],
)
def test_as_of_is_placed_before_inside_or_after_the_q006_window(as_of: date, position: str):
    window = check_enrollment_window(MARCH_12, as_of)["initial_enrollment_period"]

    assert window["as_of_is"] == position


# Coverage start in the Initial Enrollment Period


@pytest.mark.parametrize(
    ("signup", "starts"),
    [
        (date(2026, 2, 28), None),  # before the window
        (date(2026, 3, 1), date(2026, 6, 1)),  # first 3 months: the birthday month
        (date(2026, 5, 31), date(2026, 6, 1)),
        (date(2026, 6, 1), date(2026, 7, 1)),  # the month they turn 65: the month after
        (date(2026, 6, 30), date(2026, 7, 1)),
        (date(2026, 7, 1), date(2026, 8, 1)),  # last 3 months: the month after
        (date(2026, 9, 30), date(2026, 10, 1)),
        (date(2026, 10, 1), None),  # after the window
    ],
)
def test_coverage_start_for_turning_65_on_june_2(signup: date, starts: date | None):
    assert part_b_coverage_start(JUNE_2, signup) == starts


@pytest.mark.parametrize(
    ("signup", "starts"),
    [
        (date(2026, 2, 1), date(2026, 5, 1)),  # first 3 months: the month before the birthday
        (date(2026, 4, 30), date(2026, 5, 1)),
        (date(2026, 5, 1), None),  # the fourth month: page 17 does not say
        (date(2026, 5, 31), None),
        (date(2026, 6, 1), date(2026, 7, 1)),  # the month they turn 65: the month after
        (date(2026, 8, 31), date(2026, 9, 1)),  # the last day of the window
    ],
)
def test_coverage_start_for_turning_65_on_june_1(signup: date, starts: date | None):
    assert part_b_coverage_start(JUNE_1, signup) == starts


@pytest.mark.parametrize(
    ("signup", "starts"),
    [
        (date(2025, 12, 1), date(2026, 3, 1)),
        (date(2026, 2, 28), date(2026, 3, 1)),
        (date(2026, 3, 12), date(2026, 4, 1)),
        (date(2026, 6, 30), date(2026, 7, 1)),
    ],
)
def test_coverage_start_for_q006(signup: date, starts: date):
    """q006: sign up December to February and coverage begins March 1; later, the month after."""
    assert part_b_coverage_start(MARCH_12, signup) == starts


def test_a_december_signup_late_in_the_window_starts_coverage_in_january():
    # Turns 65 on 2026-10-20; the window runs July 2026 to January 2027.
    assert part_b_coverage_start(date(1961, 10, 20), date(2026, 12, 5)) == date(2027, 1, 1)


def test_the_check_reports_the_coverage_start_for_a_signup_on_as_of():
    result = check_enrollment_window(MARCH_12, date(2026, 1, 10))

    assert result["coverage_start_if_signed_up_on_as_of"] == "2026-03-01"
    assert "first 3 months" in result["coverage_start_rule"]


def test_the_check_says_when_page_17_gives_no_coverage_start():
    result = check_enrollment_window(JUNE_1, date(2026, 5, 15))

    assert result["initial_enrollment_period"]["as_of_is"] == "inside"
    assert result["coverage_start_if_signed_up_on_as_of"] is None
    assert "does not say" in result["coverage_start_rule"]


def test_outside_the_window_there_is_no_coverage_start():
    result = check_enrollment_window(JUNE_2, date(2026, 10, 1))

    assert result["coverage_start_if_signed_up_on_as_of"] is None


# The yearly periods


def test_the_general_enrollment_period_is_january_1_to_march_31():
    window = general_enrollment_period(2027)

    assert (window.starts, window.ends, window.page) == (date(2027, 1, 1), date(2027, 3, 31), 18)


def test_open_enrollment_is_october_15_to_december_7():
    window = open_enrollment_period(2026)

    assert (window.starts, window.ends, window.page) == (date(2026, 10, 15), date(2026, 12, 7), 71)


def test_the_medicare_advantage_open_enrollment_period_is_january_1_to_march_31():
    window = medicare_advantage_open_enrollment(2027)

    assert (window.starts, window.ends, window.page) == (date(2027, 1, 1), date(2027, 3, 31), 72)


@pytest.mark.parametrize(
    ("as_of", "starts", "inside"),
    [
        (date(2026, 12, 31), "2027-01-01", False),
        (date(2027, 1, 1), "2027-01-01", True),
        (date(2027, 3, 31), "2027-01-01", True),
        (date(2027, 4, 1), "2028-01-01", False),
    ],
)
def test_the_check_gives_the_general_and_ma_periods_containing_or_after_as_of(
    as_of: date, starts: str, inside: bool
):
    result = check_enrollment_window(JUNE_2, as_of)

    for key in ("general_enrollment_period", "medicare_advantage_open_enrollment_period"):
        assert result[key]["starts"] == starts
        assert result[key]["contains_as_of"] is inside


@pytest.mark.parametrize(
    ("as_of", "starts", "inside"),
    [
        (date(2026, 10, 14), "2026-10-15", False),
        (date(2026, 10, 15), "2026-10-15", True),
        (date(2026, 12, 7), "2026-10-15", True),
        (date(2026, 12, 8), "2027-10-15", False),
    ],
)
def test_the_check_gives_the_open_enrollment_containing_or_after_as_of(
    as_of: date, starts: str, inside: bool
):
    result = check_enrollment_window(JUNE_2, as_of)["open_enrollment_period"]

    assert result["starts"] == starts
    assert result["contains_as_of"] is inside


def test_inside_open_enrollment_coverage_starts_on_january_1():
    result = check_enrollment_window(JUNE_2, date(2026, 11, 2))["open_enrollment_period"]

    assert result["coverage_start_if_signed_up_on_as_of"] == "2027-01-01"


def test_a_general_enrollment_signup_after_a_missed_window_starts_coverage_the_month_after():
    """Page 18: only someone whose Initial Enrollment Period has passed uses this period."""
    missed = check_enrollment_window(JUNE_2, date(2027, 2, 10))["general_enrollment_period"]
    still_in_iep = check_enrollment_window(date(1962, 3, 12), date(2027, 2, 10))

    assert missed["coverage_start_if_signed_up_on_as_of"] == "2027-03-01"
    assert still_in_iep["initial_enrollment_period"]["as_of_is"] == "inside"
    assert still_in_iep["general_enrollment_period"]["coverage_start_if_signed_up_on_as_of"] is None


# The whole check


def test_the_check_returns_iso_dates_and_the_pages_it_read():
    result = check_enrollment_window(MARCH_12, date(2026, 1, 10))

    assert result["birth_date"] == "1961-03-12"
    assert result["as_of"] == "2026-01-10"
    assert result["turns_65_on"] == "2026-03-12"
    assert result["initial_enrollment_period"]["starts"] == "2025-12-01"
    assert result["initial_enrollment_period"]["ends"] == "2026-06-30"
    assert result["pages"] == list(ENROLLMENT_PAGES) == [17, 18, 71, 72, 80]


def test_someone_decades_from_65_still_gets_their_future_window():
    result = check_enrollment_window(date(1990, 5, 5), date(2026, 10, 10))

    assert result["initial_enrollment_period"]["starts"] == "2055-02-01"
    assert result["initial_enrollment_period"]["as_of_is"] == "before"


def test_an_as_of_before_the_birth_date_is_refused():
    with pytest.raises(ToolError, match="before the birth date"):
        check_enrollment_window(date(2000, 1, 2), date(2000, 1, 1))


@pytest.mark.corpus
@pytest.mark.parametrize(("page", "quote"), RULE_QUOTES)
def test_every_rule_sentence_quoted_is_on_its_page(page: int, quote: str):
    assert collapse(quote) in handbook_pages()[page], f"not on page {page}: {quote}"
