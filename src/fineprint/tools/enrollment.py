"""When someone can sign up for Medicare, and when their coverage starts, as dates.

"I turn 65 on March 12; when can I sign up?" has an exact answer, but it takes month arithmetic
that a model gets wrong often enough to matter. This module does the arithmetic from the rules the
handbook states, and only those. The sentences it implements, copied from the pinned handbook:

Page 17, the Initial Enrollment Period:
    "Generally, you can first sign up for Part A and/or Part B during the 7-month period that
    begins 3 months before the month you turn 65 and ends 3 months after the month you turn 65."
    "If your birthday is on the first of the month, your 7-month period starts 4 months before
    the month you turn 65 and ends 2 months after the month you turn 65."
    "Example: If you turn 65 on June 2, your 7-month period would begin in March and end in
    September. If you turn 65 on June 1, your 7-month period would begin in February and end in
    August."

Page 17, when coverage starts:
    "If you sign up for Part A and/or Part B during the first 3 months of your Initial Enrollment
    Period, in most cases, your coverage begins the first day of your birthday month. However, if
    your birthday is on the first day of the month, your coverage starts the first day of the
    prior month."
    "If you sign up the month you turn 65 or during the last 3 months of your Initial Enrollment
    Period, your coverage starts the first day of the month after you sign up."

Page 18, the General Enrollment Period:
    "You can sign up for Part B during the General Enrollment Period (January 1–March 31 each
    year) if you missed your Initial Enrollment Period and don’t qualify for a Special Enrollment
    Period."
    "When you sign up during the General Enrollment Period, your coverage starts the first day of
    the month after you sign up."

Pages 71 and 80, Open Enrollment:
    "You can join, switch, or drop a Medicare Advantage Plan (with or without drug coverage)
    during the Open Enrollment Period each year." "Your coverage starts on January 1 (as long as
    the plan gets your enrollment request by December 7)." (page 71)
    "Open Enrollment Period. From October 15 – December 7 each year, you can join, switch, or drop
    a plan." (page 80)

Pages 72 and 80, the Medicare Advantage Open Enrollment Period:
    "You can only make one change during this period, and any changes you make will be effective
    the first of the month after the plan gets your request." (page 72)
    "From January 1 – March 31 each year, you can switch to a different Medicare Advantage Plan or
    switch to Original Medicare (and join a separate Medicare drug plan) once during this time."
    (page 80)

Two cases the handbook does not settle are reported rather than guessed. Someone born on
February 29 has no 65th birthday in a non-leap year, and the handbook does not say which day
counts, so `turns_65_on` raises `ToolError`. And when the birthday is on the first of the month,
the window runs from 4 months before to 2 months after, so the month just before the birthday
month is neither one of "the first 3 months" nor "the month you turn 65" or "the last 3 months":
page 17 gives no coverage start for a sign-up in that month, and neither does this module.
"""

import calendar
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from fineprint.tools.errors import ToolError

# The pages a check reads, in the order the handbook prints them.
ENROLLMENT_PAGES = (17, 18, 71, 72, 80)

# The rule sentences above, as data: a test checks each against its page, and the tool returns
# them as its source text so an answer that uses a date can be checked against the handbook.
RULE_QUOTES: tuple[tuple[int, str], ...] = (
    (
        17,
        "Generally, you can first sign up for Part A and/or Part B during the 7-month period "
        "that begins 3 months before the month you turn 65 and ends 3 months after the month "
        "you turn 65.",
    ),
    (
        17,
        "If your birthday is on the first of the month, your 7-month period starts 4 months "
        "before the month you turn 65 and ends 2 months after the month you turn 65.",
    ),
    (
        17,
        "Example: If you turn 65 on June 2, your 7-month period would begin in March and end in "
        "September. If you turn 65 on June 1, your 7-month period would begin in February and "
        "end in August.",
    ),
    (
        17,
        "If you sign up for Part A and/or Part B during the first 3 months of your Initial "
        "Enrollment Period, in most cases, your coverage begins the first day of your birthday "
        "month. However, if your birthday is on the first day of the month, your coverage "
        "starts the first day of the prior month.",
    ),
    (
        17,
        "If you sign up the month you turn 65 or during the last 3 months of your Initial "
        "Enrollment Period, your coverage starts the first day of the month after you sign up.",
    ),
    (
        18,
        "You can sign up for Part B during the General Enrollment Period (January 1–March 31 "
        "each year) if you missed your Initial Enrollment Period and don’t qualify for a "
        "Special Enrollment Period.",
    ),
    (
        18,
        "When you sign up during the General Enrollment Period, your coverage starts the first "
        "day of the month after you sign up.",
    ),
    (
        71,
        "You can join, switch, or drop a Medicare Advantage Plan (with or without drug coverage) "
        "during the Open Enrollment Period each year.",
    ),
    (
        71,
        "Your coverage starts on January 1 (as long as the plan gets your enrollment request by "
        "December 7).",
    ),
    (
        72,
        "You can only make one change during this period, and any changes you make will be "
        "effective the first of the month after the plan gets your request.",
    ),
    (
        80,
        "Open Enrollment Period. From October 15 – December 7 each year, you can join, switch, "
        "or drop a plan.",
    ),
    (
        80,
        "From January 1 – March 31 each year, you can switch to a different Medicare Advantage "
        "Plan or switch to Original Medicare (and join a separate Medicare drug plan) once "
        "during this time (page 72).",
    ),
)

SOURCE_TEXT = "\n".join(f"Page {page}: {quote}" for page, quote in RULE_QUOTES)

LATER_PERIODS_NOTE = (
    "The General Enrollment, Open Enrollment and Medicare Advantage Open Enrollment Periods "
    "apply once the Initial Enrollment Period has passed."
)

LEAP_DAY_MESSAGE = (
    "The handbook does not say when someone born on February 29 turns 65 for Medicare, so their "
    "Initial Enrollment Period cannot be worked out from it; call Social Security or "
    "1-800-MEDICARE."
)


@dataclass(frozen=True)
class Window:
    """A period, first and last day included, and the handbook page that defines it."""

    name: str
    starts: date
    ends: date
    page: int
    note: str

    def contains(self, day: date) -> bool:
        return self.starts <= day <= self.ends


def _shift_month(day: date, months: int) -> tuple[int, int]:
    """The (year, month) that is `months` calendar months from the month of `day`."""
    index = day.year * 12 + (day.month - 1) + months
    return index // 12, index % 12 + 1


def _first_of_month(year: int, month: int) -> date:
    return date(year, month, 1)


def _last_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _first_of_next_month(day: date) -> date:
    return _first_of_month(*_shift_month(day, 1))


def _months_between(earlier: date, later: date) -> int:
    """How many calendar months the month of `later` is after the month of `earlier`."""
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def turns_65_on(birth_date: date) -> date:
    """The 65th birthday. Raises `ToolError` for February 29, which the handbook does not cover."""
    if (birth_date.month, birth_date.day) == (2, 29):
        raise ToolError(LEAP_DAY_MESSAGE)
    return birth_date.replace(year=birth_date.year + 65)


def initial_enrollment_period(birth_date: date) -> Window:
    """The 7-month window around the month someone turns 65 (page 17)."""
    birthday = turns_65_on(birth_date)
    before, after = (4, 2) if birthday.day == 1 else (3, 3)
    return Window(
        name="Initial Enrollment Period",
        starts=_first_of_month(*_shift_month(birthday, -before)),
        ends=_last_of_month(*_shift_month(birthday, after)),
        page=17,
        note=(
            f"The 7-month period to first sign up for Part A and/or Part B: {before} months "
            f"before the month of the 65th birthday to {after} months after it"
            + (", because the birthday is on the first of the month." if before == 4 else ".")
        ),
    )


def _iep_coverage(birth_date: date, signup_date: date) -> tuple[date | None, str]:
    """When coverage starts for a sign-up on `signup_date`, and the page 17 rule that says so."""
    window = initial_enrollment_period(birth_date)
    if not window.contains(signup_date):
        return None, "The sign-up date is outside the Initial Enrollment Period."
    birthday = turns_65_on(birth_date)
    if _months_between(window.starts, signup_date) < 3:
        if birthday.day == 1:
            starts = _first_of_month(*_shift_month(birthday, -1))
            return starts, (
                "Signed up in the first 3 months of the Initial Enrollment Period with a "
                "birthday on the first of the month: coverage starts the first day of the "
                "month before the birthday month."
            )
        return birthday.replace(day=1), (
            "Signed up in the first 3 months of the Initial Enrollment Period: in most cases, "
            "coverage begins the first day of the birthday month."
        )
    if _months_between(birthday, signup_date) >= 0:
        return _first_of_next_month(signup_date), (
            "Signed up in the month of the 65th birthday or the last 3 months of the Initial "
            "Enrollment Period: coverage starts the first day of the month after signing up."
        )
    return None, (
        "The birthday is on the first of the month, so this is the 4th month of the Initial "
        "Enrollment Period, which is neither one of its first 3 months nor the birthday month "
        "or its last 3 months; the handbook does not say when coverage starts for a sign-up "
        "in this month."
    )


def part_b_coverage_start(birth_date: date, signup_date: date) -> date | None:
    """When Part A and/or Part B coverage starts for a sign-up in the Initial Enrollment Period.

    None when `signup_date` is outside the window, or in the one month page 17 does not cover.
    """
    return _iep_coverage(birth_date, signup_date)[0]


def general_enrollment_period(year: int) -> Window:
    """January 1 to March 31 (page 18)."""
    return Window(
        name="General Enrollment Period",
        starts=date(year, 1, 1),
        ends=date(year, 3, 31),
        page=18,
        note=(
            "To sign up for Part B (and Part A if it has to be bought) after missing the Initial "
            "Enrollment Period, when no Special Enrollment Period applies. Coverage starts the "
            "first day of the month after signing up, and a late enrollment penalty may apply."
        ),
    )


def open_enrollment_period(year: int) -> Window:
    """October 15 to December 7 (pages 71 and 80)."""
    return Window(
        name="Open Enrollment Period",
        starts=date(year, 10, 15),
        ends=date(year, 12, 7),
        page=71,
        note=(
            "To join, switch, or drop a Medicare Advantage Plan or a Medicare drug plan. "
            "Coverage starts January 1 as long as the plan gets the request by December 7."
        ),
    )


def medicare_advantage_open_enrollment(year: int) -> Window:
    """January 1 to March 31, for people already in a Medicare Advantage Plan (pages 72 and 80)."""
    return Window(
        name="Medicare Advantage Open Enrollment Period",
        starts=date(year, 1, 1),
        ends=date(year, 3, 31),
        page=72,
        note=(
            "Only for people already in a Medicare Advantage Plan: one change, to another "
            "Medicare Advantage Plan or to Original Medicare (with a separate drug plan if "
            "wanted). It takes effect the first of the month after the plan gets the request."
        ),
    )


def _yearly_window_from(as_of: date, window_for_year: Callable[[int], Window]) -> Window:
    """This year's window if `as_of` is not past its end, otherwise next year's."""
    window = window_for_year(as_of.year)
    return window if as_of <= window.ends else window_for_year(as_of.year + 1)


def _describe(window: Window) -> dict[str, Any]:
    return {
        "name": window.name,
        "starts": window.starts.isoformat(),
        "ends": window.ends.isoformat(),
        "page": window.page,
        "note": window.note,
    }


def _later_periods(as_of: date) -> dict[str, Any]:
    """The General, Open and Medicare Advantage Open Enrollment Periods containing or after `as_of`.

    Only asked for once the Initial Enrollment Period has passed, so a sign-up in the General
    Enrollment Period is one by someone who missed their Initial Enrollment Period.
    """
    general = _yearly_window_from(as_of, general_enrollment_period)
    open_enrollment = _yearly_window_from(as_of, open_enrollment_period)
    advantage = _yearly_window_from(as_of, medicare_advantage_open_enrollment)
    return {
        "general_enrollment_period": {
            **_describe(general),
            "contains_as_of": general.contains(as_of),
            "coverage_start_if_signed_up_on_as_of": (
                _first_of_next_month(as_of).isoformat() if general.contains(as_of) else None
            ),
        },
        "open_enrollment_period": {
            **_describe(open_enrollment),
            "contains_as_of": open_enrollment.contains(as_of),
            "coverage_start_if_signed_up_on_as_of": (
                date(open_enrollment.ends.year + 1, 1, 1).isoformat()
                if open_enrollment.contains(as_of)
                else None
            ),
        },
        "medicare_advantage_open_enrollment_period": {
            **_describe(advantage),
            "contains_as_of": advantage.contains(as_of),
            "change_effective_if_plan_gets_request_on_as_of": (
                _first_of_next_month(as_of).isoformat() if advantage.contains(as_of) else None
            ),
        },
    }


def check_enrollment_window(birth_date: date, as_of: date) -> dict[str, Any]:
    """Every enrollment window that matters on `as_of` for someone born on `birth_date`.

    Returns the Initial Enrollment Period and whether `as_of` is before, inside or after it; the
    coverage start for a sign-up on `as_of` inside it; once it has passed, the General, Open and
    Medicare Advantage Open Enrollment Periods that contain `as_of` or come next (before then, a
    one-line `later_periods` note instead); and the pages, with every date in ISO form. Raises
    `ToolError` when `as_of` is before the birth date or the birth date is February 29.
    """
    if as_of < birth_date:
        raise ToolError(
            f"The reference date {as_of.isoformat()} is before the birth date "
            f"{birth_date.isoformat()}."
        )
    birthday = turns_65_on(birth_date)
    iep = initial_enrollment_period(birth_date)
    if as_of < iep.starts:
        position = "before"
    elif as_of > iep.ends:
        position = "after"
    else:
        position = "inside"
    coverage, rule = _iep_coverage(birth_date, as_of)

    result: dict[str, Any] = {
        "birth_date": birth_date.isoformat(),
        "as_of": as_of.isoformat(),
        "turns_65_on": birthday.isoformat(),
        "initial_enrollment_period": {**_describe(iep), "as_of_is": position},
        "coverage_start_if_signed_up_on_as_of": coverage.isoformat() if coverage else None,
        "coverage_start_rule": rule,
        "pages": list(ENROLLMENT_PAGES),
    }
    if position == "after":
        result.update(_later_periods(as_of))
    else:
        result["later_periods"] = LATER_PERIODS_NOTE
    return result
