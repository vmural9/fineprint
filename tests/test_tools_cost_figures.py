"""Tests for the table of 2026 dollar amounts the cost-figure tool answers from.

The unit tests pin the table's shape and its amounts. The `corpus` test checks every figure
against the handbook itself: the sentence it quotes must be on the page it names, and the amount
must be in that sentence. It reads the extracted page text in `data/interim/pages-2026.jsonl`,
falls back to extracting the PDF, and skips with the fetch command when neither is there.
"""

import json
from decimal import Decimal
from functools import cache
from pathlib import Path

import pytest

from fineprint.tools.cost_figures import COST_FIGURES, CostFigure, lookup_cost_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES_JSONL = REPO_ROOT / "data" / "interim" / "pages-2026.jsonl"
HANDBOOK = REPO_ROOT / "data" / "raw" / "medicare-and-you-2026.pdf"

EXPECTED_KEYS = {
    "part_b_standard_premium",
    "part_b_deductible",
    "part_a_deductible",
    "part_a_coinsurance_days_61_90",
    "part_a_coinsurance_lifetime_reserve_days",
    "snf_coinsurance_days_21_100",
    "part_d_out_of_pocket_cap",
    "insulin_monthly_cap",
}


def collapse(text: str) -> str:
    """`text` with every run of whitespace turned into one space."""
    return " ".join(text.split())


@cache
def handbook_pages() -> dict[int, str]:
    """Every page of the 2026 handbook by page number, whitespace collapsed, or a skip."""
    if PAGES_JSONL.is_file():
        with PAGES_JSONL.open(encoding="utf-8") as lines:
            records = [json.loads(line) for line in lines if line.strip()]
        return {record["page"]: collapse(record["text"]) for record in records}
    if HANDBOOK.is_file():
        from fineprint.pdf import extract_pages

        return {page.number: collapse(page.text) for page in extract_pages(HANDBOOK)}
    pytest.skip(
        f"no handbook text at {PAGES_JSONL} and no PDF at {HANDBOOK}; "
        "run: uv run python scripts/download_handbook.py"
    )


def test_the_table_holds_exactly_the_agreed_figures():
    assert set(COST_FIGURES) == EXPECTED_KEYS


def test_every_figure_is_filed_under_its_own_key():
    for key, figure in COST_FIGURES.items():
        assert figure.key == key


def test_the_amounts_are_the_ones_the_handbook_prints():
    """Copied from the page text; the corpus test below checks them against the handbook."""
    amounts = {key: (figure.amount, figure.page) for key, figure in COST_FIGURES.items()}

    assert amounts == {
        "part_b_standard_premium": (Decimal("202.90"), 23),
        "part_b_deductible": (Decimal("283"), 30),
        "part_a_deductible": (Decimal("1736"), 27),
        "part_a_coinsurance_days_61_90": (Decimal("434"), 27),
        "part_a_coinsurance_lifetime_reserve_days": (Decimal("868"), 27),
        "snf_coinsurance_days_21_100": (Decimal("217"), 29),
        "part_d_out_of_pocket_cap": (Decimal("2100"), 83),
        "insulin_monthly_cap": (Decimal("35"), 88),
    }


def test_amounts_are_exact_decimals_so_the_premium_keeps_its_cents():
    premium = COST_FIGURES["part_b_standard_premium"].amount

    assert isinstance(premium, Decimal)
    assert str(premium) == "202.90"


def test_every_figure_says_what_it_costs_per_and_who_pays_it():
    units = {"per month", "per year", "per benefit period", "per day", "per one-month supply"}
    for figure in COST_FIGURES.values():
        assert figure.unit in units, figure.key
        assert figure.label and figure.applies_to and figure.source_text, figure.key


def test_there_is_no_part_d_deductible_because_the_handbook_prints_none():
    """Golden question q037 is unanswerable for exactly this reason."""
    assert not any("deductible" in key and "part_d" in key for key in COST_FIGURES)


def test_lookup_returns_the_figure_and_refuses_an_unknown_key():
    assert lookup_cost_figure("part_b_deductible") is COST_FIGURES["part_b_deductible"]
    assert isinstance(lookup_cost_figure("insulin_monthly_cap"), CostFigure)
    with pytest.raises(KeyError):
        lookup_cost_figure("part_d_deductible")


@pytest.mark.corpus
@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_every_source_text_is_on_its_page_and_holds_its_amount(key: str):
    figure = COST_FIGURES[key]
    page_text = handbook_pages()[figure.page]
    source_text = collapse(figure.source_text)

    assert source_text in page_text, f"{key}: source_text is not on page {figure.page}"
    assert f"${figure.amount:,}" in source_text, f"{key}: ${figure.amount:,} not in its sentence"
