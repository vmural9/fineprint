"""Tests for handbook extraction, run against the real PDF.

Every test here reads the handbook, so the whole module skips when it has not been downloaded.
Nothing here touches the network, the database, or AWS.
"""

import re
from pathlib import Path

import pytest

from fineprint.pdf import Page, extract_pages

HANDBOOK = Path(__file__).resolve().parents[1] / "data" / "raw" / "medicare-and-you-2026.pdf"

# Recorded under "Corpus facts": 128 pages, cover to back cover, no translation of page numbers.
PAGE_COUNT = 128

# Three quotes lifted from the `evidence` entries of the golden set, each with the page it names.
KNOWN_PHRASES = [
    (
        17,
        "Generally, you can first sign up for Part A and/or Part B during the 7-month "
        "period that begins 3 months before the month you turn 65 and ends 3 months "
        "after the month you turn 65.",
    ),
    (23, "The standard Part B premium amount in 2026 is $202.90."),
    (
        83,
        "Your yearly out-of-pocket drug costs for drugs covered by your plan are capped "
        "at $2,100 in 2026.",
    ),
]

pytestmark = pytest.mark.skipif(
    not HANDBOOK.is_file(),
    reason=f"handbook PDF not found at {HANDBOOK}; run: uv run python scripts/download_handbook.py",
)


def unwrapped(text: str) -> str:
    """Collapse every run of whitespace to one space, so a quote survives line wrapping."""
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture(scope="module")
def pages() -> list[Page]:
    """The whole handbook, extracted once for the module."""
    return extract_pages(HANDBOOK)


def test_every_pdf_page_becomes_one_page(pages: list[Page]):
    """128 pages in, 128 pages out: nothing is dropped, merged, or added."""
    assert len(pages) == PAGE_COUNT


def test_page_numbers_are_the_1_based_pdf_page_index(pages: list[Page]):
    """The number a reader sees on the page is the index the code uses, with no translation."""
    assert [page.number for page in pages] == list(range(1, PAGE_COUNT + 1))


@pytest.mark.parametrize(("number", "phrase"), KNOWN_PHRASES, ids=lambda value: str(value)[:20])
def test_known_phrases_land_on_the_page_the_golden_set_names(
    pages: list[Page], number: int, phrase: str
):
    """A cited page is only worth printing if the text really came from that page."""
    text = unwrapped(pages[number - 1].text)

    assert pages[number - 1].number == number
    assert phrase in text


def test_a_phrase_does_not_also_appear_on_the_neighbouring_pages(pages: list[Page]):
    """An off-by-one in the numbering would show up as the premium sitting on page 22 or 24."""
    number, phrase = KNOWN_PHRASES[1]
    neighbours = [unwrapped(pages[number - 2].text), unwrapped(pages[number].text)]

    assert all(phrase not in neighbour for neighbour in neighbours)


def test_extraction_keeps_the_running_header_and_the_folio(pages: list[Page]):
    """Part 1 is the naive baseline on purpose: no header, index, or contents is stripped."""
    page = pages[22]

    assert page.text.startswith("Section 1: Signing up for Medicare\n23\n")
