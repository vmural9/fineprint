"""The sections chunker on the real handbook.

Every test that reads the handbook asks for it through `require_handbook`, which skips with the
command that fetches it when `data/raw/medicare-and-you-2026.pdf` is missing, and the module
carries the `corpus` marker, so `uv run pytest -m corpus` runs exactly these. Nothing here
touches the network, the database, or AWS.
"""

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import pytest

from fineprint.chunking import SECTION_WORDS, Block, Chunk, section_chunks, split_blocks
from fineprint.pdf import Page, extract_pages

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = REPO_ROOT / "data" / "raw" / "medicare-and-you-2026.pdf"
GOLDEN_SET = REPO_ROOT / "evals" / "golden_set.jsonl"

pytestmark = pytest.mark.corpus


def require_handbook() -> Path:
    """The path to the handbook PDF, or a skip that says how to fetch it."""
    if not HANDBOOK.is_file():
        pytest.skip(
            f"handbook PDF not found at {HANDBOOK}; run: uv run python scripts/download_handbook.py"
        )
    return HANDBOOK


def collapse(text: str) -> str:
    """`text` with every run of whitespace turned into one space, as chunk text is stored."""
    return " ".join(text.split())


@dataclass(frozen=True)
class Chart:
    """A table of the handbook, named by its first and last words on the pages it covers."""

    pages: tuple[int, ...]
    first: str
    last: str


WHO_PAYS_FIRST = Chart(
    pages=(21,),
    first="If you have retiree health coverage, like insurance from your or your spouse’s "
    "former employment…",
    last="If you have Medicaid... Medicare pays first.",
)
HEALTH_SAVINGS_ACCOUNT = Chart(
    pages=(20,),
    first="If you sign up for Medicare: During your Initial Enrollment Period",
    last="stopping HSA contributions 6 months before the month you apply for Medicare.",
)
AT_A_GLANCE = Chart(
    pages=(11, 12),
    first="Doctor & hospital choice Original Medicare Medicare Advantage (Part C)",
    last="an extra benefit that covers emergency and urgently needed services when traveling "
    "outside the U.S.",
)
# The chart's footnotes explain its asterisks and are part of it.
MEDIGAP_PLANS = Chart(
    pages=(76,),
    first="Medigap standardized plans Benefits A B C D F* G* K L M N",
    last="up to a $50 copayment for emergency room visits that don’t result in an inpatient "
    "admission.",
)
ENROLLMENT_PERIODS = Chart(
    pages=(71, 72),
    first="Initial Enrollment Period (page 17) When you first become eligible for Medicare",
    last="to switch from your current Medicare plan to a Medicare plan with a 5-star quality "
    "rating.",
)

# The table each `table` question of the golden set is answered from.
TABLE_QUESTIONS = {
    "q005": WHO_PAYS_FIRST,
    "q008": HEALTH_SAVINGS_ACCOUNT,
    "q009": AT_A_GLANCE,
    "q014": MEDIGAP_PLANS,
    "q027": ENROLLMENT_PERIODS,
    "q028": ENROLLMENT_PERIODS,
    "q030": ENROLLMENT_PERIODS,
    "q031": AT_A_GLANCE,
    "q038": MEDIGAP_PLANS,
}


def golden_items() -> dict[str, dict]:
    """The golden set's questions by id."""
    with GOLDEN_SET.open(encoding="utf-8") as lines:
        items = [json.loads(line) for line in lines if line.strip()]
    return {item["id"]: item for item in items}


@pytest.fixture(scope="module")
def pages() -> list[Page]:
    """The whole handbook, extracted once for the module."""
    return extract_pages(require_handbook())


@pytest.fixture(scope="module")
def blocks(pages: list[Page]) -> list[Block]:
    """The handbook's headings, paragraphs and tables, as the chunker reads them."""
    return split_blocks(pages)


@pytest.fixture(scope="module")
def chunks(pages: list[Page]) -> list[Chunk]:
    """The `sections` chunk set, with its default size."""
    return section_chunks(pages)


def word_offsets(sizes: list[int]) -> list[int]:
    """Where each of a run of pieces starts, counted in words from the start of the handbook."""
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    return offsets


def test_every_chunk_has_a_section(chunks: list[Chunk]):
    assert chunks
    assert [chunk.ordinal for chunk in chunks if chunk.section is None] == []


def test_the_chunks_hold_every_word_the_chunker_read_once_and_in_order(
    blocks: list[Block], chunks: list[Chunk]
):
    """Nothing is lost or repeated between reading the handbook and cutting it up."""
    read = [word for block in blocks for word, _ in block.words]

    assert [word for chunk in chunks for word in chunk.text.split()] == read


def test_a_heading_only_ever_opens_a_chunk(blocks: list[Block], chunks: list[Chunk]):
    """Every heading the chunker found starts a chunk; none sits inside one."""
    chunk_starts = set(word_offsets([len(chunk.text.split()) for chunk in chunks]))
    block_starts = word_offsets([len(block.words) for block in blocks])

    buried = [
        block.text
        for block, start in zip(blocks, block_starts, strict=True)
        if block.kind == "heading" and start not in chunk_starts
    ]

    assert [block for block in blocks if block.kind == "heading"], "the handbook has headings"
    assert buried == []


def test_only_a_table_or_a_heading_takes_a_chunk_past_max_words(
    blocks: list[Block], chunks: list[Chunk]
):
    """Prose is packed into at most 300 words, not counting the heading that opens a section."""
    block_starts = word_offsets([len(block.words) for block in blocks])
    chunk_starts = word_offsets([len(chunk.text.split()) for chunk in chunks])

    too_long = []
    for chunk, start in zip(chunks, chunk_starts, strict=True):
        end = start + len(chunk.text.split())
        covered = blocks[
            bisect_right(block_starts, start) - 1 : bisect_right(block_starts, end - 1)
        ]
        if any(block.kind == "table" for block in covered):
            continue
        heading = len(covered[0].words) if covered[0].kind == "heading" else 0
        if end - start - heading > SECTION_WORDS:
            too_long.append(chunk.ordinal)

    assert too_long == []


def test_no_running_title_or_page_number_is_left_in_the_text(
    pages: list[Page], chunks: list[Chunk]
):
    """Where a page opens with its running title and its number, neither reaches a chunk."""
    furniture = []
    for page in pages:
        lines = page.text.split("\n")
        if lines[0].startswith("Section ") and lines[1:2] == [str(page.number)]:
            furniture.append(f"{collapse(lines[0])} {page.number}")

    assert len(furniture) > 90, "most pages open with a running title"
    assert [text for text in furniture if any(text in chunk.text for chunk in chunks)] == []


def test_every_table_question_of_the_golden_set_names_its_table():
    """A `table` question added to the golden set must say which chart it reads."""
    table_questions = {
        question_id for question_id, item in golden_items().items() if item["type"] == "table"
    }

    assert table_questions == set(TABLE_QUESTIONS)


@pytest.mark.parametrize("question_id", sorted(TABLE_QUESTIONS))
def test_the_table_a_question_reads_is_within_one_chunk(
    question_id: str, pages: list[Page], chunks: list[Chunk]
):
    """The chart on the question's expected pages, first word to last, is in a single chunk."""
    chart = TABLE_QUESTIONS[question_id]
    expected_pages = golden_items()[question_id]["expected_pages"]
    printed = collapse(" ".join(pages[number - 1].text for number in chart.pages))
    assert set(expected_pages) <= set(chart.pages), "the chart is on the question's pages"
    assert chart.first in printed and chart.last in printed, "both ends are the handbook's words"

    holding = [
        chunk
        for chunk in chunks
        if chart.first in chunk.text
        and chart.last in chunk.text
        and chunk.text.index(chart.first) < chunk.text.index(chart.last)
    ]

    assert len(holding) == 1
    assert (holding[0].page_start, holding[0].page_end) == (chart.pages[0], chart.pages[-1])
