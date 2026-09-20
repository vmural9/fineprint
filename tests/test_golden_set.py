"""Tests for the golden set, its loader, and its evidence quotes."""

import json
from pathlib import Path

import pytest

from evals.golden import (
    CATEGORIES,
    DEFAULT_GOLDEN_SET,
    DIFFICULTIES,
    TYPES,
    GoldenSetError,
    load_golden_set,
)
from evals.verify_golden_set import DEFAULT_PDF, check_item, read_pages

VALID_ITEM = {
    "id": "q001",
    "question": "How much is the Part B premium?",
    "expected_answer": "The standard Part B premium in 2026 is $202.90 a month.",
    "expected_pages": [23],
    "alt_pages": {},
    "category": "costs",
    "difficulty": "easy",
    "type": "lookup",
    "evidence": [{"page": 23, "quote": "The standard Part B premium amount in 2026 is $202.90."}],
    "needs_review": False,
}


def write_set(tmp_path: Path, *records: dict) -> Path:
    """Write records to a JSONL file and return its path."""
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


# --- the real file ----------------------------------------------------------------


def test_real_golden_set_loads_and_validates():
    items = load_golden_set(DEFAULT_GOLDEN_SET)
    assert items, "the golden set is empty"
    assert len({item.id for item in items}) == len(items)
    for item in items:
        assert item.category in CATEGORIES
        assert item.difficulty in DIFFICULTIES
        assert item.type in TYPES
        assert bool(item.expected_pages) is item.answerable
        assert set(item.alt_pages) <= set(item.expected_pages)
        assert all(item.alt_pages.values()), "an alt_pages entry lists no alternate"
        assert not set(item.alternate_pages) & set(item.expected_pages)


def test_real_golden_set_evidence_covers_every_expected_page():
    for item in load_golden_set(DEFAULT_GOLDEN_SET):
        if item.answerable and not item.needs_review:
            assert set(item.expected_pages) <= {entry.page for entry in item.evidence}


# --- the validator rejects broken items -------------------------------------------


def test_rejects_missing_field(tmp_path):
    record = {key: value for key, value in VALID_ITEM.items() if key != "expected_answer"}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: expected_answer: required field is missing" in str(excinfo.value)


def test_rejects_bad_enum(tmp_path):
    record = VALID_ITEM | {"category": "dental"}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: category: 'dental' is not one of" in str(excinfo.value)


def test_rejects_duplicate_id(tmp_path):
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, VALID_ITEM, dict(VALID_ITEM)))
    assert ":2: id: duplicate id 'q001' (first used on line 1)" in str(excinfo.value)


def test_rejects_unanswerable_item_with_pages(tmp_path):
    record = VALID_ITEM | {"id": "q002", "type": "unanswerable"}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: expected_pages: must be empty for an unanswerable question" in str(excinfo.value)


def test_rejects_bad_id_format(tmp_path):
    record = VALID_ITEM | {"id": "1"}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: id: expected the form q001" in str(excinfo.value)


def test_rejects_answerable_item_with_no_pages(tmp_path):
    record = VALID_ITEM | {"expected_pages": [], "evidence": []}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: expected_pages: must not be empty for a lookup question" in str(excinfo.value)


def test_rejects_expected_page_without_evidence(tmp_path):
    record = VALID_ITEM | {"expected_pages": [23, 24]}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: evidence: no quote for expected page(s) [24]" in str(excinfo.value)


def test_accepts_unverified_page_when_flagged(tmp_path):
    record = VALID_ITEM | {"expected_pages": [23, 24], "needs_review": True}
    (item,) = load_golden_set(write_set(tmp_path, record))
    assert item.needs_review


def test_accepts_alt_pages_keyed_by_expected_page(tmp_path):
    record = VALID_ITEM | {"alt_pages": {"23": [24, 25]}}
    (item,) = load_golden_set(write_set(tmp_path, record))
    assert item.alt_pages == {23: (24, 25)}
    assert item.alternate_pages == (24, 25)
    assert item.pages_that_satisfy(23) == (23, 24, 25)
    assert item.cited_pages == (23, 24, 25)


def test_rejects_alt_pages_key_that_is_not_an_expected_page(tmp_path):
    record = VALID_ITEM | {"alt_pages": {"24": [25]}}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: alt_pages['24']: key must be one of expected_pages [23]" in str(excinfo.value)


def test_rejects_alternate_that_is_also_an_expected_page(tmp_path):
    record = VALID_ITEM | {
        "expected_pages": [23, 24],
        "alt_pages": {"23": [24]},
        "evidence": VALID_ITEM["evidence"] + [{"page": 24, "quote": "anything"}],
    }
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: alt_pages['23']: page(s) [24] are already in expected_pages" in str(excinfo.value)


def test_rejects_alt_pages_that_is_not_an_object(tmp_path):
    record = VALID_ITEM | {"alt_pages": [24]}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: alt_pages: expected an object mapping an expected page" in str(excinfo.value)


def test_rejects_alt_pages_entry_with_no_alternates(tmp_path):
    record = VALID_ITEM | {"alt_pages": {"23": []}}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: alt_pages['23']: must list at least one alternate page" in str(excinfo.value)


def test_rejects_alt_pages_key_that_is_not_a_page_number(tmp_path):
    record = VALID_ITEM | {"alt_pages": {"page 23": [24]}}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    assert ":1: alt_pages['page 23']: key must be a 1-based page number" in str(excinfo.value)


def test_reports_every_problem_at_once(tmp_path):
    record = VALID_ITEM | {"id": "nope", "difficulty": "trivial"}
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(write_set(tmp_path, record))
    message = str(excinfo.value)
    assert ":1: id: expected the form q001" in message
    assert ":1: difficulty: 'trivial' is not one of" in message


# --- the evidence really occurs in the handbook -----------------------------------


@pytest.mark.skipif(
    not DEFAULT_PDF.is_file(),
    reason=f"handbook PDF not found at {DEFAULT_PDF}; run scripts/download_handbook.py",
)
def test_every_evidence_quote_occurs_on_its_page():
    items = load_golden_set(DEFAULT_GOLDEN_SET)
    wanted = {page for item in items for page in item.cited_pages}
    texts, page_count = read_pages(DEFAULT_PDF, wanted)
    problems = [
        f"{item.id}: {problem}" for item in items for problem in check_item(item, texts, page_count)
    ]
    assert not problems, "\n".join(problems)
