"""The golden-set schema and a strict loader for ``evals/golden_set.jsonl``."""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

CATEGORIES = frozenset(
    {"costs", "coverage", "enrollment", "original_vs_advantage", "part_d", "medigap"}
)
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
TYPES = frozenset({"lookup", "table", "multi_section", "unanswerable"})

ID_PATTERN = re.compile(r"^q\d{3}$")
PAGE_KEY_PATTERN = re.compile(r"[1-9][0-9]*")

DEFAULT_GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.jsonl"

_REQUIRED_FIELDS = (
    "id",
    "question",
    "expected_answer",
    "expected_pages",
    "category",
    "difficulty",
    "type",
    "evidence",
)
_OPTIONAL_FIELDS = ("alt_pages", "needs_review")


class GoldenSetError(ValueError):
    """Raised when the golden set does not satisfy the schema."""


@dataclass(frozen=True, slots=True)
class Evidence:
    """A verbatim span from one page of the handbook supporting an expected answer."""

    page: int
    quote: str


@dataclass(frozen=True, slots=True)
class GoldenItem:
    """One evaluation question with its expected answer and its source pages."""

    id: str
    question: str
    expected_answer: str
    expected_pages: tuple[int, ...]
    category: str
    difficulty: str
    type: str
    evidence: tuple[Evidence, ...]
    alt_pages: Mapping[int, tuple[int, ...]] = MappingProxyType({})
    needs_review: bool = False

    @property
    def answerable(self) -> bool:
        """True when the handbook is expected to answer the question."""
        return self.type != "unanswerable"

    @property
    def alternate_pages(self) -> tuple[int, ...]:
        """Every alternate page, from every expected page, in ascending order."""
        return tuple(sorted({page for pages in self.alt_pages.values() for page in pages}))

    def pages_that_satisfy(self, expected_page: int) -> tuple[int, ...]:
        """The expected page plus any page that states the same fact, in ascending order."""
        return tuple(sorted({expected_page, *self.alt_pages.get(expected_page, ())}))

    @property
    def cited_pages(self) -> tuple[int, ...]:
        """Every page number the item names, in ascending order."""
        pages = set(self.expected_pages) | set(self.alternate_pages)
        pages.update(item.page for item in self.evidence)
        return tuple(sorted(pages))


@dataclass
class _Problems:
    """Collects validation failures so one load reports all of them."""

    source: str
    messages: list[str] = field(default_factory=list)

    def add(self, line_number: int, field_name: str, message: str) -> None:
        self.messages.append(f"{self.source}:{line_number}: {field_name}: {message}")


def load_golden_set(path: str | Path = DEFAULT_GOLDEN_SET) -> list[GoldenItem]:
    """Read and validate a golden-set JSONL file, reporting every problem found.

    Raises:
        GoldenSetError: if the file is missing or any line breaks the schema.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        raise GoldenSetError(f"{path}: cannot be read: {exc}") from exc

    problems = _Problems(str(path))
    items: list[GoldenItem] = []
    seen_ids: dict[str, int] = {}

    for line_number, line in _numbered_lines(raw):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.add(line_number, "<line>", f"not valid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict):
            problems.add(line_number, "<line>", "expected a JSON object")
            continue

        item = _parse_item(record, line_number, problems)
        if item is None:
            continue
        first_seen = seen_ids.get(item.id)
        if first_seen is not None:
            problems.add(
                line_number, "id", f"duplicate id {item.id!r} (first used on line {first_seen})"
            )
            continue
        seen_ids[item.id] = line_number
        items.append(item)

    if not items and not problems.messages:
        problems.add(1, "<file>", "the golden set is empty")
    if problems.messages:
        raise GoldenSetError("\n".join(problems.messages))
    return items


def _numbered_lines(raw: str) -> Iterator[tuple[int, str]]:
    """Yield (1-based line number, text) for every non-blank line."""
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if line.strip():
            yield line_number, line


def _parse_item(record: dict[str, Any], line: int, problems: _Problems) -> GoldenItem | None:
    """Validate one record. Returns None when it cannot be turned into a GoldenItem."""
    before = len(problems.messages)

    unknown = sorted(set(record) - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS))
    for name in unknown:
        problems.add(line, name, "unknown field")
    for name in _REQUIRED_FIELDS:
        if name not in record:
            problems.add(line, name, "required field is missing")
    if len(problems.messages) != before:
        return None

    item_id = _text(record, "id", line, problems)
    if item_id is not None and not ID_PATTERN.match(item_id):
        problems.add(line, "id", f"expected the form q001, got {item_id!r}")
    question = _text(record, "question", line, problems)
    expected_answer = _text(record, "expected_answer", line, problems)
    category = _enum(record, "category", CATEGORIES, line, problems)
    difficulty = _enum(record, "difficulty", DIFFICULTIES, line, problems)
    item_type = _enum(record, "type", TYPES, line, problems)
    expected_pages = _pages(record, "expected_pages", line, problems)
    alt_pages = _alt_pages(record, expected_pages, line, problems)
    evidence = _evidence(record, line, problems)

    needs_review = record.get("needs_review", False)
    if not isinstance(needs_review, bool):
        problems.add(line, "needs_review", f"expected a boolean, got {_type_name(needs_review)}")
        needs_review = False

    # Every helper above records a problem when it returns None, so this one guard covers
    # both "a field was rejected" and "a field is None".
    parsed = (item_id, question, expected_answer, category, difficulty, item_type)
    if len(problems.messages) != before or None in parsed:
        return None

    _check_pages(
        line,
        problems,
        item_type=item_type,
        expected_pages=expected_pages,
        evidence=evidence,
        needs_review=needs_review,
    )
    if len(problems.messages) != before:
        return None

    return GoldenItem(
        id=item_id,
        question=question,
        expected_answer=expected_answer,
        expected_pages=expected_pages,
        category=category,
        difficulty=difficulty,
        type=item_type,
        evidence=evidence,
        alt_pages=alt_pages,
        needs_review=needs_review,
    )


def _check_pages(
    line: int,
    problems: _Problems,
    *,
    item_type: str,
    expected_pages: tuple[int, ...],
    evidence: tuple[Evidence, ...],
    needs_review: bool,
) -> None:
    """Apply the cross-field rules that tie pages, evidence, and question type together."""
    if item_type == "unanswerable" and expected_pages:
        problems.add(line, "expected_pages", "must be empty for an unanswerable question")
    if item_type != "unanswerable" and not expected_pages:
        problems.add(line, "expected_pages", f"must not be empty for a {item_type} question")

    if item_type != "unanswerable" and not needs_review:
        with_evidence = {entry.page for entry in evidence}
        missing = sorted(set(expected_pages) - with_evidence)
        if missing:
            problems.add(
                line,
                "evidence",
                f"no quote for expected page(s) {missing}; set needs_review to flag this instead",
            )


def _text(record: dict[str, Any], name: str, line: int, problems: _Problems) -> str | None:
    """Read a required non-empty string field."""
    value = record[name]
    if not isinstance(value, str):
        problems.add(line, name, f"expected a string, got {_type_name(value)}")
        return None
    if not value.strip():
        problems.add(line, name, "must not be empty")
        return None
    return value


def _enum(
    record: dict[str, Any], name: str, allowed: frozenset[str], line: int, problems: _Problems
) -> str | None:
    """Read a required string field constrained to a fixed set of values."""
    value = _text(record, name, line, problems)
    if value is None:
        return None
    if value not in allowed:
        problems.add(line, name, f"{value!r} is not one of {sorted(allowed)}")
        return None
    return value


def _pages(record: dict[str, Any], name: str, line: int, problems: _Problems) -> tuple[int, ...]:
    """Read a list of 1-based page numbers."""
    return _page_list(record[name], name, line, problems) or ()


def _page_list(value: Any, where: str, line: int, problems: _Problems) -> tuple[int, ...] | None:
    """Read one list of 1-based page numbers. Returns None when the list is unusable."""
    if not isinstance(value, list):
        problems.add(line, where, f"expected a list of page numbers, got {_type_name(value)}")
        return None
    pages: list[int] = []
    for position, page in enumerate(value):
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            problems.add(line, where, f"item {position} is not a 1-based integer page number")
            continue
        pages.append(page)
    if len(pages) != len(value):
        return None
    duplicates = sorted({page for page in pages if pages.count(page) > 1})
    if duplicates:
        problems.add(line, where, f"page(s) {duplicates} listed more than once")
        return None
    return tuple(pages)


def _alt_pages(
    record: dict[str, Any], expected_pages: tuple[int, ...], line: int, problems: _Problems
) -> Mapping[int, tuple[int, ...]]:
    """Read the ``{"<expected page>": [pages that state the same fact]}`` object."""
    if "alt_pages" not in record:
        return MappingProxyType({})
    value = record["alt_pages"]
    if not isinstance(value, dict):
        problems.add(
            line,
            "alt_pages",
            "expected an object mapping an expected page to its alternates, "
            f"got {_type_name(value)}",
        )
        return MappingProxyType({})

    expected = set(expected_pages)
    alternates: dict[int, tuple[int, ...]] = {}
    for key, raw in value.items():
        where = f"alt_pages[{key!r}]"
        if not PAGE_KEY_PATTERN.fullmatch(key):
            problems.add(line, where, 'key must be a 1-based page number as a string, like "30"')
            continue
        page = int(key)
        if page not in expected:
            problems.add(line, where, f"key must be one of expected_pages {list(expected_pages)}")
            continue
        pages = _page_list(raw, where, line, problems)
        if pages is None:
            continue
        if not pages:
            problems.add(line, where, "must list at least one alternate page")
            continue
        also_expected = sorted(set(pages) & expected)
        if also_expected:
            problems.add(line, where, f"page(s) {also_expected} are already in expected_pages")
            continue
        alternates[page] = pages
    return MappingProxyType(alternates)


def _evidence(record: dict[str, Any], line: int, problems: _Problems) -> tuple[Evidence, ...]:
    """Read the list of ``{page, quote}`` objects."""
    value = record["evidence"]
    if not isinstance(value, list):
        problems.add(line, "evidence", f"expected a list, got {_type_name(value)}")
        return ()
    entries: list[Evidence] = []
    for position, raw in enumerate(value):
        where = f"evidence[{position}]"
        if not isinstance(raw, dict):
            problems.add(line, where, f"expected an object, got {_type_name(raw)}")
            continue
        unknown = sorted(set(raw) - {"page", "quote"})
        if unknown:
            problems.add(line, where, f"unknown key(s) {unknown}")
            continue
        page, quote = raw.get("page"), raw.get("quote")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            problems.add(line, f"{where}.page", "expected a 1-based integer page number")
            continue
        if not isinstance(quote, str) or not quote.strip():
            problems.add(line, f"{where}.quote", "expected a non-empty string")
            continue
        entries.append(Evidence(page=page, quote=quote))
    return tuple(entries)


def _type_name(value: object) -> str:
    """The JSON-ish name of a value's type, for error messages."""
    return {
        type(None): "null",
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }.get(type(value), type(value).__name__)
