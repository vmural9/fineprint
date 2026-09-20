"""Check every golden-set evidence quote against the text pypdf extracts from the handbook.

Run it with ``uv run python -m evals.verify_golden_set``. Exit codes: 0 all checks passed,
1 at least one check failed, 2 the handbook PDF is missing.
"""

import argparse
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from pypdf import PdfReader

from evals.golden import DEFAULT_GOLDEN_SET, GoldenItem, GoldenSetError, load_golden_set

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = REPO_ROOT / "data" / "raw" / "medicare-and-you-2026.pdf"
DOWNLOAD_HINT = "run: uv run python scripts/download_handbook.py"

_WHITESPACE = re.compile(r"\s+")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NO_PDF = 2


def normalize(text: str) -> str:
    """Collapse every run of whitespace to one space so quotes survive line wrapping."""
    return _WHITESPACE.sub(" ", text).strip()


def read_pages(pdf_path: Path, pages: set[int]) -> tuple[dict[int, str], int]:
    """Extract and normalize the given 1-based pages. Returns the texts and the page count."""
    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)
    texts = {
        number: normalize(reader.pages[number - 1].extract_text() or "")
        for number in sorted(pages)
        if 1 <= number <= page_count
    }
    return texts, page_count


def format_alt_pages(alt_pages: Mapping[int, tuple[int, ...]]) -> str:
    """Render the alternate-page map the way the file writes it, as ``{30: [23]}``."""
    inside = ", ".join(f"{page}: {list(alts)}" for page, alts in sorted(alt_pages.items()))
    return f"{{{inside}}}"


def check_item(item: GoldenItem, texts: dict[int, str], page_count: int) -> list[str]:
    """Return one message per problem found with this item; empty means it passed.

    Every page the item names must exist in the PDF: expected pages, the alternate pages
    that state the same fact, and the pages the evidence quotes come from. Evidence may
    sit on an alternate page (or, for an unanswerable question, on the page that sends the
    reader elsewhere); every quote is checked against its own page the same way.
    """
    problems = []
    for page in item.cited_pages:
        if not 1 <= page <= page_count:
            problems.append(f"page {page} is outside the PDF (1-{page_count})")
    for entry in item.evidence:
        text = texts.get(entry.page)
        if text is None:
            continue
        quote = normalize(entry.quote)
        if quote not in text:
            problems.append(f'page {entry.page}: quote not found: "{quote}"')
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify the golden set and print one line per item plus a summary."""
    parser = argparse.ArgumentParser(
        description="Check every golden-set evidence quote against the handbook PDF."
    )
    parser.add_argument(
        "--pdf", type=Path, default=DEFAULT_PDF, help=f"handbook PDF (default: {DEFAULT_PDF})"
    )
    args = parser.parse_args(argv)

    try:
        items = load_golden_set(DEFAULT_GOLDEN_SET)
    except GoldenSetError as exc:
        print(f"ERROR: {DEFAULT_GOLDEN_SET} failed validation:\n{exc}", file=sys.stderr)
        return EXIT_FAILED

    if not args.pdf.is_file():
        print(f"ERROR: handbook PDF not found at {args.pdf}; {DOWNLOAD_HINT}", file=sys.stderr)
        return EXIT_NO_PDF

    wanted = {page for item in items for page in item.cited_pages}
    texts, page_count = read_pages(args.pdf, wanted)

    failed = 0
    quotes = 0
    for item in items:
        problems = check_item(item, texts, page_count)
        quotes += len(item.evidence)
        status = "FAIL" if problems else "ok  "
        flag = " needs_review" if item.needs_review else ""
        print(
            f"{item.id}  {status}  {item.category}/{item.type}/{item.difficulty}"
            f"  expected={list(item.expected_pages)} alt={format_alt_pages(item.alt_pages)}"
            f" quotes={len(item.evidence)}{flag}"
        )
        for problem in problems:
            print(f"        {problem}")
        failed += bool(problems)

    print(
        f"\n{len(items)} items, {len(items) - failed} ok, {failed} failed; "
        f"{quotes} quotes checked against {args.pdf.name} ({page_count} pages)"
    )
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
