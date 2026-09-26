"""Reading the handbook: one `Page` per PDF page, numbered the way a reader numbers it.

Part 1 extracts the naive baseline on purpose. Whatever pypdf gives back is kept as it is:
running headers, folios, the table of contents, and the index all stay in the text. Part 2
measures what cleaning them up is worth, which is only possible with a baseline to compare to.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    """One page of the handbook, with the page number a citation can print.

    `number` is the 1-based PDF page index. In "Medicare & You 2026" that is also the number
    printed on the page for every numbered page, so nothing in this project translates between
    the two: printed page N is PDF page N for pages 2 to 126.
    """

    number: int
    text: str


def extract_pages(path: Path) -> list[Page]:
    """Extract the text of every page of the PDF at `path`, in order.

    Uses pypdf rather than pdfplumber: pdfplumber reads the handbook's two-column pages line by
    line across both columns, which scrambles the comparison tables and the index. A page whose
    text pypdf cannot recover, such as the blank page 127, comes back with an empty string
    rather than being dropped, so the page numbers stay aligned with the PDF.
    """
    reader = PdfReader(path)
    return [
        Page(number=number, text=page.extract_text() or "")
        for number, page in enumerate(reader.pages, start=1)
    ]
