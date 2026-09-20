"""The pinned editions of the handbook, and how to check a file against one.

The corpus is not "the latest Medicare & You". It is one revision of one edition, pinned by its
SHA-256, because CMS revises the handbook mid-edition and replaces it at the same URL every
September: the September 2025 printing of the *2026* edition still prints 2025 dollar amounts.
An answer is only worth citing if it came from the bytes the expected answers were written
against.

`scripts/download_handbook.py` fetches those bytes and `fineprint ingest` refuses to read a file
whose hash is not the one recorded here. Both import this table, so the pinned hash is written
down exactly once.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Read in 1 MiB blocks so the hash is computed without holding the PDF in memory.
CHUNK_BYTES = 1024 * 1024

# Where the download script writes and the ingest command reads. Relative on purpose: the
# command resolves it against the working directory, and the script against the repo root.
RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class Edition:
    """One pinned edition of the handbook.

    `urls` are tried in order; every one of them must serve these exact bytes. `sha256` and
    `size_bytes` were verified by downloading the file and, for the Internet Archive URLs, by
    checking the payload against the SHA-1 digest the Wayback CDX index records for that capture.
    """

    year: int
    sha256: str
    size_bytes: int
    pages: int
    revision: str
    urls: tuple[str, ...]

    @property
    def filename(self) -> str:
        """The name the PDF is saved under: `medicare-and-you-2026.pdf`."""
        return f"medicare-and-you-{self.year}.pdf"

    @property
    def title(self) -> str:
        """What `documents.title` holds: `Medicare & You 2026`."""
        return f"Medicare & You {self.year}"

    @property
    def primary_url(self) -> str:
        """What `documents.source_url` holds: the first URL that serves these bytes."""
        return self.urls[0]


# Only editions whose hash was verified against a real download belong here.
EDITIONS: dict[int, Edition] = {
    2026: Edition(
        year=2026,
        sha256="d7a341bc3d2d3dab59af746a0875752761e6c2f2dc107d95e9078a23933513d6",
        size_bytes=4_064_150,
        pages=128,
        # Final 2026 revision: "CMS Product No. 10050, January 2026" on the back
        # cover, PDF /ModDate 2026-06-22. Served by CMS from 2026-06-30 until the
        # 2027 edition replaced it on 2026-09-08.
        revision="January 2026 printing (PDF ModDate 2026-06-22)",
        urls=(
            "https://web.archive.org/web/20260630081453id_/"
            "https://www.medicare.gov/publications/10050-medicare-and-you.pdf",
            "https://web.archive.org/web/20260908061836id_/"
            "https://www.medicare.gov/publications/10050-medicare-and-you.pdf",
        ),
    ),
    2025: Edition(
        year=2025,
        sha256="89ba6c75d91a2cb606fd53606366d1ae977d6e5c703335569814117dcce6add9",
        size_bytes=4_719_868,
        pages=128,
        revision="final 2025 revision (PDF ModDate 2025-02-14)",
        urls=(
            "https://web.archive.org/web/20250901101454id_/"
            "https://www.medicare.gov/publications/10050-medicare-and-you.pdf",
        ),
    ),
}

DEFAULT_EDITION = 2026


def pdf_path(edition: Edition, raw_dir: Path = RAW_DIR) -> Path:
    """Where this edition's PDF lives, under `raw_dir`."""
    return raw_dir / edition.filename


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
