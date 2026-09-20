#!/usr/bin/env python3
"""Download the "Medicare & You" handbook PDF and verify it against a pinned SHA-256.

The corpus for this project is a specific revision of a specific edition, so the
download is pinned by hash rather than by URL. medicare.gov publishes the handbook
at one unversioned URL that is replaced every September, which means the live URL
serves whatever edition is current -- not the one we index. Pinned bytes are
therefore fetched from the Internet Archive, which preserves the exact payload CMS
served while that edition was live.

Run from anywhere:

    python scripts/download_handbook.py                # 2026 edition (default)
    python scripts/download_handbook.py --edition 2025 # prior edition, for part 4
    python scripts/download_handbook.py --force        # re-download even if valid
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fineprint.editions import (
    CHUNK_BYTES,
    DEFAULT_EDITION,
    EDITIONS,
    RAW_DIR,
    Edition,
    pdf_path,
    sha256_of,
)

# A browser-ish User-Agent: some CDNs in front of .gov hosts reject the urllib default.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) fineprint-handbook-downloader/1.0"

# The pinned editions, the hashing, and `data/raw` all come from `fineprint.editions`, so the
# script and the service can never disagree about which bytes the corpus is. This script runs
# from anywhere, so it resolves that relative directory against the repo root itself.
REPO_ROOT = Path(__file__).resolve().parent.parent


def download_to(url: str, destination: Path, timeout: float) -> None:
    """Stream `url` into `destination`, overwriting whatever is there."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310
        destination.open("wb") as handle,
    ):
        while chunk := response.read(CHUNK_BYTES):
            handle.write(chunk)


def fetch_edition(edition: Edition, target: Path, timeout: float) -> str:
    """Download `edition` to `target`, trying each URL until the hash matches.

    Downloads land in a sibling `.part` file that is only renamed into place once
    the hash is right, so a failed run never leaves a partial or wrong PDF behind.
    Returns the URL that worked; raises RuntimeError if none did.
    """
    partial = target.with_suffix(target.suffix + ".part")
    problems: list[str] = []

    try:
        for url in edition.urls:
            try:
                download_to(url, partial, timeout)
            except (urllib.error.URLError, OSError) as error:
                problems.append(f"{url}\n    download failed: {error}")
                continue

            actual = sha256_of(partial)
            if actual == edition.sha256:
                partial.replace(target)
                return url

            problems.append(
                f"{url}\n    sha256 mismatch: got {actual} ({partial.stat().st_size} bytes), "
                f"expected {edition.sha256} ({edition.size_bytes} bytes)"
            )
    finally:
        partial.unlink(missing_ok=True)

    raise RuntimeError(
        f"could not obtain a verified copy of the {edition.year} edition:\n  "
        + "\n  ".join(problems)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--edition",
        type=int,
        default=DEFAULT_EDITION,
        choices=sorted(EDITIONS),
        help=f"handbook edition year (default: {DEFAULT_EDITION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a correct copy is already present",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request timeout in seconds (default: 120)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    edition = EDITIONS[args.edition]
    raw_dir = REPO_ROOT / RAW_DIR
    target = pdf_path(edition, raw_dir)

    if target.exists() and not args.force:
        if sha256_of(target) == edition.sha256:
            print(f"OK: {target} already present and verified (sha256 {edition.sha256})")
            return 0
        print(f"note: {target} exists but its hash is wrong; re-downloading", file=sys.stderr)

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        url = fetch_edition(edition, target, args.timeout)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"OK: downloaded Medicare & You {edition.year} ({edition.revision}), "
        f"{edition.pages} pages, {edition.size_bytes:,} bytes -> {target}\n"
        f"    sha256 {edition.sha256}\n"
        f"    from   {url}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
