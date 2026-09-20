"""Tests for the pinned edition table.

The point of the table is that the corpus is one exact file, so these tests are mostly about
there being one copy of that fact: the download script and the service read the same rows.
"""

from pathlib import Path

from fineprint.editions import EDITIONS, pdf_path, sha256_of


def test_the_download_script_and_the_package_share_one_editions_table():
    """The pinned hash is written down once: the script imports the package's table."""
    from scripts.download_handbook import EDITIONS as SCRIPT_EDITIONS

    assert SCRIPT_EDITIONS is EDITIONS


def test_an_edition_knows_its_file_name_its_title_and_its_primary_url():
    edition = EDITIONS[2026]

    assert edition.filename == "medicare-and-you-2026.pdf"
    assert edition.title == "Medicare & You 2026"
    assert edition.primary_url == edition.urls[0]


def test_an_edition_lives_under_data_raw():
    assert pdf_path(EDITIONS[2026]) == Path("data/raw/medicare-and-you-2026.pdf")


def test_the_hash_of_a_file_is_the_hash_of_its_bytes(tmp_path: Path):
    """The empty file's SHA-256 is a published constant, so this also checks the chunked read."""
    path = tmp_path / "empty"
    path.write_bytes(b"")

    assert sha256_of(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
