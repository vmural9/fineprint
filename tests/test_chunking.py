"""Tests for the fixed-size chunker.

Every test builds its pages by hand, so the suite runs without the handbook, the database, or
AWS. Words are made distinct on purpose (`p5w07` is the eighth word of page 5) so a test can say
exactly which words a chunk holds and which page each of them came from.
"""

import pytest

from fineprint.chunking import CHUNKERS, Chunk, fixed_size_chunks
from fineprint.pdf import Page


def numbered_page(number: int, word_count: int) -> Page:
    """A page of distinct words that name their own page, like `p5w07`."""
    words = " ".join(f"p{number}w{index:02d}" for index in range(word_count))
    return Page(number=number, text=words)


def words_of(chunk: Chunk) -> list[str]:
    """The words a chunk holds, in order."""
    return chunk.text.split()


def test_a_document_shorter_than_one_window_gives_a_single_chunk():
    """Ten words do not fill a 220-word window, and a short document is still one chunk."""
    pages = [numbered_page(1, 10)]

    chunks = fixed_size_chunks(pages, words=220, overlap=40)

    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert chunks[0].text == pages[0].text


def test_ordinals_are_consecutive_from_zero():
    """The ordinal is the chunk's position in the document, and the database stores it."""
    pages = [numbered_page(number, 100) for number in range(1, 6)]

    chunks = fixed_size_chunks(pages, words=100, overlap=20)

    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_every_word_of_the_input_appears_in_some_chunk():
    """Nothing may fall between two windows: an answer can only cite text that was chunked."""
    pages = [numbered_page(number, 100) for number in range(1, 6)]
    stream = [word for page in pages for word in page.text.split()]

    chunks = fixed_size_chunks(pages, words=100, overlap=20)

    chunked = {word for chunk in chunks for word in words_of(chunk)}
    assert chunked == set(stream)


def test_the_chunks_put_the_document_back_together_in_order():
    """Dropping each window's overlap rebuilds the original stream, word for word."""
    pages = [numbered_page(number, 100) for number in range(1, 6)]
    stream = [word for page in pages for word in page.text.split()]

    chunks = fixed_size_chunks(pages, words=100, overlap=20)

    rebuilt = words_of(chunks[0]) + [word for chunk in chunks[1:] for word in words_of(chunk)[20:]]
    assert rebuilt == stream


def test_consecutive_chunks_share_exactly_the_overlap():
    """The tail of one window is the head of the next, and it is `overlap` words long."""
    pages = [numbered_page(number, 100) for number in range(1, 6)]
    overlap = 20

    chunks = fixed_size_chunks(pages, words=100, overlap=overlap)

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert words_of(earlier)[-overlap:] == words_of(later)[:overlap]
        # The words are all distinct, so the shared set is exactly the shared window.
        assert len(set(words_of(earlier)) & set(words_of(later))) == overlap


def test_windows_are_the_requested_size_apart_from_the_last_one():
    """Only the tail of the document is allowed to be short."""
    pages = [numbered_page(number, 100) for number in range(1, 6)]

    chunks = fixed_size_chunks(pages, words=100, overlap=20)

    assert all(len(words_of(chunk)) == 100 for chunk in chunks[:-1])
    assert 0 < len(words_of(chunks[-1])) <= 100


def test_a_chunk_that_crosses_a_page_boundary_records_both_pages():
    """`page_start` and `page_end` come from the first and the last word of the window."""
    pages = [numbered_page(5, 30), numbered_page(6, 30)]

    chunks = fixed_size_chunks(pages, words=40, overlap=10)

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(5, 6), (6, 6)]


def test_a_chunk_can_span_more_than_two_pages():
    """Short pages mean a window can swallow several of them; the range still holds the ends."""
    pages = [numbered_page(number, 5) for number in (9, 10, 11)]

    chunks = fixed_size_chunks(pages, words=12, overlap=4)

    assert (chunks[0].page_start, chunks[0].page_end) == (9, 11)


def test_a_page_with_no_words_leaves_no_gap():
    """Page 127 of the handbook is blank; it contributes nothing and breaks nothing."""
    pages = [numbered_page(1, 5), Page(number=2, text="  \n  "), numbered_page(3, 5)]

    chunks = fixed_size_chunks(pages, words=20, overlap=5)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 3)
    assert len(words_of(chunks[0])) == 10


def test_a_document_with_no_words_gives_no_chunks():
    """Nothing in, nothing out: no empty chunk is ever stored or embedded."""
    assert fixed_size_chunks([]) == []
    assert fixed_size_chunks([Page(number=127, text="")]) == []


def test_part_one_leaves_the_section_empty():
    """`section` is the seam for part 2's structure-aware chunker; part 1 never fills it."""
    pages = [numbered_page(number, 100) for number in range(1, 6)]

    chunks = fixed_size_chunks(pages, words=100, overlap=20)

    assert all(chunk.section is None for chunk in chunks)


@pytest.mark.parametrize(
    ("words", "overlap"),
    [(100, 100), (100, 120), (100, -1), (0, 0)],
    ids=["overlap equals words", "overlap exceeds words", "negative overlap", "empty window"],
)
def test_impossible_window_settings_are_refused(words: int, overlap: int):
    """A window that never advances would loop forever, so it is refused at the door."""
    with pytest.raises(ValueError):
        fixed_size_chunks([numbered_page(1, 10)], words=words, overlap=overlap)


def test_the_registry_holds_the_chunker_under_its_chunk_set_name():
    """`CHUNK_SET` names a chunker here, so part 2 adds one without touching any caller."""
    assert CHUNKERS["fixed-220w"] is fixed_size_chunks


def test_the_registry_name_describes_what_the_defaults_do():
    """`fixed-220w` must really mean 220-word windows, with the documented 40-word overlap."""
    pages = [numbered_page(number, 200) for number in range(1, 4)]

    chunks = CHUNKERS["fixed-220w"](pages)

    assert len(words_of(chunks[0])) == 220
    assert words_of(chunks[0])[-40:] == words_of(chunks[1])[:40]
