"""Cutting the handbook into the passages retrieval actually searches.

A chunk is a short passage of the handbook — here a window of 220 words — that gets embedded,
stored, and handed to the model as evidence. Whole pages would be too coarse: a page mixes
several topics, so its one vector is an average of all of them and matches no question well.
The whole book would be worse still. Cut too small and a passage loses the sentence that gave
it meaning, which is why consecutive windows overlap: a fact that straddles a boundary survives
whole in at least one of them.

Every chunk carries the pages its first and last word came from, so an answer can cite
"page 23" and a reader can go and check it.

Part 1 chunks by word count and nothing else: the naive baseline. Part 2 adds chunkers that
follow the handbook's structure, which is why they are looked up by name in `CHUNKERS`.
"""

from collections.abc import Callable
from dataclasses import dataclass

from fineprint.pdf import Page

DEFAULT_WORDS = 220
DEFAULT_OVERLAP = 40


@dataclass(frozen=True)
class Chunk:
    """One passage of the handbook, with the pages it came from.

    `ordinal` is the position in the document, counted from zero, and `page_start` and
    `page_end` are the pages of the first and the last word, so they are equal for a chunk that
    sits inside one page. `section` is the seam for part 2's structure-aware chunker; part 1
    leaves it empty.
    """

    ordinal: int
    page_start: int
    page_end: int
    section: str | None
    text: str


def word_stream(pages: list[Page]) -> list[tuple[str, int]]:
    """Every word of the document in reading order, each remembering its page number.

    Pages with no text, such as the handbook's blank page 127, simply contribute no words.
    """
    return [(word, page.number) for page in pages for word in page.text.split()]


def fixed_size_chunks(
    pages: list[Page], words: int = DEFAULT_WORDS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Cut `pages` into windows of `words` words that overlap by `overlap` words.

    The document is one stream of words in which each word remembers its page, so a window may
    begin on one page and end on the next. Each window starts `words - overlap` words after the
    previous one, which makes the shared tail exactly `overlap` words long; only the last window
    may be short. A document that does not fill one window gives a single chunk, and a document
    with no words at all gives none.
    """
    if words < 1:
        raise ValueError(f"words must be at least 1, got {words}")
    if not 0 <= overlap < words:
        # A window that does not advance would repeat itself for ever.
        raise ValueError(f"overlap must be at least 0 and less than words ({words}), got {overlap}")

    stream = word_stream(pages)
    step = words - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(stream):
        window = stream[start : start + words]
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                page_start=window[0][1],
                page_end=window[-1][1],
                section=None,
                text=" ".join(word for word, _ in window),
            )
        )
        if start + words >= len(stream):
            # This window reached the end of the document; another one would only repeat it.
            break
        start += step
    return chunks


# Chunkers by chunk-set name. The name is what `CHUNK_SET` holds and what the `chunks.chunk_set`
# column stores, so several chunkings can live in the database side by side. "fixed-220w" must
# keep describing this function's defaults; part 2 adds its chunker as another entry here, and
# no caller changes.
CHUNKERS: dict[str, Callable[[list[Page]], list[Chunk]]] = {
    "fixed-220w": fixed_size_chunks,
}
