"""Cutting the handbook into the passages retrieval actually searches.

A chunk is a short passage of the handbook that gets embedded, stored, and handed to the model
as evidence. Whole pages would be too coarse: a page mixes several topics, so its one vector is
an average of all of them and matches no question well. The whole book would be worse still.
Cut too small and a passage loses the sentence that gave it meaning.

Every chunk carries the pages its first and last word came from, so an answer can cite
"page 23" and a reader can go and check it.

The chunkers are looked up by name in `CHUNKERS`, so several chunkings of the same handbook can
live in the database side by side:

- `fixed-220w`, part 1's naive baseline, cuts windows of 220 words that overlap by 40, so a
  fact that straddles a boundary survives whole in at least one of them. It knows nothing of
  the handbook's structure.
- `sections` cuts where the handbook itself changes subject, at its headings, and keeps every
  table whole. The rest of this docstring is its rules.

How the sections chunker reads the handbook
-------------------------------------------

pypdf gives one line of text per printed line, and ends a line with a space when the paragraph
or table cell it belongs to carries on below. The chunker leaves out the page furniture, then
reads what is left as titles, headings, tables and paragraphs.

Page furniture: the page number alone in one of a page's first three lines, sometimes printed
twice ("123123"); the running title at the top of a page ("Section 1: Signing up for Medicare",
sometimes with the page number stuck to its end); any line found in the first three lines of
three or more pages, wherever it appears, like "Note: Go to pages 119–122 for definitions of
blue words." that opens every Section; and the labels of the margin icons, "Preventive service"
and "New!".

Titles: the contents page lists each part of the book with the page it starts on
("Section 1: Signing up for Medicare .....15"). Where a title is printed on the page it names,
often over two or three lines, it is a heading, and a "Section N" title heads the label of
every section up to the next one.

Headings: a line is a heading when

- it is at most 65 characters long, or 75 if it is a question like page 83's "What’s the
  Medicare drug coverage (Part D) late enrollment penalty?", and at most 55 straight under
  another heading, where a short first line of a paragraph would otherwise pass for one; it
  starts with a capital letter (after an opening quotation mark, if it has one) and has a
  lowercase letter in it;
- it does not end the way a sentence or a fragment does, in a full stop (inside closing
  quotation marks too), comma, colon, semicolon, exclamation mark, ampersand, slash, dash or
  digit, and it holds one phrase: no full stop, exclamation mark or colon inside it starts
  another, unless it is a question;
- it carries no web address and does not start with "Go to", like the index's cross-references;
- it follows a finished sentence or a web address printed on its own line, as page 32's
  "Bariatric surgery" follows "Medicare.gov/procedure-price-lookup", or it follows the top of a
  page, or another heading, title or table;
- the line after it neither starts in lowercase or with a bracket, nor finishes a sentence on a
  short line, any of which would make it the start of a sentence that carries on, nor ends in a
  page number, which would make it the head of a list of index entries.

A heading runs on to the next line while it ends on a small joining word ("with", "your") or
leaves a bracket or a quotation open, or when the next line opens a bracket or closes a
question, so "How do other insurance and programs work with" over "Medicare drug coverage
(Part D)?" is one heading. Headings that follow each other straight away are one heading. Words
printed three or more times in one Section, as a heading or not, are a question asked of every
entry in a list, like the "Do I need to choose a primary care doctor?" that each plan type on
pages 66–70 answers, so they stay in the text of their entry.

    Accepted: "How much does Part B coverage cost?" (page 23)
              "Durable medical equipment (DME)" (page 40)
    Rejected: "RRB Medicare Premium Payments" (page 23), an address that follows "mail your
              premium payments to:" rather than a finished sentence
              "Mental health care 46" (page 6), an index entry that ends in a page number

Tables: pypdf prints a table one cell at a time, so its lines are as narrow as its columns and
most of them wrap. A table is a run of ten or more lines that are narrow, at most 55 characters
where body text runs to about 75, or rows of dollar amounts and percentages, in which at least
half the lines are cells that wrap or such rows. A line that only finishes a wide line of prose
belongs to the prose, a heading after the last row belongs to the text below, and footnotes
marked with asterisks straight under a table on the same page stay with it. A page break does
not end a table: the comparison chart on pages 11–12 and the enrollment periods on pages 71–72
run on. A title from the contents page does, which is how the diagram at the foot of page 10
stays apart from the chart that opens page 11.

    Accepted: "If you’re 65 or older, have group health plan" (page 21), a cell of the chart of
              who pays first
              "Part A deductible 100% 100% 100% 100% 100% 50% 75% 50% 100%" (page 76), a row of
              the Medigap chart: wider than a cell, but made of percentages
    Rejected: "The standard Part B premium amount in 2026 is $202.90. Most people pay the"
              (page 23), body text: wide, and one amount does not make a row
              "Chemotherapy 34, 65" (page 4), narrow, but the index gives each entry a line of
              its own instead of wrapping cells, so its runs are never half cells

Chunks: every heading opens a section, and every chunk of the section is labelled with the
heading under its Section title: "Section 1: Signing up for Medicare > How much does Part B
coverage cost?". Text before the first heading is labelled "Front matter", so no chunk is ever
without a section. Within a section, paragraphs (lines that pypdf joins) are packed whole into
chunks of at most `max_words` words; only a paragraph longer than that is split, between
sentences, and a sentence longer than that is cut. The heading rides at the head of its
section's first chunk and does not count toward `max_words`. A table is always a chunk of its
own, however long, and is never merged with prose. A heading with nothing under it, like a
Section title directly above its first question, makes no chunk: its words live on in the
labels.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from fineprint.pdf import Page

DEFAULT_WORDS = 220
DEFAULT_OVERLAP = 40


@dataclass(frozen=True)
class Chunk:
    """One passage of the handbook, with the pages it came from.

    `ordinal` is the position in the document, counted from zero, and `page_start` and
    `page_end` are the pages of the first and the last word, so they are equal for a chunk that
    sits inside one page. `section` is the heading the chunk sits under: the sections chunker
    always fills it, and the fixed-size chunker, which knows no headings, leaves it empty.
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


# --- the sections chunker ---------------------------------------------------------------
#
# The rules, and the handbook lines they were written for, are in the module docstring; the
# constants below are the numbers and patterns those rules name.

# How many words of prose the sections chunker packs into a chunk, not counting its heading.
SECTION_WORDS = 300

# The label of the text that comes before the first heading.
NO_HEADING = "Front matter"
# What joins a Section title to a heading in a chunk's label.
SECTION_SEPARATOR = " > "

# Page furniture sits in a page's first lines.
HEADER_LINES = 3
# A line among the first `HEADER_LINES` of this many pages is furniture wherever it appears.
FURNITURE_PAGES = 3
RUNNING_TITLE = re.compile(r"^Section \d+: \S")
ICON_LABELS = frozenset({"Preventive service", "New!"})

# "Section 1: Signing up for Medicare ..............15" on the contents page.
CONTENTS_ENTRY = re.compile(r"^(?P<title>\S.*?) ?\.{5,} ?(?P<page>\d+)$")
SECTION_TITLE = re.compile(r"^Section \d+: ")

# The most lines a title or a heading is printed over.
MAX_HEADING_LINES = 3
MAX_HEADING_CHARACTERS = 65
# A question may run longer: "What’s the Medicare drug coverage (Part D) late enrollment
# penalty?" on page 83 is 67 characters.
MAX_QUESTION_CHARACTERS = 75
# Straight under another heading, where every paragraph starts, a heading is shorter still.
MAX_SUBHEADING_CHARACTERS = 55
NOT_A_HEADING_ENDING = re.compile(r"[.,;:!&/\-–—\d]$")
SECOND_PHRASE = re.compile(r"[.!:] [A-Z]")
WEB_ADDRESS = re.compile(r"\.(gov|org|com)\b")
# A line that is nothing but a web address, "Medicare.gov/procedure-price-lookup".
ONLY_A_WEB_ADDRESS = re.compile(r"^[\w.-]+\.(gov|org|com|mil)(/\S*)?$")
ENDS_IN_PAGE_NUMBER = re.compile(r"\d,?$")
OPENING_QUOTES = "“‘\"'"
CLOSING_QUOTES = "”’\"'"
# A heading does not end on one of these; when a line of it does, the heading carries on below.
JOINING_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to"}
    | {"with", "your"}
)
# Words printed this many times in one Section, as a heading or not, are a question asked of
# every entry of a list, and stay in the text.
REPEATED_HEADING = 3

NARROW_LINE_CHARACTERS = 55
MIN_TABLE_LINES = 10
AMOUNT = re.compile(r"^(\$[\d,]+(\.\d+)?|\d+(\.\d+)?%\**)$")
FOOTNOTE_MARK = "*"

# A word that ends a sentence, and a word that can start the next one.
ENDS_SENTENCE = re.compile(r"[.?!][)”’\"']*$")
STARTS_SENTENCE = re.compile(r"^[A-Z0-9•“\"(]")

# A word of the handbook and the page it is printed on.
Word = tuple[str, int]


@dataclass(frozen=True)
class Line:
    """One printed line of the handbook, with the page it is on.

    `text` has its runs of whitespace collapsed. `continues` says that the paragraph or table
    cell goes on below: pypdf ended the line with a space, or the line breaks a word at a hyphen.
    """

    page: int
    text: str
    continues: bool


@dataclass(frozen=True)
class Block:
    """A heading, a paragraph or a table, as the sections chunker reads the handbook.

    `part` is the title of the Section the block sits in, "Section 1: Signing up for Medicare",
    or None before the first Section.
    """

    kind: Literal["heading", "paragraph", "table"]
    lines: tuple[Line, ...]
    part: str | None

    @property
    def text(self) -> str:
        """The block's lines joined by single spaces."""
        return joined(self.lines)

    @property
    def words(self) -> list[Word]:
        """Every word of the block, each with the page it is printed on."""
        return [(word, line.page) for line in self.lines for word in line.text.split()]


def collapse(text: str) -> str:
    """`text` with every run of whitespace turned into one space, and the ends trimmed."""
    return " ".join(text.split())


def joined(lines: tuple[Line, ...] | list[Line]) -> str:
    """The text of several lines, as one line."""
    return " ".join(line.text for line in lines)


def contents_titles(pages: list[Page]) -> dict[str, int]:
    """Each title on the contents page, with the page its part of the book starts on."""
    titles: dict[str, int] = {}
    for page in pages:
        for line in page.text.split("\n"):
            entry = CONTENTS_ENTRY.match(collapse(line))
            if entry is not None:
                titles[entry["title"]] = int(entry["page"])
    return titles


def furniture_lines(pages: list[Page]) -> set[str]:
    """The lines that open `FURNITURE_PAGES` pages or more: running headers, whatever they say."""
    openings: Counter[str] = Counter()
    for page in pages:
        opening = {collapse(line) for line in page.text.split("\n")[:HEADER_LINES]}
        openings.update(opening - {""})
    return {text for text, pages_opened in openings.items() if pages_opened >= FURNITURE_PAGES}


def is_furniture(
    text: str, position: int, page: int, furniture: set[str], titles: dict[str, int]
) -> bool:
    """Whether a line is page furniture rather than text; `position` counts from the top."""
    if titles.get(text) == page or CONTENTS_ENTRY.match(text) is not None:
        # A part's title on the page it opens, even when it also heads the pages after it, and
        # the entries of the contents page, which can look like running titles, are text.
        return False
    if text in furniture or text in ICON_LABELS:
        return True
    near_the_top = position < HEADER_LINES
    page_number = text in (str(page), str(page) * 2)
    return near_the_top and (page_number or RUNNING_TITLE.match(text) is not None)


def read_lines(pages: list[Page], titles: dict[str, int]) -> list[Line]:
    """Every line of text in reading order, with the page furniture left out."""
    furniture = furniture_lines(pages)
    lines: list[Line] = []
    for page in pages:
        for position, raw in enumerate(page.text.split("\n")):
            text = collapse(raw)
            if text and not is_furniture(text, position, page.number, furniture, titles):
                continues = raw.endswith(" ") or text.endswith("-")
                lines.append(Line(page=page.number, text=text, continues=continues))
    return lines


def find_titles(lines: list[Line], titles: dict[str, int]) -> dict[int, int]:
    """Where a title from the contents page is printed on the page it names.

    Returns {index of the title's first line: index of the line after it}. A title may be
    printed over up to `MAX_HEADING_LINES` lines, like "Section 1:" over "Signing up for
    Medicare".
    """
    found: dict[int, int] = {}
    index = 0
    while index < len(lines):
        page = lines[index].page
        for length in range(1, MAX_HEADING_LINES + 1):
            span = lines[index : index + length]
            if len(span) == length and span[-1].page == page and titles.get(joined(span)) == page:
                found[index] = index + length
                break
        index = found.get(index, index + 1)
    return found


def heading_room(text: str) -> int:
    """How many characters a line of a heading may take: a question may take more."""
    return MAX_QUESTION_CHARACTERS if text.endswith("?") else MAX_HEADING_CHARACTERS


def looks_like_heading(text: str) -> bool:
    """Whether a line has the shape of a heading, or of a heading's first line."""
    return (
        len(text) <= heading_room(text)
        and text.lstrip(OPENING_QUOTES)[:1].isupper()
        and any(character.islower() for character in text)
        and NOT_A_HEADING_ENDING.search(text.rstrip(CLOSING_QUOTES)) is None
        and not text.startswith("Go to ")
    )


def closes_a_passage(text: str) -> bool:
    """Whether a line ends what comes before it: a finished sentence, or a web address the
    handbook prints on a line of its own at the end of a passage."""
    return ENDS_SENTENCE.search(text) is not None or ONLY_A_WEB_ADDRESS.match(text) is not None


def is_whole_heading(heading: str) -> bool:
    """Whether the lines taken for a heading make a whole one, and only one."""
    return (
        heading.split()[-1].lower() not in JOINING_WORDS
        and heading.count("(") <= heading.count(")")
        and heading.count("“") <= heading.count("”")
        and (heading.endswith("?") or SECOND_PHRASE.search(heading) is None)
        and WEB_ADDRESS.search(heading) is None
    )


def is_index_entry(text: str) -> bool:
    """Whether a line is an entry of the index: short, and ending in a page number.

    The contents page's lines end in page numbers too, but their dotted leaders set them apart.
    """
    return (
        len(text) <= NARROW_LINE_CHARACTERS
        and ENDS_IN_PAGE_NUMBER.search(text) is not None
        and CONTENTS_ENTRY.match(text) is None
    )


def runs_on(lines: list[Line], start: int, end: int) -> bool:
    """Whether the heading printed on `lines[start:end]` so far carries on to `lines[end]`."""
    if end >= len(lines) or lines[end].page != lines[start].page:
        return False
    heading, following = joined(lines[start:end]), lines[end].text
    if len(following) > heading_room(following):
        return False
    if NOT_A_HEADING_ENDING.search(following.rstrip(CLOSING_QUOTES)):
        return False
    return (
        heading.split()[-1].lower() in JOINING_WORDS
        or heading.count("(") > heading.count(")")
        or heading.count("“") > heading.count("”")
        or following.startswith("(")
        or following.endswith("?")
    )


def heading_length(lines: list[Line], index: int, previous_kind: str, taken: set[int]) -> int:
    """How many lines the heading that starts at `lines[index]` is printed over, or 0.

    `previous_kind` is what the line before belongs to: a "title", a "heading", a "table", a
    "line" of a paragraph, or "" at the start of the handbook. `taken` holds the lines of
    titles and tables, which a heading never runs into.
    """
    line = lines[index]
    if not looks_like_heading(line.text):
        return 0
    if previous_kind in ("title", "heading") and len(line.text) > MAX_SUBHEADING_CHARACTERS:
        return 0
    previous = lines[index - 1] if index > 0 else None
    at_a_break = (
        previous_kind != "line"
        or previous is None
        or previous.page != line.page
        or closes_a_passage(previous.text)
    )
    if not at_a_break:
        return 0

    end = index + 1
    while end - index < MAX_HEADING_LINES and end not in taken and runs_on(lines, index, end):
        end += 1
    if not is_whole_heading(joined(lines[index:end])):
        return 0
    following = lines[end].text if end < len(lines) else None
    if following is not None and (
        following[0].islower()
        or following.startswith("(")
        or (len(following) <= NARROW_LINE_CHARACTERS and ENDS_SENTENCE.search(following))
        or is_index_entry(following)
    ):
        return 0  # the first line of a sentence that carries on, or of a list of index entries
    return end - index


def is_row_of_amounts(text: str) -> bool:
    """A row like "100% 100% 50% 75%": two amounts or more, and half its words amounts."""
    words = text.split()
    amounts = sum(1 for word in words if AMOUNT.match(word))
    return amounts >= 2 and 2 * amounts >= len(words)


def is_narrow(line: Line) -> bool:
    """Whether a line could belong to a table: as narrow as a cell, or a row of amounts."""
    return len(line.text) <= NARROW_LINE_CHARACTERS or is_row_of_amounts(line.text)


def is_table(lines: list[Line]) -> bool:
    """Ten narrow lines or more, of which at least half are cells that wrap or rows of amounts."""
    cells = sum(1 for line in lines if line.continues or is_row_of_amounts(line.text))
    return len(lines) >= MIN_TABLE_LINES and 2 * cells >= len(lines)


def with_footnotes(lines: list[Line], end: int) -> int:
    """Where a table that stops before `lines[end]` ends, once its footnotes are counted in."""
    page = lines[end - 1].page
    while (
        end < len(lines) and lines[end].page == page and lines[end].text.startswith(FOOTNOTE_MARK)
    ):
        end += 1
        while (
            end < len(lines)
            and lines[end].page == page
            and lines[end - 1].continues
            and not lines[end].text.startswith(FOOTNOTE_MARK)
        ):
            end += 1
    return end


def find_tables(lines: list[Line], titles_at: dict[int, int]) -> dict[int, int]:
    """Where the tables are: {index of a table's first line: index of the line after it}."""
    in_a_title = {index for start, end in titles_at.items() for index in range(start, end)}
    tables: dict[int, int] = {}
    index = 0
    while index < len(lines):
        if index in in_a_title or not is_narrow(lines[index]):
            index += 1
            continue
        run_end = index
        while run_end < len(lines) and run_end not in in_a_title and is_narrow(lines[run_end]):
            run_end += 1

        start, end = index, run_end
        # A line that only finishes a wide line of prose belongs to the prose.
        while (
            start < end
            and start > 0
            and lines[start - 1].continues
            and not is_narrow(lines[start - 1])
        ):
            start += 1
        # A heading after the last row belongs to the text below it.
        while end > start and heading_length(lines, end - 1, "line", set()):
            end -= 1
        if is_table(lines[start:end]):
            tables[start] = with_footnotes(lines, end)
        index = run_end
    return tables


def split_blocks(pages: list[Page]) -> list[Block]:
    """The handbook as headings, paragraphs and tables, in reading order.

    Page furniture is gone, and every heading has something under it: a heading followed
    straight away by a title from the contents page, or by nothing at all, is left out.
    """
    titles = contents_titles(pages)
    lines = read_lines(pages, titles)
    titles_at = find_titles(lines, titles)
    tables_at = find_tables(lines, titles_at)
    taken = {
        index
        for spans in (titles_at, tables_at)
        for start, end in spans.items()
        for index in range(start, end)
    }

    # Each span of lines is a "title", a "heading", a "table", or a "line" of a paragraph.
    spans: list[tuple[int, int, str]] = []
    index, previous_kind = 0, ""
    while index < len(lines):
        if index in titles_at:
            end, kind = titles_at[index], "title"
        elif index in tables_at:
            end, kind = tables_at[index], "table"
        elif length := heading_length(lines, index, previous_kind, taken):
            end, kind = index + length, "heading"
        else:
            end, kind = index + 1, "line"
        spans.append((index, end, kind))
        index, previous_kind = end, kind
    return assemble_blocks(lines, spans)


def assemble_blocks(lines: list[Line], spans: list[tuple[int, int, str]]) -> list[Block]:
    """Turn the spans `split_blocks` found into blocks, each knowing its Section."""
    parts: list[str | None] = []
    part: str | None = None
    for start, end, kind in spans:
        if kind == "title":
            title = joined(lines[start:end])
            part = title if SECTION_TITLE.match(title) else None
        parts.append(part)
    # How often each heading's words are printed in its Section, as a heading or not.
    printings = Counter(
        (part, joined(lines[start:end]))
        for (start, end, kind), part in zip(spans, parts, strict=True)
        if kind in ("heading", "line")
    )

    blocks: list[Block] = []
    previous_kind = None
    for (start, end, kind), part in zip(spans, parts, strict=True):
        span = tuple(lines[start:end])
        if kind == "heading" and printings[(part, joined(span))] >= REPEATED_HEADING:
            kind = "line"  # a question asked of every entry of a list stays in the entry
        if kind in ("title", "heading"):
            if blocks and blocks[-1].kind == "heading":
                if kind == "heading" and previous_kind == "heading":
                    blocks[-1] = replace(blocks[-1], lines=blocks[-1].lines + span)
                    previous_kind = kind
                    continue
                blocks.pop()  # a heading with nothing under it
            blocks.append(Block(kind="heading", lines=span, part=part))
        elif kind == "table":
            blocks.append(Block(kind="table", lines=span, part=part))
        elif blocks and blocks[-1].kind == "paragraph" and blocks[-1].lines[-1].continues:
            blocks[-1] = replace(blocks[-1], lines=blocks[-1].lines + span)
        else:
            blocks.append(Block(kind="paragraph", lines=span, part=part))
        previous_kind = kind
    if blocks and blocks[-1].kind == "heading":
        blocks.pop()
    return blocks


def section_label(heading: Block) -> str:
    """A heading under its Section title, "Section 1: … > How much does Part B coverage cost?"."""
    if heading.part is None or heading.text == heading.part:
        return heading.text
    return f"{heading.part}{SECTION_SEPARATOR}{heading.text}"


def sentences(words: list[Word]) -> list[list[Word]]:
    """A paragraph's words, split after every word that ends a sentence."""
    split: list[list[Word]] = [[]]
    for position, word in enumerate(words):
        split[-1].append(word)
        following = words[position + 1][0] if position + 1 < len(words) else ""
        if ENDS_SENTENCE.search(word[0]) and STARTS_SENTENCE.match(following):
            split.append([])
    return [sentence for sentence in split if sentence]


def pieces(words: list[Word], max_words: int) -> list[list[Word]]:
    """A paragraph whole if it fits in a chunk, and otherwise its sentences, each cut to fit."""
    if len(words) <= max_words:
        return [words]
    return [
        sentence[start : start + max_words]
        for sentence in sentences(words)
        for start in range(0, len(sentence), max_words)
    ]


def new_chunk(ordinal: int, words: list[Word], section: str) -> Chunk:
    """A chunk of `words`, with the pages of its first and last word."""
    return Chunk(
        ordinal=ordinal,
        page_start=words[0][1],
        page_end=words[-1][1],
        section=section,
        text=" ".join(word for word, _ in words),
    )


def section_chunks(pages: list[Page], max_words: int = SECTION_WORDS) -> list[Chunk]:
    """Cut `pages` where the handbook changes subject, keeping every table whole.

    The rules are in the module docstring. `max_words` bounds the prose in a chunk, not
    counting the heading at its head; a table is one chunk however long it is.
    """
    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words}")

    chunks: list[Chunk] = []
    section = NO_HEADING
    heading: list[Word] = []  # waits for the first words under it
    body: list[Word] = []
    for block in split_blocks(pages):
        if block.kind == "heading":
            if body:
                chunks.append(new_chunk(len(chunks), heading + body, section))
            section, heading, body = section_label(block), block.words, []
        elif block.kind == "table":
            if body:
                chunks.append(new_chunk(len(chunks), heading + body, section))
                heading = []
            chunks.append(new_chunk(len(chunks), heading + block.words, section))
            heading, body = [], []
        else:
            for piece in pieces(block.words, max_words):
                if body and len(body) + len(piece) > max_words:
                    chunks.append(new_chunk(len(chunks), heading + body, section))
                    heading, body = [], []
                body = body + piece
    if body:
        chunks.append(new_chunk(len(chunks), heading + body, section))
    return chunks


# Chunkers by chunk-set name. The name is what `CHUNK_SET` holds and what the `chunks.chunk_set`
# column stores, so several chunkings can live in the database side by side. A name must keep
# describing what its chunker does with its defaults: "fixed-220w" is 220-word windows, and
# "sections" cuts at the handbook's headings with at most `SECTION_WORDS` words of prose.
CHUNKERS: dict[str, Callable[[list[Page]], list[Chunk]]] = {
    "fixed-220w": fixed_size_chunks,
    "sections": section_chunks,
}
