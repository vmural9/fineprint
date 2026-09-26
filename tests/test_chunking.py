"""Tests for the two chunkers.

Every test builds its pages by hand, so the suite runs without the handbook, the database, or
AWS.

For the fixed-size chunker, words are made distinct on purpose (`p5w07` is the eighth word of
page 5) so a test can say exactly which words a chunk holds and which page each of them came
from.

For the sections chunker, pages are written the way pypdf extracts the handbook: one printed
line per line of text, with a space left at the end of a line whose paragraph or table cell
carries on below. Headings and tables are copied from the handbook with the page they are
printed on, so each test shows a rule at work on the text it was written for.
"""

import pytest

from fineprint.chunking import (
    CHUNKERS,
    NO_HEADING,
    Chunk,
    fixed_size_chunks,
    section_chunks,
)
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


# --- the sections chunker ---------------------------------------------------------------


def page(number: int, *lines: str) -> Page:
    """A page as pypdf extracts it: one printed line per line of text."""
    return Page(number=number, text="\n".join(lines))


def printed(text: str) -> list[str]:
    """`text` printed as one paragraph of body text, sixteen words to a line.

    Every line but the last ends in a space, the mark pypdf leaves where a paragraph carries
    on. The lines are as wide as the handbook's body text, so they never pass for table cells.
    """
    words = text.split()
    lines = [" ".join(words[start : start + 16]) for start in range(0, len(words), 16)]
    return [line + " " for line in lines[:-1]] + [lines[-1]]


def sentence(tag: str, length: int) -> str:
    """A sentence of `length` distinct words: `A000 a001 a002 ... a039.`"""
    words = [f"{tag}{number:03d}" for number in range(length)]
    return " ".join([words[0].capitalize(), *words[1:-1], f"{words[-1]}."])


def sections_of(chunks: list[Chunk]) -> list[str | None]:
    """The section label of every chunk, in order."""
    return [chunk.section for chunk in chunks]


# Page 23 as pypdf extracts it: the running title, the page number, then the page's first
# heading and paragraph.
PART_B_PREMIUM = page(
    23,
    "Section 1: Signing up for Medicare",
    "23",
    "How much does Part B coverage cost? ",
    "The standard Part B premium amount in 2026 is $202.90. Most people pay the ",
    "standard Part B premium amount every month.",
)

# Page 21: the chart of who pays first, with the paragraph above it and the note below it.
# The chart's rows about disability, ESRD and TRICARE are left out to keep the test short.
WHO_PAYS_INTRO = [
    "How does my other insurance work with Medicare?",
    "When you have other insurance (like group health plan, retiree health, or ",
    "Medicaid coverage) and Medicare, there are rules for whether Medicare or your ",
    "other coverage pays first. ",
]
WHO_PAYS_CHART = [
    "If you have retiree health coverage, like insurance ",
    "from your or your spouse’s former employment… ",
    "Medicare pays first. ",
    "If you’re 65 or older, have group health plan ",
    "coverage based on your or your spouse’s current ",
    "employment, and the employer has 20 or more ",
    "employees OR your employer has fewer than ",
    "20 employees but is part of a multi-employer ",
    "plan that has other employers with 20 or more ",
    "employees...",
    "Your group health plan pays ",
    "first. ",
    "If you’re 65 or older, have group health plan ",
    "coverage based on your or your spouse’s current ",
    "employment, and the employer has fewer than ",
    "20 employees... ",
    "Medicare pays first. ",
    "If you have Medicaid... Medicare pays first.",
]
WHO_PAYS_NOTE = [
    "Important! If you’re still working and have employer coverage through ",
    "work, contact your employer to find out how your employer’s coverage ",
    "works with Medicare.",
]

# Page 76: the paragraph above the Medigap chart, some of its rows, and its first footnote.
MEDIGAP_INTRO = [
    "How do I compare Medigap plans? ",
    "The chart below shows basic information about the different benefits covered ",
    "by Medicare Supplement Insurance (Medigap) in 2026. If a percentage ",
    "appears, the Medigap plan covers that percentage of the benefit, and you’re ",
    "responsible for the rest. ",
]
MEDIGAP_CHART = [
    "Medigap standardized plans",
    "Benefits A B C D F* G* K L M N",
    "Medicare Part B  ",
    "coinsurance or ",
    "copayment ",
    "100% 100% 100% 100% 100% 100% 50% 75% 100% 100%***",
    "Part A deductible 100% 100% 100% 100% 100% 50% 75% 50% 100%",
    "Part B deductible 100% 100%",
    "Part B excess charges 100% 100%",
    "Foreign travel ",
    "emergency (up to plan ",
    "limits) ",
    "80% 80% 80% 80% 80% 80%",
    "Out-of-pocket ",
    "limit in 2026**",
    "$8,000 $4,000",
]
MEDIGAP_FOOTNOTE = [
    " * Plans F and G also offer a high-deductible plan in some states. You must ",
    "pay Medicare-covered costs (coinsurance, copayments, and deductibles) ",
    "up to the deductible amount of $2,950 in 2026 before your policy pays ",
    "anything. You can’t buy Plans C and F if you were new to Medicare on or ",
    "after January 1, 2020 (page 75).",
]

# Pages 71 and 72: the chart of enrollment periods runs from the foot of one page onto the next.
ENROLLMENT_FOOT_OF_71 = [
    "Open ",
    "Enrollment ",
    "Period",
    "October 15 to  ",
    "December 7",
    "You can join, switch, or drop a Medicare ",
    "Advantage Plan (with or without drug ",
    "coverage) during the Open Enrollment ",
    "Period each year. ",
    "Your coverage starts on January 1 (as long ",
    "as the plan gets your enrollment request ",
    "by December 7).",
]
ENROLLMENT_TOP_OF_72 = [
    "Medicare ",
    "Advantage ",
    "Open ",
    "Enrollment ",
    "Period",
    "January 1 to ",
    "March 31",
    "Note: You can ",
    "only switch ",
    "plans once ",
    "during this ",
    "period. ",
    "Coverage ",
    "starts the first ",
    "of the month ",
    "after the plan ",
    "gets your ",
    "request.",
]

# Page 10's diagram of the two ways to get Medicare, and the chart that opens page 11.
OPTIONS_INTRO = [
    "When you first sign up for Medicare, and during certain times of the year, you ",
    "can choose how you get your Medicare coverage. There are 2 main ways to get ",
    "Medicare:",
]
OPTIONS_DIAGRAM = [
    "Original Medicare",
    "• Includes Medicare Part A (Hospital ",
    "Insurance) and Part B (Medical ",
    "Insurance).",
    "• You can join a separate Medicare ",
    "drug plan to get Medicare drug ",
    "coverage (Part D).",
    "• You can use any doctor or hospital ",
    "that takes Medicare, anywhere in ",
    "the U.S.",
    "• You can also use or shop for and ",
    "buy supplemental coverage that ",
    "helps pay your out-of-pocket costs ",
    "(like your 20% coinsurance). ",
]
AT_A_GLANCE_CHART = [
    "Doctor & hospital choice",
    "Original Medicare Medicare Advantage (Part C)",
    "You can use any doctor or hospital that ",
    "takes Medicare, anywhere in the U.S.* ",
    "You may pay more if your doctor doesn’t ",
    "accept assignment.",
    "You may need to use doctors and other ",
    "providers who are in the plan’s network ",
    "and service area (for non-emergency ",
    "care). Some plans offer non-emergency ",
    "coverage out of network, but typically at ",
    "a higher cost. ",
    "In most cases, you don’t need a referral ",
    "to use a specialist.",
    "You may need to get a referral to use a ",
    "specialist.",
]


# Rule 1: headings are found in the text, and page furniture is not text.


def test_a_heading_opens_a_chunk_and_names_its_section():
    """Page 23's question in bold starts a chunk of its own, and the chunk begins with it."""
    before = page(22, *printed(sentence("a", 40)))

    chunks = section_chunks([before, PART_B_PREMIUM])

    assert len(chunks) == 2
    assert chunks[1].section == "How much does Part B coverage cost?"
    assert chunks[1].text.startswith("How much does Part B coverage cost? The standard Part B")


def test_the_running_title_and_the_page_number_are_left_out():
    """Nothing from the top of page 23 survives but its heading."""
    (chunk,) = section_chunks([PART_B_PREMIUM])

    assert chunk.text == (
        "How much does Part B coverage cost? The standard Part B premium amount in 2026 is "
        "$202.90. Most people pay the standard Part B premium amount every month."
    )


def test_a_line_that_does_not_follow_a_finished_sentence_is_not_a_heading():
    """Page 23 prints an address after "mail your premium payments to:". It is text."""
    lines = [
        "How can I pay my Part B premium? ",
        *printed(sentence("a", 20)),
        "Note: If you get a bill from the RRB, mail your premium payments to:  ",
        "RRB Medicare Premium Payments ",
        "PO Box 979024 ",
        "St. Louis, MO 63197-9000",
        "If you have questions about bills you get from the RRB, call 1-877-772-5772. ",
        "TTY users can call 1-312-751-4701.",
    ]

    chunks = section_chunks([page(23, *lines)])

    assert sections_of(chunks) == ["How can I pay my Part B premium?"]
    assert "RRB Medicare Premium Payments PO Box 979024 St. Louis" in chunks[0].text


def test_an_index_entry_is_never_a_heading():
    """An index line ends in the page it points to ("Mental health care 46", page 6), and one
    that heads a list of them ("Social Security", page 7) is index text too."""
    lines = [
        "Index of topics",
        "Medigap. Go to Medicare Supplement ",
        "Insurance.",
        "Mental health care 46",
        "MSN. Go to Medicare Summary Notice.",
        "SNP. Go to Special Needs Plan.",
        "Social Security",
        "Change address on MSN 59",
        "Extra Help paying Part D costs 92–94",
        "Other helpful contacts 112–113",
    ]

    chunks = section_chunks([page(6, *lines)])

    assert sections_of(chunks) == ["Index of topics"]


def test_the_contents_page_opens_with_its_heading():
    """Page 3's entries end in page numbers too, but "Contents" above them is a heading."""
    contents = page(
        3,
        "3",
        "Contents",
        f"What’s new & important? {'.' * 60}2",
        f"Index of topics {'.' * 60}4",
    )

    chunks = section_chunks([contents])

    assert sections_of(chunks) == ["Contents"]


def test_a_heading_printed_over_two_lines_is_one_heading():
    """Page 88 breaks a question after "with"; the next line finishes the heading."""
    lines = [
        *printed(sentence("a", 30)),
        "How do other insurance and programs work with ",
        "Medicare drug coverage (Part D)? ",
        *printed(sentence("b", 30)),
    ]

    chunks = section_chunks([page(88, *lines)])

    assert chunks[-1].section == (
        "How do other insurance and programs work with Medicare drug coverage (Part D)?"
    )


def test_headings_that_follow_each_other_are_one_heading():
    """Page 66 names the list of plan types and then its first entry, straight after it."""
    lines = [
        *printed(sentence("a", 30)),
        "Types of Medicare Advantage Plans ",
        "Health Maintenance Organization (HMO) Plan ",
        *printed(sentence("b", 30)),
    ]

    chunks = section_chunks([page(66, *lines)])

    assert chunks[-1].section == (
        "Types of Medicare Advantage Plans Health Maintenance Organization (HMO) Plan"
    )


def test_a_heading_may_open_with_a_quotation_mark():
    """Page 54's “Welcome to Medicare” visit, under the label of a margin icon."""
    lines = [
        *printed(sentence("a", 30)),
        "Preventive service  ",
        "“Welcome to Medicare” preventive visit ",
        "During the first 12 months that you have Part B, you can get a “Welcome ",
        "to Medicare” preventive visit.",
    ]

    chunks = section_chunks([page(54, *lines)])

    assert chunks[-1].section == "“Welcome to Medicare” preventive visit"
    assert all("Preventive service" not in chunk.text for chunk in chunks)


def test_a_question_may_run_longer_than_other_headings():
    """Page 83's question is 67 characters, past the 65 of other headings but within the 75 a
    question may run to."""
    lines = [
        "Note: This payment option may not be the best choice for you if you get or ",
        "are eligible for Extra Help from Medicare, including if you get coverage from a ",
        "Medicare Savings Program.",
        "What\u2019s the Medicare drug coverage (Part D) late enrollment penalty? ",
        "The late enrollment penalty is an amount that\u2019s permanently added to your ",
        "Medicare drug coverage (Part D) premium.",
    ]

    chunks = section_chunks([page(83, *lines)])

    assert chunks[-1].section == (
        "What\u2019s the Medicare drug coverage (Part D) late enrollment penalty?"
    )


def test_a_question_asked_of_every_plan_type_stays_in_the_text():
    """Pages 66-70 ask each plan type the same questions. A heading repeated three or more
    times in one Section belongs to the entry it is in, and the plan type stays the section."""
    plans = [
        "Health Maintenance Organization (HMO) Plan",
        "Preferred Provider Organization (PPO) Plan",
        "Private Fee-for-Service (PFFS) Plan",
    ]
    lines = []
    for number, plan in enumerate(plans):
        lines += [
            f"{plan} ",
            *printed(sentence(f"p{number}", 20)),
            "Do I need to choose a primary care doctor? ",
            *printed(sentence(f"q{number}", 20)),
        ]

    chunks = section_chunks([page(66, *lines)])

    assert sections_of(chunks) == plans
    assert all("Do I need to choose a primary care doctor?" in chunk.text for chunk in chunks)


def test_page_furniture_is_left_out():
    """A line that opens three or more pages is furniture wherever it appears, like the note
    that opens every Section; so are page numbers, even printed twice, and the margin icons'
    labels."""
    note = "Note: Go to pages 119–122 for definitions of blue words. "
    pages = [
        page(number, str(number), note, "Acupuncture", *printed(sentence(f"a{number}", 20)))
        for number in (25, 57, 61)
    ]
    pages.append(page(123, "123123", "Blood", *printed(sentence("b", 20)), note, "New!"))

    chunks = section_chunks(pages)

    words = " ".join(chunk.text for chunk in chunks).split()
    assert "blue" not in words
    assert "New!" not in words
    assert not {"25", "57", "61", "123123"} & set(words)


# Rule 2: every chunk is labelled with its heading, under the Section it sits in.


CONTENTS = page(3, "3", "Contents", f"Section 1: Signing up for Medicare {'.' * 60}15")


def test_a_section_label_starts_with_the_section_it_sits_in():
    """The contents page says Section 1 starts on page 15, where its title runs over two
    lines. Every heading after it is labelled with it, and the title alone labels nothing."""
    opener = page(
        15,
        "15",
        "Section 1:  ",
        "Signing up for Medicare",
        "Will I get Part A and Part B automatically?",
        *printed(sentence("a", 30)),
    )
    later = page(
        16,
        "Section 1: Signing up for Medicare",
        "16",
        "Where can I get more information?",
        *printed(sentence("b", 30)),
    )

    chunks = section_chunks([CONTENTS, opener, later])

    assert sections_of(chunks) == [
        "Contents",
        "Section 1: Signing up for Medicare > Will I get Part A and Part B automatically?",
        "Section 1: Signing up for Medicare > Where can I get more information?",
    ]


def test_a_heading_with_nothing_under_it_makes_no_chunk():
    """Page 96 is a blank page headed "Notes"; its heading has nothing to label."""
    pages = [page(95, "Acupuncture", *printed(sentence("a", 20))), page(96, "96", "Notes")]

    chunks = section_chunks(pages)

    assert sections_of(chunks) == ["Acupuncture"]


def test_text_before_the_first_heading_is_labelled_front_matter():
    """No chunk is ever left without a section."""
    chunks = section_chunks([page(1, *printed(sentence("a", 20)))])

    assert sections_of(chunks) == [NO_HEADING]


# Rule 3: paragraphs are packed whole, and only a paragraph too long for a chunk is split.


def test_paragraphs_are_packed_whole_up_to_max_words():
    """Three paragraphs of 20 words with room for 45: two share a chunk, the third starts the
    next one, and none is cut."""
    lines = [
        *printed(sentence("a", 20)),
        *printed(sentence("b", 20)),
        *printed(sentence("c", 20)),
    ]

    chunks = section_chunks([page(1, *lines)], max_words=45)

    assert [len(words_of(chunk)) for chunk in chunks] == [40, 20]
    assert words_of(chunks[1])[0] == "C000"


def test_a_paragraph_longer_than_max_words_is_split_between_sentences():
    """Three sentences of 30 words in one paragraph, with room for 40: a chunk each."""
    sentences = [sentence("a", 30), sentence("b", 30), sentence("c", 30)]

    chunks = section_chunks([page(1, *printed(" ".join(sentences)))], max_words=40)

    assert [chunk.text for chunk in chunks] == sentences


def test_a_sentence_longer_than_max_words_is_cut_as_a_last_resort():
    """A sentence of 100 words with room for 40 is cut rather than left too long."""
    chunks = section_chunks([page(1, *printed(sentence("a", 100)))], max_words=40)

    assert [len(words_of(chunk)) for chunk in chunks] == [40, 40, 20]


def test_a_heading_does_not_count_toward_max_words():
    """The heading rides at the head of its section's first chunk, on top of the passage."""
    lines = ["How much does Part B coverage cost? ", *printed(sentence("a", 40))]

    (chunk,) = section_chunks([page(23, *lines)], max_words=40)

    assert len(words_of(chunk)) == 7 + 40


# Rule 4: a table is one chunk of its own, never split and never merged with prose.


def test_a_table_is_one_chunk_however_long():
    """Page 21's chart of who pays first holds more than max_words and stays whole."""
    lines = [*WHO_PAYS_INTRO, *WHO_PAYS_CHART, *WHO_PAYS_NOTE]

    chunks = section_chunks([page(21, *lines)], max_words=40)

    charts = [chunk for chunk in chunks if "like insurance from your or your spouse" in chunk.text]
    assert len(charts) == 1
    assert len(words_of(charts[0])) > 40
    assert charts[0].text.startswith("If you have retiree health coverage, like insurance")
    assert charts[0].text.endswith("If you have Medicaid... Medicare pays first.")


def test_a_table_is_never_merged_with_the_prose_around_it():
    """The paragraph above the chart and the note below it would fit beside it; they don't."""
    lines = [*WHO_PAYS_INTRO, *WHO_PAYS_CHART, *WHO_PAYS_NOTE]

    chunks = section_chunks([page(21, *lines)])

    assert [chunk.text.split()[0] for chunk in chunks] == ["How", "If", "Important!"]
    assert set(sections_of(chunks)) == {"How does my other insurance work with Medicare?"}


def test_a_row_of_amounts_is_a_table_line_however_wide():
    """Page 76's "Part A deductible 100% 100% ..." row is wider than a cell, but a row of
    percentages is a row of the chart, and the chart stays one chunk."""
    lines = [*MEDIGAP_INTRO, *MEDIGAP_CHART]

    chunks = section_chunks([page(76, *lines)], max_words=20)

    charts = [chunk for chunk in chunks if "Benefits A B C D F* G* K L M N" in chunk.text]
    assert len(charts) == 1
    assert charts[0].text.startswith("Medigap standardized plans Benefits")
    assert charts[0].text.endswith("limit in 2026** $8,000 $4,000")


def test_footnotes_marked_with_asterisks_stay_with_their_table():
    """The footnote under the Medigap chart explains its asterisks, so it is part of it."""
    lines = [*MEDIGAP_INTRO, *MEDIGAP_CHART, *MEDIGAP_FOOTNOTE]

    chunks = section_chunks([page(76, *lines)])

    chart = next(chunk for chunk in chunks if "Benefits A B C D" in chunk.text)
    assert chart.text.endswith(
        "You can’t buy Plans C and F if you were new to Medicare on "
        "or after January 1, 2020 (page 75)."
    )


def test_a_table_runs_on_across_a_page_break():
    """The enrollment periods on pages 71-72 are one chart, and one chunk."""
    foot_of_71 = page(71, "71", *printed(sentence("a", 30)), *ENROLLMENT_FOOT_OF_71)
    top_of_72 = page(
        72,
        "Section 4: Medicare Advantage Plans & other options",
        "72",
        *ENROLLMENT_TOP_OF_72,
        *printed(sentence("b", 30)),
    )

    chunks = section_chunks([foot_of_71, top_of_72])

    charts = [chunk for chunk in chunks if chunk.text.startswith("Open Enrollment Period")]
    assert len(charts) == 1
    assert (charts[0].page_start, charts[0].page_end) == (71, 72)
    assert charts[0].text.endswith("after the plan gets your request.")


def test_a_title_from_the_contents_page_ends_a_table():
    """The diagram at the foot of page 10 and the chart that opens page 11 meet at the page
    break, but the contents page says a new part of the book starts on page 11."""
    contents = page(
        3,
        "3",
        "Contents",
        f"Your Medicare options {'.' * 60}10",
        f"At a glance: Original Medicare vs. Medicare Advantage Plan {'.' * 20}11",
    )
    options = page(10, "10", "Your Medicare options", *OPTIONS_INTRO, *OPTIONS_DIAGRAM)
    at_a_glance = page(
        11,
        "11",
        "At a glance: Original Medicare ",
        "vs. Medicare Advantage Plan",
        *AT_A_GLANCE_CHART,
    )

    chunks = section_chunks([contents, options, at_a_glance])

    assert [(chunk.section, chunk.page_start, chunk.page_end) for chunk in chunks[1:]] == [
        ("Your Medicare options", 10, 10),
        ("Your Medicare options", 10, 10),
        ("At a glance: Original Medicare vs. Medicare Advantage Plan", 11, 11),
    ]
    assert chunks[2].text.startswith("Original Medicare • Includes Medicare Part A")
    assert chunks[3].text.startswith(
        "At a glance: Original Medicare vs. Medicare Advantage Plan Doctor & hospital choice"
    )


def test_short_lines_that_do_not_wrap_are_not_a_table():
    """The index on page 4 is narrow too, but each line is one entry, not a cell that wraps,
    so it packs like any text: with room for 10 words it takes several chunks."""
    entries = [
        "Chemotherapy 34, 65",
        "Chiropractic services 35",
        "Chronic care management services 35",
        "Claims 40, 58, 59, 102–105, 109",
        "Clinical research studies 27, 35, 62, 109",
        "COBRA 18–19, 89",
        "Cognitive assessment 35–36, 54",
        "Colonoscopies 36",
        "Connected apps 59, 109",
        "Cosmetic surgery 55",
        "Cost Plan. Go to Medicare Cost Plans.",
    ]

    chunks = section_chunks([page(4, *entries)], max_words=10)

    assert len(chunks) > 1


# Rule 5: pages come from the words, and ordinals count from zero.


def test_a_chunk_records_the_pages_of_its_first_and_last_word():
    """A paragraph that runs from the foot of page 5 onto page 6 makes a chunk for 5-6."""
    lines = printed(sentence("a", 40))

    chunks = section_chunks([page(5, *lines[:2]), page(6, *lines[2:])])

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(5, 6)]


def test_ordinals_count_from_zero_in_document_order():
    """The ordinal is a chunk's position in the handbook, which the database stores."""
    headings = ["Acupuncture", "Advance care planning", "Ambulance services", "Blood"]
    lines = []
    for number, heading in enumerate(headings):
        lines += [heading, *printed(sentence(f"a{number}", 20))]

    chunks = section_chunks([page(31, *lines)])

    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2, 3]
    assert sections_of(chunks) == headings


def test_the_registry_holds_the_sections_chunker_beside_the_fixed_one():
    """`CHUNK_SET=sections` picks the sections chunker, with its default of 300 words."""
    assert CHUNKERS["sections"] is section_chunks
    assert sorted(CHUNKERS) == ["fixed-220w", "sections"]


def test_a_document_with_no_words_has_no_sections():
    """Nothing in, nothing out, and a blank page like page 127 adds nothing."""
    assert section_chunks([]) == []
    assert section_chunks([Page(number=127, text="")]) == []


@pytest.mark.parametrize("max_words", [0, -5])
def test_a_chunk_must_have_room_for_a_word(max_words: int):
    """A chunk with room for no words could never hold a paragraph."""
    with pytest.raises(ValueError):
        section_chunks([page(1, *printed(sentence("a", 20)))], max_words=max_words)
