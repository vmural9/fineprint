"""Tests for turning retrieved passages into a cited answer.

Nothing here calls a model or a database. `FakeChatModel` returns whatever draft the test
hands it, which is how the tests can ask what the service does with a bad draft — a citation
for a chunk that was never retrieved, a quote the handbook does not contain — without having
to talk a real model into producing one.

The chunk texts are real sentences from the 2026 handbook, taken from the golden set's
verified evidence, so a passing test says something about the handbook and not only about
the code.
"""

import pytest

from fineprint.answer import (
    SYSTEM_PROMPT,
    AnswerResponse,
    Citation,
    DraftAnswer,
    answer_question,
    build_user_prompt,
    collapse_whitespace,
    format_excerpt,
)
from fineprint.retrieval import RetrievedChunk
from tests.fakes import FakeChatModel

PREMIUM_TEXT = (
    "The standard Part B premium amount in 2026 is $202.90. Most people pay the standard "
    "Part B premium amount every month. If your modified adjusted gross income is above a "
    "certain amount, you may pay an Income-Related Monthly Adjustment Amount (IRMAA)."
)
HEARING_TEXT = (
    "Note: Original Medicare doesn't cover hearing aids or exams for fitting hearing aids. "
    "Medicare covers diagnostic hearing and balance (fall risk) exams if your doctor or "
    "health care provider orders them to see if you need medical treatment."
)
IRMAA_TEXT = (
    "For 2026, if your modified adjusted gross income for 2024 was above $109,000 if you "
    "file individually or $218,000 if you're married and file jointly, then you may pay an "
    "IRMAA. Visit Medicare.gov for more information."
)

PREMIUM = RetrievedChunk(
    chunk_id=61,
    page_start=23,
    page_end=23,
    text=PREMIUM_TEXT,
    score=0.0325,
    rank=1,
    lexical_rank=1,
    vector_rank=2,
)
IRMAA = RetrievedChunk(
    chunk_id=62,
    page_start=23,
    page_end=24,
    text=IRMAA_TEXT,
    score=0.0310,
    rank=2,
    lexical_rank=3,
    vector_rank=1,
)
HEARING = RetrievedChunk(
    chunk_id=118,
    page_start=42,
    page_end=42,
    text=HEARING_TEXT,
    score=0.0161,
    rank=3,
    lexical_rank=None,
    vector_rank=4,
)


class FakeRetriever:
    """Returns a fixed list of chunks and remembers how it was asked for them."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks
        self.calls: list[tuple[str, str, int | None]] = []

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((question, mode, top_k))
        return list(self.chunks)


def draft(
    answer: str = "The standard Part B premium in 2026 is $202.90 a month.",
    *,
    found_in_handbook: bool = True,
    citations: list[Citation] | None = None,
    confidence: str = "high",
) -> DraftAnswer:
    """A model's draft, with one exact quote from the premium chunk unless told otherwise."""
    if citations is None:
        citations = [
            Citation(chunk_id=61, quote="The standard Part B premium amount in 2026 is $202.90.")
        ]
    return DraftAnswer(
        answer=answer,
        found_in_handbook=found_in_handbook,
        citations=citations,
        confidence=confidence,
    )


def ask(
    response: DraftAnswer,
    chunks: list[RetrievedChunk] | None = None,
    question: str = "How much is the Part B premium in 2026?",
    top_k: int | None = None,
    mode: str = "hybrid",
) -> tuple[AnswerResponse, FakeChatModel, FakeRetriever]:
    """Run `answer_question` with a canned draft, and hand back everything worth asserting on."""
    retriever = FakeRetriever(chunks if chunks is not None else [PREMIUM, IRMAA, HEARING])
    chat = FakeChatModel(response)
    answered = answer_question(question, retriever, chat, top_k=top_k, mode=mode)
    return answered, chat, retriever


# --- what the model is shown ------------------------------------------------------------


def test_every_retrieved_chunk_appears_once_in_ranked_order():
    _, chat, _ = ask(draft())

    user = chat.calls[0].user
    positions = [user.index(f'<chunk id="{chunk.chunk_id}"') for chunk in (PREMIUM, IRMAA, HEARING)]
    assert positions == sorted(positions), "excerpts must be in the order retrieval ranked them"
    for chunk in (PREMIUM, IRMAA, HEARING):
        assert user.count(f'<chunk id="{chunk.chunk_id}"') == 1, "each excerpt exactly once"
        assert chunk.text in user


def test_an_excerpt_carries_its_chunk_id_and_its_pages():
    assert format_excerpt(PREMIUM).startswith('<chunk id="61" pages="23">')
    assert format_excerpt(IRMAA).startswith('<chunk id="62" pages="23-24">')
    assert format_excerpt(PREMIUM).endswith("</chunk>")


def test_the_question_is_in_the_prompt_and_the_schema_is_the_draft():
    _, chat, _ = ask(draft())

    call = chat.calls[0]
    assert "How much is the Part B premium in 2026?" in call.user
    assert call.schema is DraftAnswer


def test_the_system_prompt_carries_every_rule_the_answer_depends_on():
    lowered = SYSTEM_PROMPT.lower()
    assert "medicare & you 2026" in lowered
    assert "only" in lowered, "answer only from the excerpts"
    assert "exactly" in lowered, "copy dollar amounts and phone numbers exactly"
    assert "found_in_handbook" in lowered, "what to do when the handbook does not say"
    assert "medicare.gov" in lowered and "1-800-medicare" in lowered
    assert "state health insurance assistance program" in lowered
    assert "social security" in lowered
    assert "medical advice" in lowered
    # A quote is checked against its excerpt, so the model is told to copy one exactly. The
    # handbook's curly apostrophes are the ones a model straightens without noticing.
    assert "character for character" in lowered
    assert "’" in SYSTEM_PROMPT
    # The three confidence levels are defined, not just named.
    assert "high" in lowered and "medium" in lowered and "low" in lowered
    assert "directly" in lowered and "inference" in lowered and "partial" in lowered


def test_a_search_that_found_nothing_still_says_so_to_the_model():
    answered, chat, _ = ask(
        draft(found_in_handbook=False, citations=[], confidence="low"), chunks=[]
    )

    assert "no excerpts" in chat.calls[0].user.lower()
    assert answered.retrieved_chunk_ids == []
    assert answered.citations == []


def test_top_k_is_passed_through_to_the_retriever():
    _, _, retriever = ask(draft(), top_k=3)

    assert retriever.calls == [("How much is the Part B premium in 2026?", "hybrid", 3)]


def test_mode_is_forwarded_to_the_retriever_when_no_chunks_are_given():
    _, _, retriever = ask(draft(), mode="lexical")

    assert retriever.calls == [("How much is the Part B premium in 2026?", "lexical", None)]


def test_given_chunks_skip_retrieval_and_mode_is_ignored():
    # The retriever is primed with a different chunk than the one passed in through `chunks`.
    # If `answer_question` retrieved anyway, the answer would be built from IRMAA, not PREMIUM.
    retriever = FakeRetriever([IRMAA])
    chat = FakeChatModel(draft())

    answered = answer_question(
        "How much is the Part B premium in 2026?",
        retriever,
        chat,
        mode="lexical",
        chunks=[PREMIUM],
    )

    assert retriever.calls == [], "chunks were given, so the retriever must not be called"
    assert answered.retrieved_chunk_ids == [61]
    user = chat.calls[0].user
    assert 'chunk id="61"' in user, "the answer is built from the given chunk"
    assert 'chunk id="62"' not in user, "not from what the retriever would have returned"
    assert answered.citations[0].chunk_id == 61
    assert answered.citations[0].quote_verified is True


def test_the_prompt_is_built_from_the_chunks_it_is_given():
    prompt = build_user_prompt("Does Medicare cover hearing aids?", [HEARING])

    assert "Does Medicare cover hearing aids?" in prompt
    assert '<chunk id="118" pages="42">' in prompt


# --- what comes back --------------------------------------------------------------------


def test_page_numbers_come_from_the_retrieved_rows_not_from_the_model():
    answered, _, _ = ask(
        draft(
            citations=[
                Citation(chunk_id=62, quote="For 2026, if your modified adjusted gross income")
            ]
        )
    )

    (cited,) = answered.citations
    assert (cited.chunk_id, cited.page_start, cited.page_end) == (62, 23, 24)


def test_a_citation_for_a_chunk_that_was_not_retrieved_is_dropped_and_counted():
    answered, _, _ = ask(
        draft(
            citations=[
                Citation(
                    chunk_id=61, quote="The standard Part B premium amount in 2026 is $202.90."
                ),
                Citation(chunk_id=999, quote="Medicare pays for everything."),
            ]
        )
    )

    assert [cited.chunk_id for cited in answered.citations] == [61]
    assert answered.dropped_citations == 1


def test_an_exact_quote_verifies():
    answered, _, _ = ask(draft())

    assert answered.citations[0].quote_verified is True


def test_a_quote_that_differs_only_in_whitespace_verifies():
    answered, _, _ = ask(
        draft(
            citations=[
                Citation(
                    chunk_id=61,
                    quote="The standard Part B premium\n  amount in 2026\tis $202.90.",
                )
            ]
        )
    )

    assert answered.citations[0].quote_verified is True


def test_an_empty_quote_does_not_verify():
    answered, _, _ = ask(draft(citations=[Citation(chunk_id=61, quote="   ")]))

    (cited,) = answered.citations
    assert cited.quote_verified is False, "an empty span is in every text; it is not evidence"


def test_an_invented_quote_does_not_verify():
    answered, _, _ = ask(
        draft(
            citations=[
                Citation(chunk_id=61, quote="The standard Part B premium in 2026 is $164.90.")
            ]
        )
    )

    (cited,) = answered.citations
    assert cited.quote_verified is False
    assert cited.quote == "The standard Part B premium in 2026 is $164.90.", "kept, not hidden"


def test_the_answer_carries_the_question_the_chunks_and_what_the_call_cost():
    answered, _, _ = ask(draft())

    assert answered.question == "How much is the Part B premium in 2026?"
    assert answered.answer == "The standard Part B premium in 2026 is $202.90 a month."
    assert answered.found_in_handbook is True
    assert answered.confidence == "high"
    assert answered.retrieved_chunk_ids == [61, 62, 118]
    assert answered.model == "fake-chat-model"
    assert (answered.input_tokens, answered.output_tokens) == (900, 120)
    assert answered.latency_ms == 12.5


def test_an_abstention_is_an_ordinary_answer():
    answered, _, _ = ask(
        draft(
            answer=(
                "The handbook doesn't list the drugs a plan covers. Check the plan's formulary "
                "or visit Medicare.gov/plan-compare."
            ),
            found_in_handbook=False,
            citations=[],
            confidence="low",
        ),
        question="Does Medicare drug coverage pay for Eliquis?",
    )

    assert answered.found_in_handbook is False
    assert answered.citations == []
    assert answered.dropped_citations == 0
    assert "Medicare.gov" in answered.answer


def test_several_quotes_from_the_same_chunk_are_all_kept():
    answered, _, _ = ask(
        draft(
            citations=[
                Citation(
                    chunk_id=61, quote="The standard Part B premium amount in 2026 is $202.90."
                ),
                Citation(chunk_id=61, quote="Most people pay the standard Part B premium amount"),
            ]
        )
    )

    assert [cited.chunk_id for cited in answered.citations] == [61, 61]
    assert all(cited.quote_verified for cited in answered.citations)


# --- the whitespace rule itself ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one  two", "one two"),
        ("one\ntwo", "one two"),
        ("  one\t\ttwo  ", "one two"),
        ("one", "one"),
        ("", ""),
    ],
)
def test_whitespace_runs_collapse_to_one_space(text: str, expected: str):
    assert collapse_whitespace(text) == expected
