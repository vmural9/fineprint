"""Turning retrieved passages into an answer that can be checked.

Retrieval finds the handbook passages a question is about. This module hands those passages
to Claude with strict instructions, and then checks what comes back before anyone sees it.

The check is the point. A model asked for a page number will happily produce one, and a
wrong page number is worse than no answer: it reads like evidence. So the model is never
asked for a page. It is shown each excerpt wrapped in a tag that carries the excerpt's chunk
id — `<chunk id="61" pages="23">…</chunk>` — and it cites those ids. The service then does
three things the model cannot influence:

1. It drops any citation whose chunk id is not one of the chunks it just retrieved, and
   counts the drops in `dropped_citations`. An id the model invented cannot survive.
2. It fills in `page_start` and `page_end` from the retrieved rows, which came from the
   database, which came from the PDF. Every page number a reader sees was measured, not
   written by a model.
3. It sets `quote_verified` by looking for the quote in the chunk's own text. A quote that
   is not there is still shown, marked unverified, because hiding it would hide the failure.

Verification collapses whitespace runs on both sides and compares nothing else. The chunker
already joins words with single spaces, so the handbook's line breaks are gone from the
stored text; collapsing the model's quote too means a quote copied across a line break still
matches. Everything else — a changed digit, a straightened apostrophe, a dropped word — is
left to fail, because those are exactly the differences worth seeing.
"""

from typing import Literal

from pydantic import BaseModel, Field

from fineprint.providers.base import ChatModel
from fineprint.retrieval import RetrievedChunk, Retriever

# What the model is told before it sees the question. Every rule here exists because the
# alternative is an answer that looks right and is not: a remembered premium from a previous
# year, a rounded dollar amount, a helpful suggestion about somebody's health.
SYSTEM_PROMPT = """\
You answer questions about Medicare for people on Medicare and their families, using only \
the excerpts of the official U.S. government handbook "Medicare & You 2026" that come with \
the question.

Each excerpt arrives wrapped in a tag that carries its chunk id and the handbook pages it \
came from, like this: <chunk id="61" pages="23">…</chunk>. The excerpts are ordered with the \
most relevant first.

Rules:

1. Answer only from the excerpts. They are the whole of the evidence. Do not add anything \
else you know about Medicare, do not fill a gap from memory, and do not guess.

2. Copy dollar amounts, percentages, dates, deadlines, phone numbers and web addresses \
exactly as the excerpts write them. Never round a figure, never update one to a year the \
excerpts do not mention, and never convert one into other words.

3. Cite every excerpt you used. A citation is the chunk id of an excerpt you were given, \
plus a short quote from that same excerpt — long enough to show the fact, and no longer than \
about 30 words. Copy the quote character for character: one continuous span of the excerpt, \
with the handbook's own punctuation, including its curly apostrophes and quotation marks \
(’ “ ”) and its en dashes (–). Do not straighten a mark, tidy the wording, or join two \
passages with an ellipsis; quote a shorter span instead. The service looks for every quote \
in the excerpt it names and marks the ones it cannot find, so a quote that is not copied \
exactly makes a correct answer look doubtful. Do not cite a chunk id you were not given, and \
do not put page numbers in your citations: the service fills those in from its own records.

4. If the excerpts do not answer the question, set found_in_handbook to false, say plainly \
that the handbook does not say, and send the reader somewhere that does know: a resource \
named in the excerpts if there is one, otherwise Medicare.gov, 1-800-MEDICARE \
(TTY 1-877-486-2048), their State Health Insurance Assistance Program, or Social Security \
for questions about enrolling and premiums. Cite any excerpt that explains why the handbook \
does not carry the answer. Set found_in_handbook to true only when the excerpts really do \
answer the question.

5. Never give personal medical advice and never tell the reader what care to have. Say what \
the handbook says about what is covered and what it costs, and leave the medical decision to \
the reader and their doctor.

6. Write plainly, in the fewest sentences that answer the question. Use the reader's words \
where the handbook allows it, and explain a handbook term the first time you use it.

Confidence:
- high: the text you cited states the answer directly.
- medium: the answer puts two or more passages together, or needs a small inference from them.
- low: the support in the excerpts is partial — they point towards the answer without \
settling it.\
"""

# What a search that returned nothing says in place of the excerpts. It is spelled out rather
# than left blank so the model abstains for the honest reason instead of inventing evidence.
NO_EXCERPTS = "No excerpts were retrieved for this question."


class Citation(BaseModel):
    """One piece of evidence, as the model gives it: which excerpt, and the words used."""

    chunk_id: int = Field(description="The id of the excerpt this fact came from")
    quote: str = Field(description="A short span copied word for word from that excerpt")


class DraftAnswer(BaseModel):
    """What the model returns. Nothing here is trusted until the service has checked it."""

    answer: str = Field(description="The answer to the question, in plain language")
    found_in_handbook: bool = Field(
        description="True only when the excerpts really answer the question"
    )
    citations: list[Citation] = Field(description="The excerpts the answer used")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high: stated directly; medium: combined or inferred; low: partial support"
    )


class CitedChunk(BaseModel):
    """One citation after checking: the model's quote, and the pages the database gave."""

    chunk_id: int
    page_start: int
    page_end: int
    quote: str
    quote_verified: bool


class AnswerResponse(BaseModel):
    """The whole answer, with everything a reader or an eval needs to judge it.

    `dropped_citations` and `retrieved_chunk_ids` are here so a failure is visible rather
    than silent: the first counts citations that named an excerpt the model was never given,
    the second says what it actually had to work with.
    """

    question: str
    answer: str
    found_in_handbook: bool
    citations: list[CitedChunk]
    dropped_citations: int
    confidence: Literal["high", "medium", "low"]
    retrieved_chunk_ids: list[int]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def collapse_whitespace(text: str) -> str:
    """Return `text` with every run of whitespace turned into one space, and the ends trimmed."""
    return " ".join(text.split())


def format_pages(chunk: RetrievedChunk) -> str:
    """`23` for a chunk inside one page, `23-24` for one that crosses a boundary."""
    if chunk.page_start == chunk.page_end:
        return str(chunk.page_start)
    return f"{chunk.page_start}-{chunk.page_end}"


def format_excerpt(chunk: RetrievedChunk) -> str:
    """One excerpt as the model sees it, carrying the id it must cite."""
    return f'<chunk id="{chunk.chunk_id}" pages="{format_pages(chunk)}">{chunk.text}</chunk>'


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """The question and the excerpts, most relevant first."""
    excerpts = "\n\n".join(format_excerpt(chunk) for chunk in chunks) if chunks else NO_EXCERPTS
    return (
        f"Question: {question}\n\n"
        f"Excerpts from Medicare & You 2026, most relevant first:\n\n"
        f"{excerpts}"
    )


def check_citations(
    citations: list[Citation], chunks: list[RetrievedChunk]
) -> tuple[list[CitedChunk], int]:
    """Keep the citations that name a retrieved chunk, and count the ones that do not.

    The kept ones get their pages from the retrieved row and a `quote_verified` flag from
    looking for the quote in that row's text. The dropped ones are only counted: a citation
    for an excerpt the model was never shown says nothing about the handbook.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    kept: list[CitedChunk] = []
    dropped = 0
    for citation in citations:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            dropped += 1
            continue
        kept.append(
            CitedChunk(
                chunk_id=chunk.chunk_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                quote=citation.quote,
                quote_verified=collapse_whitespace(citation.quote)
                in collapse_whitespace(chunk.text),
            )
        )
    return kept, dropped


def answer_question(
    question: str, retriever: Retriever, chat: ChatModel, top_k: int | None = None
) -> AnswerResponse:
    """Search the handbook for `question`, ask the model, and check what it says.

    The model is asked even when the search found nothing: it is the model's job to say the
    handbook does not answer this and where else to look, and a service-written abstention
    would be a sentence nobody wrote for this question.
    """
    chunks = list(retriever.search(question, top_k=top_k))
    result = chat.complete_structured(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(question, chunks),
        schema=DraftAnswer,
    )
    draft = result.parsed
    citations, dropped = check_citations(draft.citations, chunks)
    return AnswerResponse(
        question=question,
        answer=draft.answer,
        found_in_handbook=draft.found_in_handbook,
        citations=citations,
        dropped_citations=dropped,
        confidence=draft.confidence,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
