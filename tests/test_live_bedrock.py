"""The tests that really call Bedrock.

Everything else in the suite runs offline. These carry the `live` marker, so a plain
`uv run pytest` never reaches AWS; run them deliberately with

    AWS_PROFILE=<your-aws-profile> AWS_REGION=us-west-2 uv run pytest -m live -q

They cost a few cents. `DraftAnswer` is repeated here rather than imported from
`fineprint.answer` so that this file tests the provider against a schema of its own: a change to
the service's answer model cannot silently change what the live test asserts.
"""

from typing import Literal

import boto3
import pytest
from pydantic import BaseModel

from fineprint.config import Settings
from fineprint.providers.factory import get_chat_model, get_embedder, get_reranker

pytestmark = pytest.mark.live


class Citation(BaseModel):
    chunk_id: int
    quote: str


class DraftAnswer(BaseModel):
    answer: str
    found_in_handbook: bool
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Settings for the real Bedrock models, skipping when no AWS credentials are around."""
    if boto3.session.Session().get_credentials() is None:
        pytest.skip("no AWS credentials: run with AWS_PROFILE=<your-profile> AWS_REGION=us-west-2")
    return Settings()


def test_titan_returns_1024_numbers_and_the_same_text_twice_gives_the_same_vector(
    settings: Settings,
):
    embedder = get_embedder(settings)
    question = "What is the standard Part B premium in 2026?"

    first = embedder.embed_query(question)
    second = embedder.embed_query(question)

    assert len(first) == embedder.dimension == 1024
    assert first == second


def test_claude_opus_5_answers_one_excerpt_as_a_validated_draft_answer(settings: Settings):
    chat = get_chat_model(settings)
    system = (
        "You answer questions about Medicare using only the supplied excerpts of the handbook "
        "Medicare & You 2026. Copy dollar amounts exactly. Cite the chunk ids you used, each "
        "with a short verbatim quote from that chunk."
    )
    user = (
        '<chunk id="7" pages="23-23">In 2026, you pay a standard monthly Part B premium of '
        "$202.90. Some people pay more based on income.</chunk>\n\n"
        "Question: What is the standard Part B premium in 2026?"
    )

    result = chat.complete_structured(system, user, DraftAnswer)

    assert isinstance(result.parsed, DraftAnswer)
    assert result.parsed.found_in_handbook is True
    assert "202.90" in result.parsed.answer
    assert [citation.chunk_id for citation in result.parsed.citations] == [7]
    assert result.model == settings.llm_model
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms > 0


def test_cohere_rerank_puts_the_passage_that_answers_the_question_first(settings: Settings):
    reranker = get_reranker(settings.model_copy(update={"reranker_provider": "bedrock"}))
    assert reranker is not None
    # Question q004 of the golden set, and three passages copied from the handbook's pages 40,
    # 53 and 42. The one that answers the question goes last, so finding it first is the
    # re-ranker's doing.
    question = (
        "My mother is having trouble hearing. Does Original Medicare pay for hearing aids, or "
        "for the exam to fit them?"
    )
    passages = [
        "Medicare covers medically necessary items like oxygen and oxygen equipment, walkers, "
        "and hospital beds when a doctor or other health care provider orders them for use in "
        "the home.",
        "Medicare may cover medically necessary ambulance transportation to a foreign hospital "
        "only with admission for medically necessary covered inpatient hospital services.",
        "Note: Original Medicare doesn’t cover hearing aids or exams for fitting hearing aids.",
    ]

    results = reranker.rerank(question, passages, top_n=3)

    assert results[0].index == 2
    assert sorted(result.index for result in results) == [0, 1, 2]
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
