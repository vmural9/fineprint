"""The two tests that really call Bedrock.

Everything else in the suite runs offline. These carry the `live` marker, so a plain
`uv run pytest` never reaches AWS; run them deliberately with

    AWS_PROFILE=merlion-brands AWS_REGION=us-west-2 uv run pytest -m live -q

They cost a few cents. `DraftAnswer` is repeated here rather than imported because it belongs
to `fineprint.answer`, which task 7 adds.
"""

from typing import Literal

import boto3
import pytest
from pydantic import BaseModel

from fineprint.config import Settings
from fineprint.providers.factory import get_chat_model, get_embedder

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
        pytest.skip("no AWS credentials: run with AWS_PROFILE=merlion-brands AWS_REGION=us-west-2")
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
        '<chunk id="7" pages="18-18">In 2026, you pay a standard monthly Part B premium of '
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
