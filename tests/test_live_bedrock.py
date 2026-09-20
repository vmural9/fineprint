"""The tests that really call Bedrock.

Everything else in the suite runs offline. These carry the `live` marker, so a plain
`uv run pytest` never reaches AWS; run them deliberately with

    AWS_PROFILE=merlion-brands AWS_REGION=us-west-2 uv run pytest -m live -q

They cost a few cents.
"""

import boto3
import pytest

from fineprint.config import Settings
from fineprint.providers.factory import get_embedder

pytestmark = pytest.mark.live


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
