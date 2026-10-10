"""The test that really sends a trace to Langfuse and reads it back.

Everything else in the suite runs offline. This one carries the `live` marker, so a plain
`uv run pytest` never reaches Langfuse; run it deliberately, with the three Langfuse values in
`.env`, using

    uv run pytest -m live tests/test_live_langfuse.py -q

It makes no Bedrock call. It sends one trace holding one generation span with fixed token
counts, priced as Claude Opus 5, and checks that Langfuse hands back the usage and the cost
exactly as they were sent. Langfuse ingests asynchronously, so the read is retried for up to a
minute.
"""

import time

import pytest

from fineprint.config import Settings
from fineprint.langfuse_sink import LangfuseSink
from fineprint.pricing import generation_cost
from fineprint.tracing import Tracer

pytestmark = pytest.mark.live

MODEL = "us.anthropic.claude-opus-5"
INPUT_TOKENS = 4000
OUTPUT_TOKENS = 800
WAIT_SECONDS = 60
POLL_SECONDS = 3


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Settings with the Langfuse project's keys, skipping when any of the three is missing."""
    settings = Settings()
    missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
            ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
            ("LANGFUSE_BASE_URL", settings.langfuse_base_url),
        )
        if value is None
    ]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set; put the Langfuse project's values in .env")
    return settings


def test_one_generation_comes_back_with_the_usage_and_cost_it_was_sent(settings: Settings):
    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    assert settings.langfuse_base_url is not None
    sink = LangfuseSink(
        settings.langfuse_public_key,
        settings.langfuse_secret_key.get_secret_value(),
        settings.langfuse_base_url,
    )
    tracer = Tracer(sink)
    cost = generation_cost(MODEL, INPUT_TOKENS, OUTPUT_TOKENS)

    try:
        with (
            tracer.trace("live-check", input="usage and cost", tags=["live-test"]) as trace,
            tracer.span("generation", "answer", model=MODEL) as span,
        ):
            span.end(
                output="fixed token counts, no model was called",
                model=MODEL,
                input_tokens=INPUT_TOKENS,
                output_tokens=OUTPUT_TOKENS,
                cost=cost,
            )
        assert trace.trace_id is not None
        tracer.flush()

        generation = None
        deadline = time.monotonic() + WAIT_SECONDS
        while generation is None and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            found = sink.client.api.observations.get_many(
                trace_id=trace.trace_id, fields="core,basic,usage"
            )
            generation = next((o for o in found.data if o.type == "GENERATION"), None)
    finally:
        tracer.shutdown()

    assert generation is not None, f"Langfuse returned no generation within {WAIT_SECONDS} seconds"
    assert generation.usage_details is not None
    assert generation.usage_details["input"] == INPUT_TOKENS
    assert generation.usage_details["output"] == OUTPUT_TOKENS
    assert generation.usage_details["total"] == INPUT_TOKENS + OUTPUT_TOKENS
    assert generation.cost_details is not None
    assert generation.cost_details["input"] == pytest.approx(cost.input_usd)
    assert generation.cost_details["output"] == pytest.approx(cost.output_usd)
    assert generation.cost_details["total"] == pytest.approx(cost.total_usd)
    assert generation.total_cost == pytest.approx(cost.total_usd)
