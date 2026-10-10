"""Prices and the arithmetic built on them.

One test pins the four list prices. Every other test passes its own small table as `prices=`,
so a price change never breaks an unrelated test.
"""

import pytest

from fineprint.pricing import (
    PRICES,
    Cost,
    Price,
    UnknownPriceError,
    embedding_cost,
    generation_cost,
    rerank_cost,
    rerank_units,
)

BEDROCK_FOUNDATION_MODELS_FILE = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/"
    "20261008235637/us-west-2/index.json"
)
BEDROCK_FILE = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/"
    "20261006144726/us-west-2/index.json"
)

# A table of round numbers, so the expected results can be checked by hand.
TABLE = {
    "chat": Price("chat", 2.0, 10.0, None, "tokens", "https://example.test/chat", "2026-01-01"),
    "embed": Price("embed", 0.5, None, None, "tokens", "https://example.test/e", "2026-01-01"),
    "rank": Price("rank", None, None, 0.25, "search unit", "https://example.test/r", "2026-01-01"),
}


def test_the_four_list_prices_are_pinned():
    assert set(PRICES) == {
        "us.anthropic.claude-opus-5",
        "us.anthropic.claude-sonnet-5",
        "amazon.titan-embed-text-v2:0",
        "cohere.rerank-v3-5:0",
    }
    opus = PRICES["us.anthropic.claude-opus-5"]
    assert (opus.input_per_million, opus.output_per_million) == (5.50, 27.50)
    sonnet = PRICES["us.anthropic.claude-sonnet-5"]
    assert (sonnet.input_per_million, sonnet.output_per_million) == (2.20, 11.00)
    titan = PRICES["amazon.titan-embed-text-v2:0"]
    assert (titan.input_per_million, titan.output_per_million) == (0.02, None)
    rerank = PRICES["cohere.rerank-v3-5:0"]
    assert (rerank.per_unit, rerank.unit) == (0.002, "search unit")


def test_every_price_names_its_source_and_the_day_it_was_read():
    for model, price in PRICES.items():
        assert price.model == model
        assert price.retrieved_on == "2026-10-10"
        assert price.source_url.startswith("https://pricing.us-east-1.amazonaws.com/offers/")
    assert PRICES["amazon.titan-embed-text-v2:0"].source_url == BEDROCK_FILE
    for model in (
        "us.anthropic.claude-opus-5",
        "us.anthropic.claude-sonnet-5",
        "cohere.rerank-v3-5:0",
    ):
        assert PRICES[model].source_url == BEDROCK_FOUNDATION_MODELS_FILE


def test_generation_cost_prices_input_and_output_tokens_separately():
    cost = generation_cost("chat", 500_000, 100_000, prices=TABLE)

    assert cost == Cost(input_usd=1.0, output_usd=1.0)
    assert cost.total_usd == 2.0


def test_generation_cost_of_no_tokens_is_zero():
    assert generation_cost("chat", 0, 0, prices=TABLE).total_usd == 0.0


def test_embedding_cost_has_only_an_input_part():
    cost = embedding_cost("embed", 2_000_000, prices=TABLE)

    assert cost == Cost(input_usd=1.0, output_usd=0.0)


def test_rerank_cost_is_units_times_the_unit_price_and_all_of_it_is_input():
    cost = rerank_cost("rank", 3, prices=TABLE)

    assert cost == Cost(input_usd=0.75, output_usd=0.0)


@pytest.mark.parametrize(
    ("documents", "units"), [(0, 0), (1, 1), (20, 1), (100, 1), (101, 2), (200, 2), (201, 3)]
)
def test_rerank_units_is_one_per_hundred_documents_rounded_up(documents, units):
    assert rerank_units(documents) == units


def test_an_unknown_model_raises_and_names_the_models_it_knows():
    with pytest.raises(UnknownPriceError) as excinfo:
        generation_cost("mystery", 1, 1, prices=TABLE)

    assert "mystery" in str(excinfo.value)
    assert "chat" in str(excinfo.value)
    assert isinstance(excinfo.value, LookupError)


def test_a_model_priced_the_wrong_way_raises_rather_than_costing_zero():
    with pytest.raises(UnknownPriceError):
        generation_cost("rank", 1, 1, prices=TABLE)  # priced per search unit, not per token
    with pytest.raises(UnknownPriceError):
        embedding_cost("rank", 1, prices=TABLE)
    with pytest.raises(UnknownPriceError):
        rerank_cost("chat", 1, prices=TABLE)  # priced per token, not per search unit
