"""What a model call costs, from the list price and the units the call used.

Every figure below is a public Amazon Bedrock list price for us-west-2, read from the AWS Price
List bulk files on the date each entry names. The pricing web page fills its cells with
JavaScript, so a script cannot read it; the bulk files are what a script can check. The Claude
rows are the "Standard" on-demand rows, which is what the `us.` regional inference profiles this
service calls are billed at (the "Standard, Global" rows are 10% lower and belong to the
`global.` profiles). Prompt caching is not used, so the cache prices are absent.

Cost is worked out when a call returns, from token counts the provider reports, so a trace and an
eval result can both carry it. A price that changes is changed here and nowhere else.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

# One Rerank search unit covers up to this many passages.
RERANK_DOCUMENTS_PER_UNIT = 100


@dataclass(frozen=True)
class Price:
    """The list price of one model, and where it was read."""

    model: str
    input_per_million: float | None  # USD per million input tokens
    output_per_million: float | None  # USD per million output tokens
    per_unit: float | None  # USD per billed unit, for models billed per request (Rerank)
    unit: str  # "tokens" or "search unit"
    source_url: str  # the AWS price-list file the figure was read from
    retrieved_on: str  # YYYY-MM-DD


@dataclass(frozen=True)
class Cost:
    """What a call cost in US dollars, split the way the provider bills it."""

    input_usd: float
    output_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd


class UnknownPriceError(LookupError):
    """The price table has no entry for this model, or none of the kind the call needs."""


_FOUNDATION_MODELS_FILE = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/"
    "20261008235637/us-west-2/index.json"
)
_BEDROCK_FILE = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/"
    "20261006144726/us-west-2/index.json"
)
_READ_ON = "2026-10-10"

PRICES: Mapping[str, Price] = {
    "us.anthropic.claude-opus-5": Price(
        model="us.anthropic.claude-opus-5",
        input_per_million=5.50,
        output_per_million=27.50,
        per_unit=None,
        unit="tokens",
        source_url=_FOUNDATION_MODELS_FILE,
        retrieved_on=_READ_ON,
    ),
    "us.anthropic.claude-sonnet-5": Price(
        model="us.anthropic.claude-sonnet-5",
        input_per_million=2.20,
        output_per_million=11.00,
        per_unit=None,
        unit="tokens",
        source_url=_FOUNDATION_MODELS_FILE,
        retrieved_on=_READ_ON,
    ),
    # Listed as $0.00002 per 1,000 input tokens, which is $0.02 per million.
    "amazon.titan-embed-text-v2:0": Price(
        model="amazon.titan-embed-text-v2:0",
        input_per_million=0.02,
        output_per_million=None,
        per_unit=None,
        unit="tokens",
        source_url=_BEDROCK_FILE,
        retrieved_on=_READ_ON,
    ),
    "cohere.rerank-v3-5:0": Price(
        model="cohere.rerank-v3-5:0",
        input_per_million=None,
        output_per_million=None,
        per_unit=0.002,
        unit="search unit",
        source_url=_FOUNDATION_MODELS_FILE,
        retrieved_on=_READ_ON,
    ),
}


def _price(prices: Mapping[str, Price], model: str) -> Price:
    """The entry for `model`, or an error that lists the models the table does know."""
    try:
        return prices[model]
    except KeyError:
        raise UnknownPriceError(
            f"no price for {model!r}; the table has {', '.join(sorted(prices))}."
        ) from None


def generation_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: Mapping[str, Price] = PRICES,
) -> Cost:
    """The cost of one chat call, from its input and output token counts."""
    price = _price(prices, model)
    if price.input_per_million is None or price.output_per_million is None:
        raise UnknownPriceError(f"{model!r} has no per-token input and output price.")
    return Cost(
        input_usd=input_tokens * price.input_per_million / 1_000_000,
        output_usd=output_tokens * price.output_per_million / 1_000_000,
    )


def embedding_cost(model: str, input_tokens: int, *, prices: Mapping[str, Price] = PRICES) -> Cost:
    """The cost of embedding calls, from their input token count. Embeddings bill no output."""
    price = _price(prices, model)
    if price.input_per_million is None:
        raise UnknownPriceError(f"{model!r} has no per-token input price.")
    return Cost(input_usd=input_tokens * price.input_per_million / 1_000_000, output_usd=0.0)


def rerank_units(documents: int) -> int:
    """Search units one re-rank call is billed for: one per 100 passages, rounded up."""
    return math.ceil(documents / RERANK_DOCUMENTS_PER_UNIT) if documents > 0 else 0


def rerank_cost(model: str, units: int, *, prices: Mapping[str, Price] = PRICES) -> Cost:
    """The cost of `units` search units, all of it booked as input."""
    price = _price(prices, model)
    if price.per_unit is None:
        raise UnknownPriceError(f"{model!r} has no per-unit price.")
    return Cost(input_usd=units * price.per_unit, output_usd=0.0)
