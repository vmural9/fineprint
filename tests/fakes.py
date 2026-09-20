"""Stand-ins for the two Bedrock models, so the unit tests never call AWS.

Both satisfy the protocols in `fineprint.providers.base`, so anything written against those
protocols — ingestion, retrieval, answering — can be tested with these in place of the real
models. Import them as `from tests.fakes import FakeChatModel, FakeEmbedder`.
"""

import hashlib
import math
import random
from typing import NamedTuple

from pydantic import BaseModel

from fineprint.providers.base import LLMResult


class FakeEmbedder:
    """Vectors made from a hash of the text: no network, and the same text always wins the
    same vector, which is what makes ingestion and retrieval tests repeatable."""

    dimension = 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        numbers = random.Random(seed)
        values = [numbers.uniform(-1.0, 1.0) for _ in range(self.dimension)]
        # Unit length, like Titan's normalized vectors, so cosine distances look real.
        length = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / length for value in values]


class FakeCall(NamedTuple):
    """One call made to `FakeChatModel`, kept so a test can inspect the prompt."""

    system: str
    user: str
    schema: type[BaseModel]


class FakeChatModel:
    """Hands back one canned object instead of asking a model anything."""

    model = "fake-chat-model"

    def __init__(self, response: BaseModel, *, input_tokens: int = 900, output_tokens: int = 120):
        self.response = response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[FakeCall] = []

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        self.calls.append(FakeCall(system, user, schema))
        if not isinstance(self.response, schema):
            raise TypeError(
                f"this FakeChatModel was given a {type(self.response).__name__} but was asked "
                f"for a {schema.__name__}"
            )
        return LLMResult(
            parsed=self.response,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=12.5,
        )
