"""What the service needs from a model, and nothing more.

Two protocols, so the rest of the code depends on these four methods rather than on AWS.
Anything with the right methods qualifies — the Bedrock classes next door, and the fakes in
`tests/fakes.py` that keep the unit tests offline.
"""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


class Embedder(Protocol):
    """Turns text into a vector of `dimension` numbers.

    Documents and queries get separate methods because many embedding models treat them
    differently; Titan does not, but retrieval code written against this protocol keeps
    working when a model that does arrives.
    """

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class LLMResult[T]:
    """One answer from a chat model, with what it cost to get it.

    Tokens and latency travel with the parsed object so part 3 can report cost and p95
    latency without changing any signature.
    """

    parsed: T
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ChatModel(Protocol):
    """Asks a model a question and insists on an answer shaped like `schema`."""

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]: ...
