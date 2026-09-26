"""What the service needs from a model, and nothing more.

Three protocols, so the rest of the code depends on these four methods rather than on AWS.
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


@dataclass(frozen=True)
class RerankResult:
    """Where one passage landed when a re-ranker read it beside the query.

    `index` points back into the list of passages that was passed in, so the caller keeps its
    own objects (a chunk's id, its pages) and takes only the new order and the score.
    """

    index: int  # position in the documents list that was passed in
    score: float  # the model's relevance score, higher is better


class Reranker(Protocol):
    """Puts passages in order of how well each one answers a query.

    Retrieval finds its candidates by shared words and by nearby vectors, and neither reads the
    question and a passage together. A re-ranker does, so it can move the passage that answers
    the question above one that only talks about the same thing. `rerank` returns at most
    `top_n` results, the best first, and never names the same passage twice.
    """

    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]: ...
