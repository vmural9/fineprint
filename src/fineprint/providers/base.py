"""What the service needs from a model, and nothing more.

Four protocols, so the rest of the code depends on these methods rather than on AWS. Anything
with the right methods qualifies — the Bedrock classes next door, and the fakes in
`tests/fakes.py` that keep the unit tests offline.

The `ProviderError` family also lives here. A provider turns its SDK's failures into one of
three of them, so the code above can tell "try again or fall back" from "this request is wrong"
without knowing which SDK raised.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel


class ProviderError(RuntimeError):
    """A model provider failed in a way the service may act on."""


class ProviderTimeoutError(ProviderError):
    """The provider's SDK gave up waiting for a reply, after its own retries."""


class ProviderUnavailableError(ProviderError):
    """The provider could not serve the request: throttled, overloaded, down, or unreachable."""


class ProviderRequestError(ProviderError):
    """The provider refused the request itself (a 4xx other than 429). Retrying cannot help."""


@dataclass(frozen=True)
class EmbeddingResult:
    """Vectors for a list of texts, with what embedding them used."""

    vectors: list[list[float]]
    input_tokens: int | None  # Titan reports it; None when a provider does not
    latency_ms: float


class Embedder(Protocol):
    """Turns text into a vector of `dimension` numbers.

    Documents and queries get separate methods because many embedding models treat them
    differently; Titan does not, but retrieval code written against this protocol keeps
    working when a model that does arrives. Both are the same call as `embed`, keeping only
    the vectors; `embed` also reports the tokens read and the time taken.
    """

    dimension: int

    def embed(self, texts: list[str]) -> EmbeddingResult: ...

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
    # True when the model that answered is the fallback, not the one that was configured.
    fallback_used: bool = False


class ChatModel(Protocol):
    """Asks a model a question and insists on an answer shaped like `schema`."""

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]: ...


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: its name, what it does, and the JSON schema of its input."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One call the model made to a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolTurn:
    """One reply from a model that was offered tools, with what it cost to get it."""

    tool_calls: list[ToolCall]  # never empty when a tool call was demanded and honoured
    content: Any  # the assistant content blocks as returned, echoed back on the next turn
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None
    # True when the model that answered is the fallback, not the one that was configured.
    fallback_used: bool = False


class ToolCallingModel(Protocol):
    """Offers a model some tools and returns the calls it makes.

    `messages` are in the Anthropic Messages shape (a role and a list of content blocks).
    `force_tool` names the one tool the model must call; None means it must call some tool.
    """

    def complete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec],
        *,
        force_tool: str | None = None,
    ) -> ToolTurn: ...


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
