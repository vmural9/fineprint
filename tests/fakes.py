"""Stand-ins for the Bedrock models, so the unit tests never call AWS.

Each one satisfies a protocol in `fineprint.providers.base`, so anything written against those
protocols — ingestion, retrieval, answering, the eval metrics — can be tested with these in place
of the real models:

- `FakeEmbedder` makes a vector from a hash of the text, so the same text always gets the same
  vector.
- `FakeChatModel` hands back one canned object, whatever it is asked.
- `ScriptedChatModel` hands back canned objects in order, from a separate queue for each schema,
  for code that asks a model several different things, or asks the same thing twice.
- `FakeReranker` puts passages in order of how many of the query's words each one contains.
- `RecordingSink` is a tracing sink that keeps every trace and observation it is handed, with
  the updates made to each, so a test can read back what was traced without Langfuse.

Import them as
`from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker, ScriptedChatModel`.
"""

import hashlib
import math
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from pydantic import BaseModel

from fineprint.providers.base import EmbeddingResult, LLMResult, RerankResult


class FakeEmbedder:
    """Vectors made from a hash of the text: no network, and the same text always wins the
    same vector, which is what makes ingestion and retrieval tests repeatable."""

    dimension = 1024

    def embed(self, texts: list[str]) -> EmbeddingResult:
        # The word count stands in for Titan's token count, so tests can assert cost arithmetic.
        return EmbeddingResult(
            vectors=[self.embed_query(text) for text in texts],
            input_tokens=sum(len(text.split()) for text in texts),
            latency_ms=0.0,
        )

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
    """One call made to a fake chat model, kept so a test can inspect the prompt."""

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


class ScriptedChatModel:
    """Hands back canned objects in order, from a separate queue for each schema.

    `ScriptedChatModel({SentenceVerdicts: [first, second], AnswerClaims: [claims]})` answers the
    first request for a `SentenceVerdicts` with `first` and the next with `second`, whatever
    other schemas are asked for in between. That is what code making several different calls,
    or retrying one, needs in order to be tested call by call. A request the script has no
    object left for fails the test, so an unexpected extra call cannot pass unnoticed.
    """

    model = "scripted-chat-model"

    def __init__(
        self,
        script: dict[type[BaseModel], list[BaseModel]],
        *,
        input_tokens: int = 900,
        output_tokens: int = 120,
    ):
        for schema, responses in script.items():
            for response in responses:
                if not isinstance(response, schema):
                    raise TypeError(
                        f"the script files a {type(response).__name__} under {schema.__name__}"
                    )
        # Copies, so popping from them leaves the test's own lists as they were.
        self.script = {schema: list(responses) for schema, responses in script.items()}
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[FakeCall] = []

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        self.calls.append(FakeCall(system, user, schema))
        queue = self.script.get(schema)
        if not queue:
            raise AssertionError(
                f"ScriptedChatModel was asked for a {schema.__name__}, and its script has none "
                "left to hand back"
            )
        return LLMResult(
            parsed=queue.pop(0),
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=12.5,
        )


def _words(text: str) -> set[str]:
    """The distinct words in `text`, lower-cased, so "Premium" and "premium" are one word."""
    return set(re.findall(r"\w+", text.lower()))


class FakeRerankCall(NamedTuple):
    """One call made to `FakeReranker`, kept so a test can see what it was asked to rank."""

    query: str
    documents: list[str]
    top_n: int


class FakeReranker:
    """Ranks passages by how many of the query's words each one contains.

    Whole words, ignoring case, each query word counted once; a tie goes to the passage that
    came first. Crude next to a real re-ranker, but it needs no network and a test can predict
    its order exactly.
    """

    model = "fake-reranker"

    def __init__(self) -> None:
        self.calls: list[FakeRerankCall] = []

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        self.calls.append(FakeRerankCall(query, list(documents), top_n))
        query_words = _words(query)
        scores = [len(query_words & _words(document)) for document in documents]
        # Sorting on (-score, index) puts the highest score first and breaks a tie by position.
        order = sorted(range(len(documents)), key=lambda index: (-scores[index], index))
        return [RerankResult(index=index, score=float(scores[index])) for index in order[:top_n]]


@dataclass
class RecordedObservation:
    """One trace root or one observation a `RecordingSink` was asked to open.

    `fields` merges every update made to it, so a test reads `observation.fields["output"]`
    rather than walking the list; a later update wins over an earlier one for the same field.
    """

    kind: str  # "trace" for a trace's root observation
    name: str
    input: Any = None
    model: str | None = None
    parent: "RecordedObservation | None" = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    updates: list[dict[str, Any]] = field(default_factory=list)
    children: list["RecordedObservation"] = field(default_factory=list)

    def update(self, **changes: Any) -> None:
        self.updates.append(changes)

    @property
    def fields(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for changes in self.updates:
            merged.update({key: value for key, value in changes.items() if value is not None})
        return merged


class RecordingSink:
    """A tracing `Sink` that writes nothing anywhere and remembers everything.

    `traces` holds each trace's root, `observations` every other observation in the order it
    was opened, and each one's `parent` and `children` give the nesting. Every trace gets
    `trace_id` as its id, so a test can assert on the id that comes back in a response.
    """

    def __init__(self, trace_id: str = "trace-1") -> None:
        self.trace_id = trace_id
        self.traces: list[RecordedObservation] = []
        self.observations: list[RecordedObservation] = []
        self.flushes = 0
        self.shutdowns = 0
        self._stack: list[RecordedObservation] = []

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> Iterator[RecordedObservation]:
        root = RecordedObservation(
            kind="trace",
            name=name,
            input=input,
            session_id=session_id,
            tags=list(tags),
            metadata=dict(metadata or {}),
        )
        self.traces.append(root)
        self._stack.append(root)
        try:
            yield root
        finally:
            self._stack.pop()

    @contextmanager
    def observation(
        self, kind: str, name: str, *, input: Any = None, model: str | None = None
    ) -> Iterator[RecordedObservation]:
        parent = self._stack[-1] if self._stack else None
        observation = RecordedObservation(
            kind=kind, name=name, input=input, model=model, parent=parent
        )
        if parent is not None:
            parent.children.append(observation)
        self.observations.append(observation)
        self._stack.append(observation)
        try:
            yield observation
        finally:
            self._stack.pop()

    def current_trace_id(self) -> str | None:
        return self.trace_id if self._stack else None

    def trace_url(self, trace_id: str) -> str | None:
        return f"https://tracing.example/traces/{trace_id}"

    def flush(self) -> None:
        self.flushes += 1

    def shutdown(self) -> None:
        self.shutdowns += 1
