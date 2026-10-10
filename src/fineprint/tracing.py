"""Seeing what one question did: which calls ran, how long each took, and what they cost.

A *trace* is the record of one question. It is made of *spans*, one per call that matters: the
search, the embedding of the question, the re-rank, the answer. Spans nest the way the calls do,
so a trace reads like the call stack: the retriever span holds the embedding span and the
re-rank span that happened inside it.

The `Tracer` does two things with every span. It keeps a local `TraceRecord`, so the cost and
the latency of a question are known even when nothing is sent anywhere. And it forwards the span
to a `Sink` if there is one: `langfuse_sink.py` is the sink that sends it to Langfuse. Nothing
else in the service imports Langfuse.

Spans are made by three small wrappers around the real models (`TracedChatModel`,
`TracedEmbedder`, `TracedReranker`), not inside the Bedrock classes, so the classes that talk to
AWS stay about AWS and the fakes in the tests stay untraced.

Cost is worked out when a call returns, from the tokens or search units it used, against the
list prices in `pricing.py`. A call whose price is not known has no cost, and a trace holding
one has no cost either: an unknown amount is never added up as zero.
"""

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel

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
from fineprint.providers.base import (
    ChatModel,
    Embedder,
    EmbeddingResult,
    LLMResult,
    Reranker,
    RerankResult,
    ToolCallingModel,
    ToolSpec,
    ToolTurn,
)

Kind = Literal["retriever", "embedding", "rerank", "generation", "tool", "agent"]

# The kinds of call that cost money. A call of one of these kinds with no cost makes the trace's
# total unknown; a tool or a search does not.
PRICED_KINDS: frozenset[str] = frozenset({"embedding", "rerank", "generation"})


@dataclass(frozen=True)
class CallRecord:
    """One finished call, as the results files and the scoreboard keep it."""

    kind: Kind
    name: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    units: int | None  # rerank search units, or tool calls
    latency_ms: float
    cost_usd: float | None
    error: str | None
    fallback: bool = False


@dataclass(frozen=True)
class TraceRecord:
    """One finished question: how long it took, what it cost, and the calls inside it.

    `trace_id` is the id the sink gave the trace, and None when there is no sink. `cost_usd` is
    the sum over the calls that have a cost, and None when a call that should have one does not.
    """

    trace_id: str | None
    name: str
    latency_ms: float
    cost_usd: float | None
    calls: list[CallRecord]


class Observation(Protocol):
    """What a sink hands back for one span, so the tracer can fill in what it learns later."""

    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        units: int | None = None,
        cost: Cost | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...


class Sink(Protocol):
    """Where finished spans go. `langfuse_sink.LangfuseSink` is the one real implementation."""

    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> AbstractContextManager[Observation]:
        """Open a trace; the observation it yields is the root, holding its input and output."""
        ...

    def observation(
        self, kind: Kind, name: str, *, input: Any = None, model: str | None = None
    ) -> AbstractContextManager[Observation]:
        """Open a span inside whichever trace or span is open."""
        ...

    def current_trace_id(self) -> str | None: ...

    def trace_url(self, trace_id: str) -> str | None:
        """A link to the trace in the sink's UI. May make a network call, so only the CLI asks."""
        ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


class _NoObservation:
    """The observation a tracer with no sink opens: it takes every update and keeps none."""

    def update(self, **fields: Any) -> None:
        return None


def describe_error(error: BaseException) -> str:
    """The class name and the message, which is what a person reading a trace needs."""
    return f"{type(error).__name__}: {error}"


def now_ms(since: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading."""
    return (time.perf_counter() - since) * 1000


def total_cost(calls: Sequence[CallRecord]) -> float | None:
    """The cost of a set of calls: None when a call that should have a price has none.

    A call that failed is left out of that check. It has no token counts, so it has no cost,
    and holding that against the trace would make every trace with a fallback in it unknown.
    """
    for call in calls:
        if call.kind in PRICED_KINDS and call.cost_usd is None and call.error is None:
            return None
    return sum((call.cost_usd for call in calls if call.cost_usd is not None), 0.0)


class Span:
    """One call in progress. Call `end()` with what the call learned; leaving the block ends it."""

    def __init__(
        self,
        trace: "Trace",
        kind: Kind,
        name: str,
        model: str | None,
        observation: Observation,
    ) -> None:
        self._trace = trace
        self._kind = kind
        self._name = name
        self._model = model
        self._observation = observation
        self._started = time.perf_counter()
        self.ended = False

    def end(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        units: int | None = None,
        cost: Cost | None = None,
        error: BaseException | None = None,
        fallback: bool = False,
    ) -> None:
        """Record the call: once. A second `end()` on the same span is ignored."""
        if self.ended:
            return
        self.ended = True
        model = model if model is not None else self._model
        message = describe_error(error) if error is not None else None
        self._observation.update(
            output=output,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            units=units,
            cost=cost,
            error=message,
            metadata={"fallback": True} if fallback else None,
        )
        self._trace.calls.append(
            CallRecord(
                kind=self._kind,
                name=self._name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                units=units,
                latency_ms=now_ms(self._started),
                cost_usd=cost.total_usd if cost is not None else None,
                error=message,
                fallback=fallback,
            )
        )


class Trace:
    """One question in progress. Its `record` is complete once the `with` block has exited."""

    def __init__(
        self, name: str, root: Observation, trace_id: str | None, tracer: "Tracer"
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.calls: list[CallRecord] = []
        self._root = root
        self._tracer = tracer
        self._started = time.perf_counter()
        self._latency_ms: float | None = None

    def set_output(self, output: Any) -> None:
        """Set what the trace shows as its result, for example the answer's text."""
        self._root.update(output=output)

    def _finish(self) -> None:
        self._latency_ms = now_ms(self._started)

    @property
    def record(self) -> TraceRecord:
        """The trace so far, or all of it once the block has exited."""
        latency = self._latency_ms if self._latency_ms is not None else now_ms(self._started)
        calls = list(self.calls)
        return TraceRecord(
            trace_id=self.trace_id,
            name=self.name,
            latency_ms=latency,
            cost_usd=total_cost(calls),
            calls=calls,
        )


# The trace that is open in this thread of work. A contextvar, so a request handled in another
# thread or task does not see this one's trace.
_ACTIVE: ContextVar[Trace | None] = ContextVar("fineprint_active_trace", default=None)


class Tracer:
    """Opens traces and spans, keeps their records, and forwards them to a sink if there is one."""

    def __init__(self, sink: Sink | None = None) -> None:
        self.sink = sink

    def active_trace(self) -> Trace | None:
        """The trace this tracer has open right now, or None."""
        active = _ACTIVE.get()
        return active if active is not None and active._tracer is self else None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Trace]:
        """Open a trace around one question.

        Inside an open trace this joins it rather than starting a second one, so a caller that
        wants a trace of its own (the API, the CLI, the eval runner) and a function that wants
        one when nobody else opened it (`answer_question`) can both ask without coordinating.
        The first caller's name, tags and session win.
        """
        active = self.active_trace()
        if active is not None:
            yield active
            return

        # Short strings only: Langfuse coerces metadata values to strings and drops long ones.
        sink_metadata = {key: str(value) for key, value in (metadata or {}).items()}
        opened: AbstractContextManager[Observation] = (
            self.sink.trace(
                name, input=input, session_id=session_id, tags=tags, metadata=sink_metadata
            )
            if self.sink is not None
            else nullcontext(_NoObservation())
        )
        with opened as root:
            trace = Trace(
                name,
                root,
                self.sink.current_trace_id() if self.sink is not None else None,
                self,
            )
            token = _ACTIVE.set(trace)
            try:
                yield trace
            except BaseException as error:
                root.update(error=describe_error(error))
                raise
            finally:
                _ACTIVE.reset(token)
                trace._finish()

    @contextmanager
    def span(
        self, kind: Kind, name: str, *, input: Any = None, model: str | None = None
    ) -> Iterator[Span]:
        """Open a span inside the open trace, or inside a trace of its own if none is open."""
        trace = self.active_trace()
        if trace is None:
            with self.trace(name), self.span(kind, name, input=input, model=model) as span:
                yield span
            return

        opened: AbstractContextManager[Observation] = (
            self.sink.observation(kind, name, input=input, model=model)
            if self.sink is not None
            else nullcontext(_NoObservation())
        )
        with opened as observation:
            span = Span(trace, kind, name, model, observation)
            try:
                yield span
            except BaseException as error:
                span.end(error=error)
                raise
            else:
                span.end()

    def trace_url(self, trace_id: str) -> str | None:
        """A link to the trace in the sink's UI, or None without a sink."""
        return self.sink.trace_url(trace_id) if self.sink is not None else None

    def flush(self) -> None:
        """Send whatever the sink is holding. A service calls this when it is about to idle."""
        if self.sink is not None:
            self.sink.flush()

    def shutdown(self) -> None:
        """Flush and stop the sink's background work. A process calls this before it exits."""
        if self.sink is not None:
            self.sink.shutdown()


def _price_or_none(compute: Any, *args: Any, prices: Mapping[str, Price]) -> Cost | None:
    """`compute(*args, prices=prices)`, or None when the table has no usable price."""
    try:
        return compute(*args, prices=prices)
    except UnknownPriceError:
        return None


def _jsonable(value: Any) -> Any:
    """A pydantic model as plain data for a trace; anything else as it is."""
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


class TracedChatModel:
    """Wraps a chat model so each call is one "generation" span, priced from its tokens.

    Works for both ways of calling a model: `complete_structured` for the answer and
    `complete_with_tools` for the agent's turns. The span's input is the prompt the model saw,
    as a list of messages, so the Langfuse UI shows it as a conversation.
    """

    def __init__(
        self,
        inner: ChatModel | ToolCallingModel,
        tracer: Tracer,
        *,
        name: str = "answer",
        prices: Mapping[str, Price] = PRICES,
    ) -> None:
        self._inner = inner
        self._tracer = tracer
        self._name = name
        self._prices = prices
        self.model = getattr(inner, "model", None)

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with self._tracer.span("generation", self._name, input=messages, model=self.model) as span:
            result = self._inner.complete_structured(system, user, schema)
            span.end(
                output=_jsonable(result.parsed),
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=_price_or_none(
                    generation_cost,
                    result.model,
                    result.input_tokens,
                    result.output_tokens,
                    prices=self._prices,
                ),
                fallback=result.fallback_used,
            )
            return result

    def complete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec],
        *,
        force_tool: str | None = None,
    ) -> ToolTurn:
        call = (
            self._inner.complete_with_tools
        )  # an inner model without it fails here, as it would bare
        shown = [{"role": "system", "content": system}, *messages]
        with self._tracer.span("generation", self._name, input=shown, model=self.model) as span:
            turn = call(system, messages, tools, force_tool=force_tool)
            span.end(
                output=[
                    {"name": made.name, "arguments": made.arguments} for made in turn.tool_calls
                ],
                model=turn.model,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                cost=_price_or_none(
                    generation_cost,
                    turn.model,
                    turn.input_tokens,
                    turn.output_tokens,
                    prices=self._prices,
                ),
            )
            return turn


class TracedEmbedder:
    """Wraps an embedder so each call is one "embedding" span, priced from Titan's token count."""

    def __init__(
        self, inner: Embedder, tracer: Tracer, *, prices: Mapping[str, Price] = PRICES
    ) -> None:
        self._inner = inner
        self._tracer = tracer
        self._prices = prices
        self.model = getattr(inner, "model", None)
        self.dimension = inner.dimension

    def embed(self, texts: list[str]) -> EmbeddingResult:
        return self._embed(texts, "embed")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "embed_documents").vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "embed_query").vectors[0]

    def _embed(self, texts: list[str], name: str) -> EmbeddingResult:
        shown = texts[0] if len(texts) == 1 else f"{len(texts)} texts"
        with self._tracer.span("embedding", name, input=shown, model=self.model) as span:
            result = self._inner.embed(texts)
            cost = None
            if result.input_tokens is not None and self.model is not None:
                cost = _price_or_none(
                    embedding_cost, self.model, result.input_tokens, prices=self._prices
                )
            span.end(input_tokens=result.input_tokens, cost=cost)
            return result


class TracedReranker:
    """Wraps a re-ranker so each call is one "rerank" span, billed in search units."""

    def __init__(
        self, inner: Reranker, tracer: Tracer, *, prices: Mapping[str, Price] = PRICES
    ) -> None:
        self._inner = inner
        self._tracer = tracer
        self._prices = prices
        self.model = inner.model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        shown = {"query": query, "documents": len(documents), "top_n": top_n}
        with self._tracer.span("rerank", "rerank", input=shown, model=self.model) as span:
            results = self._inner.rerank(query, documents, top_n)
            units = rerank_units(len(documents))
            span.end(
                output=[{"index": result.index, "score": result.score} for result in results],
                units=units,
                cost=_price_or_none(rerank_cost, self.model, units, prices=self._prices),
            )
            return results
