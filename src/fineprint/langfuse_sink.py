"""Sending traces to Langfuse.

This is the only module in the service that imports `langfuse`. It turns the tracer's spans into
Langfuse observations, so everything else talks to the small `Sink` protocol in `tracing.py` and
a test can swap Langfuse for a recording stand-in.

Three things about the Langfuse 4 SDK shape this file:

- A trace has no object of its own. Its name, session, tags and metadata are set with
  `propagate_attributes`, opened around the root observation, and the trace's input and output
  are the root observation's. Metadata values must be short strings (the SDK coerces them to
  strings and drops any over 200 characters), so the tracer passes only that.
- `usage_details` and `cost_details` are free dictionaries; the SDK fixes no key names. The keys
  below are the ones its own OpenAI integration writes, so Langfuse's views show them as input,
  output and total tokens, and as the cost of the observation.
- Cost is sent as computed from `pricing.py`. Langfuse's own price table is never consulted for
  these models.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from langfuse import Langfuse, propagate_attributes

from fineprint.pricing import Cost
from fineprint.tracing import Kind

# The Langfuse observation type each kind of span becomes. A re-rank is sent as a generation so
# that Langfuse's cost views, which add up generation-like observations, include it.
KIND_TO_AS_TYPE: dict[str, str] = {
    "retriever": "retriever",
    "embedding": "embedding",
    "rerank": "generation",
    "generation": "generation",
    "tool": "tool",
    "agent": "agent",
}

# The observation types that carry a model, usage and cost.
GENERATION_LIKE = frozenset({"generation", "embedding"})

USAGE_KEYS = ("input", "output", "total")  # usage_details for generations and embeddings
RERANK_USAGE_KEY = "search_units"  # usage_details for a re-rank: {"search_units": n}
COST_KEYS = ("input", "output", "total")  # cost_details of every priced observation

# Langfuse drops a metadata value longer than this.
MAX_METADATA_CHARACTERS = 200


class _LangfuseObservation:
    """What the tracer updates: turns its fields into the arguments of Langfuse's `update()`."""

    def __init__(self, observation: Any, kind: str | None) -> None:
        self._observation = observation
        self._kind = kind  # None for the root of a trace
        self._generation_like = kind is not None and KIND_TO_AS_TYPE[kind] in GENERATION_LIKE

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
    ) -> None:
        fields: dict[str, Any] = {}
        if output is not None:
            fields["output"] = output
        if self._generation_like:
            if model is not None:
                fields["model"] = model
            usage = self._usage(input_tokens, output_tokens, units)
            if usage:
                fields["usage_details"] = usage
            if cost is not None:
                fields["cost_details"] = dict(
                    zip(
                        COST_KEYS,
                        (cost.input_usd, cost.output_usd, cost.total_usd),
                        strict=True,
                    )
                )
        if error is not None:
            fields["level"] = "ERROR"
            fields["status_message"] = error
        if metadata:
            fields["metadata"] = dict(metadata)
        if fields:
            self._observation.update(**fields)

    def _usage(
        self, input_tokens: int | None, output_tokens: int | None, units: int | None
    ) -> dict[str, int]:
        if self._kind == "rerank":
            return {RERANK_USAGE_KEY: units} if units is not None else {}
        if input_tokens is None and output_tokens is None:
            return {}
        read, written = input_tokens or 0, output_tokens or 0
        return dict(zip(USAGE_KEYS, (read, written, read + written), strict=True))


class LangfuseSink:
    """A `Sink` that sends every trace and span to a Langfuse project.

    `client` and `propagate` can be replaced, which is how the tests see exactly what is sent
    without building a real client.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str,
        *,
        client: Any | None = None,
        propagate: Callable[..., AbstractContextManager[Any]] = propagate_attributes,
    ) -> None:
        self.client = (
            client
            if client is not None
            else Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
        )
        self._propagate = propagate

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> Iterator[_LangfuseObservation]:
        """Open the trace's attributes, and inside them its root observation."""
        short = (
            {key: value[:MAX_METADATA_CHARACTERS] for key, value in metadata.items()}
            if metadata
            else None
        )
        with (
            self._propagate(
                trace_name=name,
                session_id=session_id,
                tags=list(tags) or None,
                metadata=short,
            ),
            self.client.start_as_current_observation(
                as_type="span", name=name, input=input
            ) as root,
        ):
            yield _LangfuseObservation(root, None)

    @contextmanager
    def observation(
        self, kind: Kind, name: str, *, input: Any = None, model: str | None = None
    ) -> Iterator[_LangfuseObservation]:
        """Open one observation, a child of whichever is open. Only models are named up front."""
        as_type = KIND_TO_AS_TYPE[kind]
        arguments: dict[str, Any] = {"as_type": as_type, "name": name, "input": input}
        if as_type in GENERATION_LIKE and model is not None:
            arguments["model"] = model
        with self.client.start_as_current_observation(**arguments) as observation:
            yield _LangfuseObservation(observation, kind)

    def current_trace_id(self) -> str | None:
        return self.client.get_current_trace_id()

    def trace_url(self, trace_id: str) -> str | None:
        """The trace's page in the Langfuse UI. Looking up the project id is a network call."""
        try:
            return self.client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def flush(self) -> None:
        self.client.flush()

    def shutdown(self) -> None:
        self.client.shutdown()
