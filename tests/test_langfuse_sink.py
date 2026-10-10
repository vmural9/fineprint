"""Tests for the Langfuse sink, with a stub standing in for the Langfuse client.

The stub records the exact calls the sink makes, which is the contract with Langfuse: the
observation type of each kind of span, and the keys of the usage and cost dictionaries. Nothing
here builds a real `Langfuse` object or touches the network; `test_live_langfuse.py` does that.
"""

from contextlib import contextmanager
from typing import Any

import pytest

from fineprint.langfuse_sink import (
    COST_KEYS,
    KIND_TO_AS_TYPE,
    RERANK_USAGE_KEY,
    USAGE_KEYS,
    LangfuseSink,
)
from fineprint.pricing import Cost
from fineprint.tracing import Tracer


class StubObservation:
    def __init__(self, log: list[tuple[str, dict[str, Any]]]) -> None:
        self.log = log

    def update(self, **fields: Any) -> None:
        self.log.append(("update", fields))


class StubClient:
    """Records every call; nothing is sent anywhere."""

    def __init__(self) -> None:
        self.log: list[tuple[str, dict[str, Any]]] = []
        self.trace_id: str | None = "0123456789abcdef0123456789abcdef"
        self.url_error: Exception | None = None

    @contextmanager
    def start_as_current_observation(self, **arguments: Any):
        self.log.append(("start", arguments))
        yield StubObservation(self.log)

    def get_current_trace_id(self) -> str | None:
        return self.trace_id

    def get_trace_url(self, *, trace_id: str) -> str | None:
        if self.url_error is not None:
            raise self.url_error
        return f"https://langfuse.example/project/p1/traces/{trace_id}"

    def flush(self) -> None:
        self.log.append(("flush", {}))

    def shutdown(self) -> None:
        self.log.append(("shutdown", {}))

    def starts(self) -> list[dict[str, Any]]:
        return [arguments for name, arguments in self.log if name == "start"]

    def updates(self) -> list[dict[str, Any]]:
        return [arguments for name, arguments in self.log if name == "update"]


@pytest.fixture
def client() -> StubClient:
    return StubClient()


@pytest.fixture
def propagated(client: StubClient) -> list[dict[str, Any]]:
    return []


@pytest.fixture
def sink(client: StubClient, propagated: list[dict[str, Any]]) -> LangfuseSink:
    @contextmanager
    def propagate_attributes(**attributes: Any):
        propagated.append(attributes)
        client.log.append(("propagate", attributes))
        yield

    return LangfuseSink(
        "pk-test",
        "sk-test",
        "https://langfuse.example",
        client=client,
        propagate=propagate_attributes,
    )


def test_the_constants_name_the_contract():
    assert KIND_TO_AS_TYPE == {
        "retriever": "retriever",
        "embedding": "embedding",
        "rerank": "generation",
        "generation": "generation",
        "tool": "tool",
        "agent": "agent",
    }
    assert USAGE_KEYS == ("input", "output", "total")
    assert COST_KEYS == ("input", "output", "total")
    assert RERANK_USAGE_KEY == "search_units"


def test_without_a_client_it_builds_one_from_the_three_values(monkeypatch):
    built: list[dict[str, Any]] = []

    class Built:
        def __init__(self, **arguments: Any) -> None:
            built.append(arguments)

    monkeypatch.setattr("fineprint.langfuse_sink.Langfuse", Built)

    sink = LangfuseSink("pk-test", "sk-test", "https://langfuse.example")

    assert built == [
        {"public_key": "pk-test", "secret_key": "sk-test", "base_url": "https://langfuse.example"}
    ]
    assert isinstance(sink.client, Built)


def test_a_trace_sets_its_attributes_and_opens_the_root_span_inside_them(
    sink: LangfuseSink, client: StubClient, propagated: list[dict[str, Any]]
):
    with sink.trace(
        "ask",
        input="How much is the Part B premium?",
        session_id="run-1",
        tags=["eval", "sections"],
        metadata={"question_id": "q001"},
    ):
        pass

    assert propagated == [
        {
            "trace_name": "ask",
            "session_id": "run-1",
            "tags": ["eval", "sections"],
            "metadata": {"question_id": "q001"},
        }
    ]
    assert client.starts() == [
        {"as_type": "span", "name": "ask", "input": "How much is the Part B premium?"}
    ]
    assert [name for name, _ in client.log] == ["propagate", "start"]


def test_a_trace_without_attributes_passes_none_rather_than_empty_values(
    sink: LangfuseSink, propagated: list[dict[str, Any]]
):
    with sink.trace("search"):
        pass

    assert propagated == [
        {"trace_name": "search", "session_id": None, "tags": None, "metadata": None}
    ]


def test_metadata_values_are_cut_to_the_length_langfuse_accepts(
    sink: LangfuseSink, propagated: list[dict[str, Any]]
):
    with sink.trace("ask", metadata={"note": "x" * 500}):
        pass

    assert propagated[0]["metadata"] == {"note": "x" * 200}


@pytest.mark.parametrize(
    ("kind", "as_type", "takes_model"),
    [
        ("retriever", "retriever", False),
        ("embedding", "embedding", True),
        ("rerank", "generation", True),
        ("generation", "generation", True),
        ("tool", "tool", False),
        ("agent", "agent", False),
    ],
)
def test_each_kind_opens_the_observation_type_the_contract_names(
    sink: LangfuseSink, client: StubClient, kind: str, as_type: str, takes_model: bool
):
    with sink.observation(kind, "step", input="in", model="some-model"):  # type: ignore[arg-type]
        pass

    expected = {"as_type": as_type, "name": "step", "input": "in"}
    if takes_model:
        expected["model"] = "some-model"
    assert client.starts() == [expected]


def test_a_generation_without_a_model_is_opened_without_one(sink: LangfuseSink, client: StubClient):
    with sink.observation("generation", "answer", input="in"):
        pass

    assert client.starts() == [{"as_type": "generation", "name": "answer", "input": "in"}]


def test_a_generation_update_sends_usage_and_cost_with_the_contract_keys(
    sink: LangfuseSink, client: StubClient
):
    with sink.observation("generation", "answer") as observation:
        observation.update(
            output={"answer": "$202.90"},
            model="us.anthropic.claude-opus-5",
            input_tokens=1000,
            output_tokens=200,
            cost=Cost(input_usd=0.0055, output_usd=0.0055),
        )

    assert client.updates() == [
        {
            "output": {"answer": "$202.90"},
            "model": "us.anthropic.claude-opus-5",
            "usage_details": {"input": 1000, "output": 200, "total": 1200},
            "cost_details": {"input": 0.0055, "output": 0.0055, "total": 0.011},
        }
    ]


def test_an_embedding_update_sends_input_tokens_and_a_zero_output(
    sink: LangfuseSink, client: StubClient
):
    with sink.observation("embedding", "embed_query") as observation:
        observation.update(
            model="amazon.titan-embed-text-v2:0",
            input_tokens=13,
            cost=Cost(input_usd=0.00000026, output_usd=0.0),
        )

    assert client.updates() == [
        {
            "model": "amazon.titan-embed-text-v2:0",
            "usage_details": {"input": 13, "output": 0, "total": 13},
            "cost_details": {"input": 0.00000026, "output": 0.0, "total": 0.00000026},
        }
    ]


def test_a_rerank_update_sends_search_units_and_the_cost(sink: LangfuseSink, client: StubClient):
    with sink.observation("rerank", "rerank") as observation:
        observation.update(
            model="cohere.rerank-v3-5:0", units=1, cost=Cost(input_usd=0.002, output_usd=0.0)
        )

    assert client.updates() == [
        {
            "model": "cohere.rerank-v3-5:0",
            "usage_details": {"search_units": 1},
            "cost_details": {"input": 0.002, "output": 0.0, "total": 0.002},
        }
    ]


def test_usage_and_cost_are_not_sent_for_a_span_that_is_not_generation_like(
    sink: LangfuseSink, client: StubClient
):
    with sink.observation("tool", "lookup_cost_figure") as observation:
        observation.update(output={"page": 23}, units=1, model="ignored")

    assert client.updates() == [{"output": {"page": 23}}]


def test_an_unknown_cost_sends_the_usage_without_cost_details(
    sink: LangfuseSink, client: StubClient
):
    with sink.observation("generation", "answer") as observation:
        observation.update(input_tokens=10, output_tokens=2, cost=None)

    assert client.updates() == [{"usage_details": {"input": 10, "output": 2, "total": 12}}]


def test_an_error_marks_the_observation_as_an_error(sink: LangfuseSink, client: StubClient):
    with sink.observation("generation", "answer") as observation:
        observation.update(error="ProviderTimeoutError: timed out", metadata={"fallback": True})

    assert client.updates() == [
        {
            "level": "ERROR",
            "status_message": "ProviderTimeoutError: timed out",
            "metadata": {"fallback": True},
        }
    ]


def test_the_root_of_a_trace_takes_an_output_and_an_error_but_no_usage(
    sink: LangfuseSink, client: StubClient
):
    with sink.trace("ask") as root:
        root.update(output="the answer", input_tokens=5)

    assert client.updates() == [{"output": "the answer"}]


def test_the_trace_id_url_flush_and_shutdown_are_the_clients(
    sink: LangfuseSink, client: StubClient
):
    assert sink.current_trace_id() == "0123456789abcdef0123456789abcdef"
    assert sink.trace_url("abc") == "https://langfuse.example/project/p1/traces/abc"
    sink.flush()
    sink.shutdown()

    assert [name for name, _ in client.log] == ["flush", "shutdown"]


def test_a_trace_url_that_cannot_be_looked_up_is_none(sink: LangfuseSink, client: StubClient):
    client.url_error = ConnectionError("no route to host")

    assert sink.trace_url("abc") is None


def test_a_tracer_over_the_sink_sends_what_the_spans_learned(
    sink: LangfuseSink, client: StubClient
):
    tracer = Tracer(sink)

    with (
        tracer.trace("ask", input="question", tags=["api"]) as trace,
        tracer.span("generation", "answer", model="m") as span,
    ):
        span.end(
            output="text",
            input_tokens=3,
            output_tokens=4,
            cost=Cost(input_usd=0.1, output_usd=0.2),
        )

    assert trace.trace_id == "0123456789abcdef0123456789abcdef"
    assert client.starts() == [
        {"as_type": "span", "name": "ask", "input": "question"},
        {"as_type": "generation", "name": "answer", "input": None, "model": "m"},
    ]
    assert client.updates()[0]["usage_details"] == {"input": 3, "output": 4, "total": 7}
    assert client.updates()[0]["cost_details"] == {
        "input": 0.1,
        "output": 0.2,
        "total": pytest.approx(0.3),
    }
