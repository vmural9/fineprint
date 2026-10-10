"""The Bedrock chat model, exercised against a fake Anthropic client so no test calls AWS.

The fake returns real `anthropic.types` objects, so a wrong assumption about the response
shape fails here rather than in production.
"""

from typing import Literal

import anthropic
import httpx
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from pydantic import BaseModel

from fineprint.providers.base import (
    ProviderError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolSpec,
)
from fineprint.providers.bedrock_chat import (
    TOOL_NAME,
    BedrockChatModel,
    InvalidToolInputError,
    ModelRefusedError,
    NoToolCallError,
    ResponseTruncatedError,
)

MODEL = "us.anthropic.claude-opus-5"


class Citation(BaseModel):
    chunk_id: int
    quote: str


class DraftAnswer(BaseModel):
    answer: str
    found_in_handbook: bool
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


VALID_INPUT = {
    "answer": "The standard Part B premium is $202.90 a month in 2026.",
    "found_in_handbook": True,
    "citations": [{"chunk_id": 7, "quote": "the Part B standard monthly premium is $202.90"}],
    "confidence": "high",
}
# `confidence` is not one of the three allowed words, so Pydantic rejects it.
INVALID_INPUT = VALID_INPUT | {"confidence": "very sure indeed"}


def message(
    *,
    blocks: list,
    stop_reason: str = "tool_use",
    input_tokens: int = 860,
    output_tokens: int = 170,
) -> Message:
    return Message(
        id="msg_test",
        content=blocks,
        model=MODEL,
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def tool_call(tool_input: dict, *, block_id: str = "toolu_1") -> ToolUseBlock:
    return ToolUseBlock(id=block_id, name=TOOL_NAME, input=tool_input, type="tool_use")


class FakeMessages:
    def __init__(self, responses: list[Message]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> Message:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAnthropicBedrock:
    """The one call the chat model makes, plus a record of how it was made."""

    def __init__(self, *responses: Message):
        self.messages = FakeMessages(list(responses))


def chat_model(*responses: Message) -> tuple[BedrockChatModel, FakeAnthropicBedrock]:
    client = FakeAnthropicBedrock(*responses)
    return BedrockChatModel(model=MODEL, region="us-west-2", client=client), client


def test_a_valid_tool_call_comes_back_parsed_with_tokens_and_latency():
    chat, _ = chat_model(message(blocks=[tool_call(VALID_INPUT)]))

    result = chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert isinstance(result.parsed, DraftAnswer)
    assert result.parsed.confidence == "high"
    assert result.parsed.citations[0].chunk_id == 7
    assert result.model == MODEL
    assert result.input_tokens == 860
    assert result.output_tokens == 170
    assert result.latency_ms > 0


def test_the_request_asks_for_the_schema_as_a_tool_and_sends_no_sampling_parameters():
    chat, client = chat_model(message(blocks=[tool_call(VALID_INPUT)]))

    chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    (call,) = client.messages.calls
    assert call["model"] == MODEL
    assert call["system"] == "system rules"
    assert call["messages"] == [{"role": "user", "content": "the excerpts"}]
    (tool,) = call["tools"]
    assert tool["name"] == TOOL_NAME
    assert tool["input_schema"] == DraftAnswer.model_json_schema()
    # Verified live on 2026-09-21: Bedrock rejects `strict` on this path with a 400.
    assert "strict" not in tool
    # One call per reply. Verified live on 2026-09-28: Bedrock accepts the field, and asked for
    # three calls at once, Claude Sonnet 5 made one.
    assert call["tool_choice"] == {
        "type": "tool",
        "name": TOOL_NAME,
        "disable_parallel_tool_use": True,
    }
    # Claude Opus 5 returns 400 for temperature and every other sampling parameter.
    assert not {"temperature", "top_p", "top_k"} & set(call)


def test_an_invalid_tool_call_is_retried_once_with_the_validation_error():
    first = message(blocks=[tool_call(INVALID_INPUT)])
    second = message(blocks=[tool_call(VALID_INPUT, block_id="toolu_2")])
    chat, client = chat_model(first, second)

    result = chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert result.parsed.confidence == "high"
    assert len(client.messages.calls) == 2
    retry_messages = client.messages.calls[1]["messages"]
    assert [entry["role"] for entry in retry_messages] == ["user", "assistant", "user"]
    assert retry_messages[1]["content"] == first.content
    (tool_result,) = retry_messages[2]["content"]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["is_error"] is True
    assert "confidence" in tool_result["content"]


def test_a_reply_sent_back_gets_a_tool_result_for_every_tool_call_it_holds():
    # A reply can hold several calls to the tool at once. Neither of these two validates, so the
    # reply is sent back, and the API refuses that turn with a 400 unless every tool_use id in
    # the replayed reply has a tool_result after it.
    first = message(
        blocks=[
            tool_call(INVALID_INPUT, block_id="toolu_1"),
            tool_call(INVALID_INPUT, block_id="toolu_2"),
        ]
    )
    second = message(blocks=[tool_call(VALID_INPUT, block_id="toolu_3")])
    chat, client = chat_model(first, second)

    result = chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert result.parsed.confidence == "high"
    retry_messages = client.messages.calls[1]["messages"]
    assert retry_messages[1]["content"] == first.content
    tool_use_ids = [block.id for block in retry_messages[1]["content"] if block.type == "tool_use"]
    tool_results = retry_messages[2]["content"]
    assert [entry["tool_use_id"] for entry in tool_results] == tool_use_ids
    assert tool_use_ids == ["toolu_1", "toolu_2"]
    assert all(entry["type"] == "tool_result" for entry in tool_results)
    assert all(entry["is_error"] is True for entry in tool_results)
    # The first call gets the validation error; the other, which failed too, is told so, and
    # that one call is expected.
    assert "confidence" in tool_results[0]["content"]
    assert "did not fit the schema either" in tool_results[1]["content"]
    assert f"one call to {TOOL_NAME}" in tool_results[1]["content"]


def test_a_later_tool_call_that_validates_is_the_answer_when_an_earlier_one_does_not():
    first = message(
        blocks=[
            tool_call(INVALID_INPUT, block_id="toolu_1"),
            tool_call(VALID_INPUT, block_id="toolu_2"),
        ]
    )
    # Asked for only if the valid call above were thrown away.
    unused = message(blocks=[tool_call(VALID_INPUT | {"confidence": "low"}, block_id="toolu_3")])
    chat, client = chat_model(first, unused)

    result = chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert result.parsed.confidence == "high"
    assert len(client.messages.calls) == 1
    assert (result.input_tokens, result.output_tokens) == (860, 170)


def test_a_tool_call_that_is_invalid_twice_raises_with_the_validation_error():
    chat, client = chat_model(
        message(blocks=[tool_call(INVALID_INPUT)]),
        message(blocks=[tool_call(INVALID_INPUT, block_id="toolu_2")]),
    )

    with pytest.raises(InvalidToolInputError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert len(client.messages.calls) == 2
    assert "DraftAnswer" in str(raised.value)
    assert "confidence" in str(raised.value)


def test_a_refusal_is_reported_before_any_content_is_read():
    chat, _ = chat_model(message(blocks=[], stop_reason="refusal"))

    with pytest.raises(ModelRefusedError):
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)


def test_an_answer_cut_off_by_max_tokens_is_reported_as_such():
    chat, _ = chat_model(message(blocks=[tool_call(VALID_INPUT)], stop_reason="max_tokens"))

    with pytest.raises(ResponseTruncatedError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert "max_tokens" in str(raised.value)


def test_a_reply_with_no_tool_call_says_what_came_back_instead():
    chat, _ = chat_model(
        message(
            blocks=[TextBlock(text="I would rather just chat.", type="text")],
            stop_reason="end_turn",
        )
    )

    with pytest.raises(NoToolCallError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert TOOL_NAME in str(raised.value)
    assert "end_turn" in str(raised.value)


# Bedrock bills the tokens of a reply that was then refused, cut off, or unusable, so each of
# these errors says how many were used, summed over every request made so far.


def test_a_refusal_carries_the_tokens_it_used():
    chat, _ = chat_model(
        message(blocks=[], stop_reason="refusal", input_tokens=700, output_tokens=15)
    )

    with pytest.raises(ModelRefusedError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert (raised.value.input_tokens, raised.value.output_tokens) == (700, 15)


def test_a_truncated_answer_carries_the_tokens_it_used():
    chat, _ = chat_model(
        message(
            blocks=[tool_call(VALID_INPUT)],
            stop_reason="max_tokens",
            input_tokens=900,
            output_tokens=4096,
        )
    )

    with pytest.raises(ResponseTruncatedError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert (raised.value.input_tokens, raised.value.output_tokens) == (900, 4096)


def test_a_reply_with_no_tool_call_carries_the_tokens_it_used():
    chat, _ = chat_model(
        message(
            blocks=[TextBlock(text="Just chatting.", type="text")],
            stop_reason="end_turn",
            input_tokens=650,
            output_tokens=9,
        )
    )

    with pytest.raises(NoToolCallError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert (raised.value.input_tokens, raised.value.output_tokens) == (650, 9)


def test_a_tool_call_invalid_twice_carries_the_tokens_of_both_requests():
    chat, _ = chat_model(
        message(blocks=[tool_call(INVALID_INPUT)], input_tokens=800, output_tokens=100),
        message(
            blocks=[tool_call(INVALID_INPUT, block_id="toolu_2")],
            input_tokens=1000,
            output_tokens=120,
        ),
    )

    with pytest.raises(InvalidToolInputError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert (raised.value.input_tokens, raised.value.output_tokens) == (1800, 220)


def test_a_chat_error_that_was_raised_without_usage_has_none():
    assert ModelRefusedError("no reply").input_tokens is None
    assert ModelRefusedError("no reply").output_tokens is None


REQUEST = httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com/v1/messages")


# The SDK's own class for each status it has one for; the base class for the rest.
STATUS_CLASSES = {
    400: anthropic.BadRequestError,
    401: anthropic.AuthenticationError,
    403: anthropic.PermissionDeniedError,
    404: anthropic.NotFoundError,
    413: anthropic.RequestTooLargeError,
    429: anthropic.RateLimitError,
    500: anthropic.InternalServerError,
    503: anthropic.ServiceUnavailableError,
    504: anthropic.DeadlineExceededError,
    529: anthropic.OverloadedError,
}


def status_error(status: int) -> anthropic.APIStatusError:
    """The error the SDK raises for an HTTP status: its subclass for that status where it has one
    (408, 424 and 502 have none, so those are the base class)."""
    response = httpx.Response(status, request=REQUEST)
    error_class = STATUS_CLASSES.get(status, anthropic.APIStatusError)
    return error_class(f"HTTP {status}", response=response, body=None)


# One case per row of the mapping: what the SDK raises after its own retries, and what the
# service raises instead.
SDK_FAILURES = [
    pytest.param(anthropic.APITimeoutError(request=REQUEST), ProviderTimeoutError, id="timeout"),
    pytest.param(
        anthropic.APIConnectionError(request=REQUEST), ProviderUnavailableError, id="connection"
    ),
    pytest.param(status_error(408), ProviderTimeoutError, id="408"),
    pytest.param(status_error(504), ProviderTimeoutError, id="504"),
    pytest.param(status_error(424), ProviderUnavailableError, id="424"),
    pytest.param(status_error(429), ProviderUnavailableError, id="429"),
    pytest.param(status_error(500), ProviderUnavailableError, id="500"),
    pytest.param(status_error(502), ProviderUnavailableError, id="502"),
    pytest.param(status_error(503), ProviderUnavailableError, id="503"),
    pytest.param(status_error(529), ProviderUnavailableError, id="529"),
    pytest.param(status_error(400), ProviderRequestError, id="400"),
    pytest.param(status_error(401), ProviderRequestError, id="401"),
    pytest.param(status_error(403), ProviderRequestError, id="403"),
    pytest.param(status_error(404), ProviderRequestError, id="404"),
    pytest.param(status_error(413), ProviderRequestError, id="413"),
]


class RaisingMessages:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def create(self, **kwargs: object) -> Message:
        self.calls += 1
        raise self.error


class RaisingClient:
    def __init__(self, error: Exception):
        self.messages = RaisingMessages(error)


@pytest.mark.parametrize(("error", "expected"), SDK_FAILURES)
def test_an_sdk_failure_becomes_its_provider_error_naming_the_model(error, expected):
    chat = BedrockChatModel(model=MODEL, region="us-west-2", client=RaisingClient(error))

    with pytest.raises(ProviderError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    # Exactly the mapped class, not a sibling: the three are told apart by what the caller does.
    assert type(raised.value) is expected
    assert MODEL in str(raised.value)
    assert raised.value.__cause__ is error


def test_a_failure_that_is_not_the_providers_passes_through_unchanged():
    error = ValueError("a bug, not an outage")
    chat = BedrockChatModel(model=MODEL, region="us-west-2", client=RaisingClient(error))

    with pytest.raises(ValueError) as raised:
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)

    assert raised.value is error


def test_a_timeout_is_a_timeout_although_the_sdk_makes_it_a_kind_of_connection_error():
    # APITimeoutError subclasses APIConnectionError, which maps to "unavailable", so the
    # timeout has to be checked first.
    assert issubclass(anthropic.APITimeoutError, anthropic.APIConnectionError)
    error = anthropic.APITimeoutError(request=REQUEST)
    chat = BedrockChatModel(model=MODEL, region="us-west-2", client=RaisingClient(error))

    with pytest.raises(ProviderTimeoutError):
        chat.complete_structured("system rules", "the excerpts", DraftAnswer)


def test_the_provider_errors_are_not_the_reply_errors():
    # T4: the service falls back on ProviderError and never on a reply it could not use.
    for reply_error in (
        ModelRefusedError,
        ResponseTruncatedError,
        NoToolCallError,
        InvalidToolInputError,
    ):
        assert not issubclass(reply_error, ProviderError)


class RecordingAnthropicBedrock:
    """Stands in for the `AnthropicBedrock` class, keeping the arguments it was built with."""

    instances: list["RecordingAnthropicBedrock"] = []

    def __init__(self, **kwargs: object):
        self.kwargs = kwargs
        self.instances.append(self)


def test_the_anthropic_client_gets_the_region_the_timeout_and_the_retry_count(monkeypatch):
    RecordingAnthropicBedrock.instances = []
    monkeypatch.setattr(
        "fineprint.providers.bedrock_chat.AnthropicBedrock", RecordingAnthropicBedrock
    )
    chat = BedrockChatModel(model=MODEL, region="us-east-1", max_retries=2, timeout=30.0)

    client = chat.client

    assert client is RecordingAnthropicBedrock.instances[0]
    assert client.kwargs == {"aws_region": "us-east-1", "max_retries": 2, "timeout": 30.0}
    assert chat.client is client, "the client is built once"


def test_the_defaults_are_sixty_seconds_and_four_retries(monkeypatch):
    RecordingAnthropicBedrock.instances = []
    monkeypatch.setattr(
        "fineprint.providers.bedrock_chat.AnthropicBedrock", RecordingAnthropicBedrock
    )

    client = BedrockChatModel(model=MODEL, region="us-west-2").client

    assert client.kwargs == {"aws_region": "us-west-2", "max_retries": 4, "timeout": 60.0}


# --- complete_with_tools ------------------------------------------------------------------

LOOKUP = ToolSpec(
    name="lookup_cost_figure",
    description="Look up one of the listed 2026 Medicare dollar amounts.",
    input_schema={
        "type": "object",
        "properties": {"figure": {"type": "string", "enum": ["part_b_deductible"]}},
        "required": ["figure"],
    },
)
ANSWER = ToolSpec(
    name=TOOL_NAME,
    description="Record your final answer.",
    input_schema=DraftAnswer.model_json_schema(),
)


def named_call(name: str, tool_input: dict, *, block_id: str) -> ToolUseBlock:
    return ToolUseBlock(id=block_id, name=name, input=tool_input, type="tool_use")


def test_a_tool_turn_offers_every_tool_and_lets_the_model_choose_any_of_them():
    reply = message(
        blocks=[named_call("lookup_cost_figure", {"figure": "part_b_deductible"}, block_id="t1")]
    )
    chat, client = chat_model(reply)
    conversation = [{"role": "user", "content": "the question and the excerpts"}]

    chat.complete_with_tools("system rules", conversation, [LOOKUP, ANSWER])

    (request,) = client.messages.calls
    assert request["model"] == MODEL
    assert request["system"] == "system rules"
    assert request["messages"] == conversation
    assert request["max_tokens"] == chat.max_tokens
    assert request["tools"] == [
        {
            "name": "lookup_cost_figure",
            "description": LOOKUP.description,
            "input_schema": LOOKUP.input_schema,
        },
        {
            "name": TOOL_NAME,
            "description": ANSWER.description,
            "input_schema": ANSWER.input_schema,
        },
    ]
    # "any": some tool must be called, and several calls in one reply are allowed.
    assert request["tool_choice"] == {"type": "any"}
    assert "temperature" not in request


def test_a_forced_tool_turn_names_the_one_tool_the_model_must_call():
    chat, client = chat_model(message(blocks=[named_call(TOOL_NAME, VALID_INPUT, block_id="t1")]))

    chat.complete_with_tools("system rules", [], [LOOKUP, ANSWER], force_tool=TOOL_NAME)

    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": TOOL_NAME}


def test_every_tool_call_in_a_reply_comes_back_in_order_with_the_reply_to_echo():
    reply = message(
        blocks=[
            TextBlock(text="I need two figures.", type="text"),
            named_call("lookup_cost_figure", {"figure": "part_b_deductible"}, block_id="t1"),
            named_call("check_enrollment_window", {"birth_date": "1961-05-10"}, block_id="t2"),
        ],
        input_tokens=1200,
        output_tokens=95,
    )
    chat, _ = chat_model(reply)

    turn = chat.complete_with_tools("system rules", [], [LOOKUP, ANSWER])

    assert turn.tool_calls == [
        ToolCall(id="t1", name="lookup_cost_figure", arguments={"figure": "part_b_deductible"}),
        ToolCall(id="t2", name="check_enrollment_window", arguments={"birth_date": "1961-05-10"}),
    ]
    assert turn.content == reply.content
    assert turn.model == MODEL
    assert (turn.input_tokens, turn.output_tokens) == (1200, 95)
    assert turn.stop_reason == "tool_use"
    assert turn.latency_ms >= 0
    assert turn.fallback_used is False


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (message(blocks=[], stop_reason="refusal"), ModelRefusedError),
        (
            message(
                blocks=[named_call(TOOL_NAME, VALID_INPUT, block_id="t1")],
                stop_reason="max_tokens",
            ),
            ResponseTruncatedError,
        ),
        (
            message(
                blocks=[TextBlock(text="No tools today.", type="text")], stop_reason="end_turn"
            ),
            NoToolCallError,
        ),
    ],
    ids=["refusal", "max_tokens", "no tool call"],
)
def test_a_tool_turn_that_cannot_be_used_raises_carrying_the_tokens_billed(reply, expected):
    reply.usage = Usage(input_tokens=1500, output_tokens=40)
    chat, _ = chat_model(reply)

    with pytest.raises(expected) as raised:
        chat.complete_with_tools("system rules", [], [LOOKUP, ANSWER])

    assert (raised.value.input_tokens, raised.value.output_tokens) == (1500, 40)


def test_a_tool_turn_maps_sdk_failures_to_provider_errors():
    chat = BedrockChatModel(
        model=MODEL, region="us-west-2", client=RaisingClient(status_error(529))
    )

    with pytest.raises(ProviderUnavailableError):
        chat.complete_with_tools("system rules", [], [LOOKUP, ANSWER])
