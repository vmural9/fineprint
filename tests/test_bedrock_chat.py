"""The Bedrock chat model, exercised against a fake Anthropic client so no test calls AWS.

The fake returns real `anthropic.types` objects, so a wrong assumption about the response
shape fails here rather than in production.
"""

from typing import Literal

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from pydantic import BaseModel

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
    assert call["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
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
