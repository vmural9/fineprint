"""`FallbackChatModel`: which failures reach the second model, and what the reply says."""

import pytest

from fineprint.answer import DraftAnswer
from fineprint.providers.base import (
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolTurn,
)
from fineprint.providers.bedrock_chat import (
    InvalidToolInputError,
    ModelRefusedError,
    ResponseTruncatedError,
)
from fineprint.providers.fallback import FallbackChatModel
from tests.fakes import FakeChatModel, FlakyChatModel

DRAFT = DraftAnswer(
    answer="The standard Part B premium in 2026 is $202.90 a month.",
    found_in_handbook=True,
    citations=[],
    confidence="high",
)


def tool_turn(model: str) -> ToolTurn:
    return ToolTurn(
        tool_calls=[ToolCall(id="call-1", name="record_answer", arguments={})],
        content=[],
        model=model,
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
        stop_reason="tool_use",
    )


def ask(model: FallbackChatModel):
    return model.complete_structured("system", "user", DraftAnswer)


def test_a_working_primary_answers_and_the_fallback_is_never_asked():
    primary, fallback = FlakyChatModel([DRAFT]), FakeChatModel(DRAFT)

    result = ask(FallbackChatModel(primary, fallback))

    assert (result.model, result.fallback_used) == ("flaky-chat-model", False)
    assert fallback.calls == []


@pytest.mark.parametrize("error_type", [ProviderUnavailableError, ProviderTimeoutError])
def test_an_unavailable_or_slow_primary_hands_the_question_to_the_fallback(error_type):
    primary = FlakyChatModel([error_type("down")])
    fallback = FakeChatModel(DRAFT)

    result = ask(FallbackChatModel(primary, fallback))

    assert result.parsed == DRAFT
    assert result.fallback_used is True
    assert result.model == "fake-chat-model", "the fallback's own id, not the primary's"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        ModelRefusedError("refused"),
        ResponseTruncatedError("cut off"),
        InvalidToolInputError("bad arguments"),
        ProviderRequestError("400"),
        ValueError("a bug"),
    ],
)
def test_any_other_failure_propagates_and_the_fallback_is_not_asked(error):
    fallback = FakeChatModel(DRAFT)

    with pytest.raises(type(error)) as raised:
        ask(FallbackChatModel(FlakyChatModel([error]), fallback))

    assert raised.value is error
    assert fallback.calls == []


def test_when_both_fail_the_fallbacks_error_is_raised_with_the_primarys_as_its_cause():
    primary_error = ProviderTimeoutError("primary timed out")
    fallback_error = ProviderUnavailableError("fallback throttled")

    with pytest.raises(ProviderUnavailableError) as raised:
        ask(FallbackChatModel(FlakyChatModel([primary_error]), FlakyChatModel([fallback_error])))

    assert raised.value is fallback_error
    assert raised.value.__cause__ is primary_error


def test_the_next_call_goes_to_the_primary_again():
    primary = FlakyChatModel([ProviderUnavailableError("blip"), DRAFT])
    fallback = FakeChatModel(DRAFT)
    model = FallbackChatModel(primary, fallback)

    first, second = ask(model), ask(model)

    assert (first.fallback_used, second.fallback_used) == (True, False)
    assert second.model == "flaky-chat-model"
    assert len(fallback.calls) == 1


def test_tool_calls_fall_back_the_same_way():
    primary = FlakyChatModel([ProviderUnavailableError("down")])
    fallback = FlakyChatModel([tool_turn("us.anthropic.claude-sonnet-5")])

    turn = FallbackChatModel(primary, fallback).complete_with_tools(
        "system", [{"role": "user", "content": "hi"}], [], force_tool="record_answer"
    )

    assert turn.model == "us.anthropic.claude-sonnet-5"
    assert turn.fallback_used is True
    assert primary.calls == ["complete_with_tools"]
    assert fallback.calls == ["complete_with_tools"]


def test_a_refusal_during_a_tool_call_propagates():
    refusal = ModelRefusedError("refused")
    fallback = FlakyChatModel([])

    with pytest.raises(ModelRefusedError):
        FallbackChatModel(FlakyChatModel([refusal]), fallback).complete_with_tools("system", [], [])

    assert fallback.calls == []


def test_when_both_fail_during_a_tool_call_the_cause_chain_is_the_same():
    primary_error = ProviderTimeoutError("slow")
    fallback_error = ProviderTimeoutError("also slow")

    with pytest.raises(ProviderTimeoutError) as raised:
        FallbackChatModel(
            FlakyChatModel([primary_error]), FlakyChatModel([fallback_error])
        ).complete_with_tools("system", [], [])

    assert raised.value is fallback_error and raised.value.__cause__ is primary_error
