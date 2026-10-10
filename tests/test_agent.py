"""Tests for the agent: the tool loop that answers with the handbook's figures and dates.

Nothing here calls a model, a database or AWS. `ScriptedToolModel` plays the model one turn at a
time, which is how the tests can make it call two tools at once, call a tool the wrong way, or
hand in an answer that does not validate, and then check what the loop sends back.

The chunk text and the figures are the handbook's own: the sentences on page 27 that golden
question q023 is about, and the two tools' real results.
"""

import json
from datetime import date
from pathlib import Path

import boto3
import pytest

from fineprint.agent import (
    AGENT_SYSTEM_PROMPT,
    RECORD_ANSWER,
    AgentResponse,
    answer_with_tools,
)
from fineprint.answer import SYSTEM_PROMPT, build_user_prompt
from fineprint.config import Settings
from fineprint.db import connection_pool
from fineprint.pricing import Price
from fineprint.providers.base import ToolCall, ToolTurn
from fineprint.providers.bedrock_chat import (
    InvalidToolInputError,
    NoToolCallError,
    ResponseTruncatedError,
)
from fineprint.providers.factory import get_chat_model, get_embedder, get_reranker
from fineprint.retrieval import RetrievedChunk, Retriever
from fineprint.tools import TOOL_SPECS, run_tool
from fineprint.tracing import TracedChatModel, Tracer
from tests.fakes import FlakyChatModel, RecordingSink, ScriptedToolModel

# Two consecutive sentences of page 27, as the golden set's verified evidence quotes them.
HOSPITAL_TEXT = (
    "Days 61–90: You pay $434 each day. Days 91–150: You pay $868 each day while using your "
    "60 lifetime reserve days."
)
HOSPITAL = RetrievedChunk(
    chunk_id=74,
    ordinal=73,
    section="Inpatient hospital care",
    page_start=27,
    page_end=27,
    text=HOSPITAL_TEXT,
    score=0.0325,
    rank=1,
    lexical_rank=1,
    vector_rank=1,
    fused_rank=1,
    rerank_score=None,
)

QUESTION = (
    "Dad has been an inpatient in the hospital for 80 days straight in the same benefit "
    "period. What is he paying per day now?"
)
TODAY = date(2026, 10, 10)

VALID_ANSWER = {
    "answer": "On day 80 he pays $434 each day (page 27).",
    "found_in_handbook": True,
    "citations": [{"chunk_id": 74, "quote": "Days 61–90: You pay $434 each day."}],
    "confidence": "high",
}
# `confidence` is not one of the three allowed words, so the draft does not validate.
INVALID_ANSWER = VALID_ANSWER | {"confidence": "very sure indeed"}

LOOKUP_61_90 = ToolCall(
    id="toolu_lookup",
    name="lookup_cost_figure",
    arguments={"figure": "part_a_coinsurance_days_61_90"},
)
ENROLLMENT = ToolCall(
    id="toolu_enrollment",
    name="check_enrollment_window",
    arguments={"birth_date": "1961-05-10", "as_of": "2026-10-10"},
)


def record_answer(arguments: dict, *, call_id: str = "toolu_answer") -> ToolCall:
    return ToolCall(id=call_id, name=RECORD_ANSWER, arguments=arguments)


class FakeRetriever:
    """Returns canned chunks and remembers what it was asked."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = chunks if chunks is not None else [HOSPITAL]
        self.calls: list[tuple[str, str, int | None]] = []

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((question, mode, top_k))
        return list(self.chunks)


def ask(chat, **options) -> AgentResponse:
    options.setdefault("today", TODAY)
    return answer_with_tools(QUESTION, FakeRetriever(), chat, top_k=5, **options)


def tool_results(message: dict) -> dict[str, dict]:
    """The `tool_result` blocks of a user message, by the id of the call they answer."""
    assert message["role"] == "user"
    return {block["tool_use_id"]: block for block in message["content"]}


# --- the prompt -------------------------------------------------------------------------------


def test_the_agent_prompt_is_the_answer_prompt_with_the_tool_rules_added():
    assert AGENT_SYSTEM_PROMPT.startswith(SYSTEM_PROMPT)
    rules = AGENT_SYSTEM_PROMPT[len(SYSTEM_PROMPT) :]
    for name in ("lookup_cost_figure", "check_enrollment_window", RECORD_ANSWER):
        assert name in rules


# --- the loop ---------------------------------------------------------------------------------


def test_a_question_the_excerpts_answer_can_be_answered_in_one_turn():
    chat = ScriptedToolModel([[record_answer(VALID_ANSWER)]])
    retriever = FakeRetriever()

    response = answer_with_tools(QUESTION, retriever, chat, top_k=5, today=TODAY)

    assert retriever.calls == [(QUESTION, "hybrid", 5)]
    (call,) = chat.calls
    assert call.system == AGENT_SYSTEM_PROMPT
    assert call.messages == [{"role": "user", "content": build_user_prompt(QUESTION, [HOSPITAL])}]
    assert [tool.name for tool in call.tools] == [
        *(tool.name for tool in TOOL_SPECS),
        RECORD_ANSWER,
    ]
    assert call.force_tool is None
    assert response.answer == VALID_ANSWER["answer"]
    assert response.turns == 1
    assert response.tool_calls == []
    assert response.retrieved_chunk_ids == [74]
    assert [(cited.page_start, cited.quote_verified) for cited in response.citations] == [
        (27, True)
    ]
    assert (response.input_tokens, response.output_tokens) == (900, 120)
    assert response.model == "scripted-tool-model"
    assert response.cost_usd is None and response.trace_id is None


def test_the_record_answer_tool_takes_the_draft_answer_schema():
    chat = ScriptedToolModel([[record_answer(VALID_ANSWER)]])
    ask(chat)

    answer_tool = chat.calls[0].tools[-1]
    assert set(answer_tool.input_schema["required"]) == {
        "answer",
        "found_in_handbook",
        "citations",
        "confidence",
    }


def test_given_chunks_are_answered_from_without_searching():
    chat = ScriptedToolModel([[record_answer(VALID_ANSWER)]])
    retriever = FakeRetriever()

    response = answer_with_tools(QUESTION, retriever, chat, chunks=[HOSPITAL], today=TODAY)

    assert retriever.calls == []
    assert response.retrieved_chunk_ids == [74]


def test_two_tools_called_at_once_both_run_and_both_results_go_back_in_one_message():
    chat = ScriptedToolModel([[LOOKUP_61_90, ENROLLMENT], [record_answer(VALID_ANSWER)]])

    response = ask(chat)

    second = chat.calls[1].messages
    assert len(second) == 3
    assert second[1] == {"role": "assistant", "content": chat_content([LOOKUP_61_90, ENROLLMENT])}
    results = tool_results(second[2])
    assert list(results) == ["toolu_lookup", "toolu_enrollment"]
    lookup = run_tool("lookup_cost_figure", LOOKUP_61_90.arguments, today=TODAY)
    assert json.loads(results["toolu_lookup"]["content"]) == lookup.output
    assert json.loads(results["toolu_lookup"]["content"])["amount"] == "434"
    assert results["toolu_lookup"].get("is_error") is not True
    assert results["toolu_enrollment"].get("is_error") is not True

    assert response.turns == 2
    assert [record.name for record in response.tool_calls] == [
        "lookup_cost_figure",
        "check_enrollment_window",
    ]
    first = response.tool_calls[0]
    assert first.arguments == {"figure": "part_a_coinsurance_days_61_90"}
    assert first.output == lookup.output
    assert first.error is None
    assert first.pages == [27]
    assert first.source_text == "Days 61–90: You pay $434 each day."
    assert first.latency_ms >= 0


def chat_content(calls: list[ToolCall]) -> list[dict]:
    """The content blocks `ScriptedToolModel` hands back for a turn making `calls`."""
    return [
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in calls
    ]


def test_tokens_are_summed_over_every_turn():
    chat = ScriptedToolModel([[LOOKUP_61_90], [record_answer(VALID_ANSWER)]])

    response = ask(chat)

    assert (response.input_tokens, response.output_tokens) == (1800, 240)


def test_the_enrollment_tool_falls_back_to_today_when_the_model_gives_no_date():
    no_date = ToolCall(
        id="toolu_enrollment",
        name="check_enrollment_window",
        arguments={"birth_date": "1961-05-10"},
    )
    chat = ScriptedToolModel([[no_date], [record_answer(VALID_ANSWER)]])

    response = ask(chat, today=date(2026, 3, 1))

    assert response.tool_calls[0].output["as_of"] == "2026-03-01"


def test_when_the_turn_budget_is_spent_the_answer_is_forced():
    chat = ScriptedToolModel([[LOOKUP_61_90], [LOOKUP_61_90], [record_answer(VALID_ANSWER)]])

    response = ask(chat, max_tool_turns=2)

    assert [call.force_tool for call in chat.calls] == [None, None, RECORD_ANSWER]
    assert response.turns == 3
    assert len(response.tool_calls) == 2


def test_a_budget_of_no_tool_turns_forces_the_answer_at_once():
    chat = ScriptedToolModel([[record_answer(VALID_ANSWER)]])

    ask(chat, max_tool_turns=0)

    assert chat.calls[0].force_tool == RECORD_ANSWER


def test_a_forced_turn_that_still_records_no_answer_is_an_error_carrying_the_tokens():
    chat = ScriptedToolModel([[LOOKUP_61_90], [LOOKUP_61_90]])

    with pytest.raises(NoToolCallError) as raised:
        ask(chat, max_tool_turns=1)

    assert (raised.value.input_tokens, raised.value.output_tokens) == (1800, 240)


def test_a_model_failure_mid_loop_carries_the_tokens_of_the_turns_before_it():
    cut_off = ResponseTruncatedError("hit max_tokens")
    cut_off.input_tokens, cut_off.output_tokens = 1000, 16000
    first = ToolTurn(
        tool_calls=[LOOKUP_61_90],
        content=chat_content([LOOKUP_61_90]),
        model="flaky-chat-model",
        input_tokens=900,
        output_tokens=120,
        latency_ms=10.0,
        stop_reason="tool_use",
    )

    with pytest.raises(ResponseTruncatedError) as raised:
        ask(FlakyChatModel([first, cut_off]))

    assert (raised.value.input_tokens, raised.value.output_tokens) == (1900, 16120)


def test_an_answer_that_does_not_validate_gets_one_correction_turn():
    chat = ScriptedToolModel(
        [[record_answer(INVALID_ANSWER, call_id="toolu_bad")], [record_answer(VALID_ANSWER)]]
    )

    response = ask(chat)

    correction = chat.calls[1]
    assert correction.force_tool == RECORD_ANSWER
    result = tool_results(correction.messages[-1])["toolu_bad"]
    assert result["is_error"] is True
    assert "confidence" in result["content"]
    assert response.confidence == "high"
    assert response.turns == 2
    assert (response.input_tokens, response.output_tokens) == (1800, 240)


def test_a_second_answer_that_does_not_validate_raises_carrying_the_tokens_of_both_turns():
    chat = ScriptedToolModel(
        [
            [record_answer(INVALID_ANSWER, call_id="toolu_bad")],
            [record_answer(INVALID_ANSWER, call_id="toolu_worse")],
        ]
    )

    with pytest.raises(InvalidToolInputError) as raised:
        ask(chat)

    assert "confidence" in str(raised.value)
    assert (raised.value.input_tokens, raised.value.output_tokens) == (1800, 240)


def test_a_tool_error_goes_back_to_the_model_marked_as_an_error():
    leap_day = ToolCall(
        id="toolu_leap",
        name="check_enrollment_window",
        arguments={"birth_date": "1960-02-29", "as_of": "2026-10-10"},
    )
    chat = ScriptedToolModel([[leap_day], [record_answer(VALID_ANSWER)]])

    response = ask(chat)

    result = tool_results(chat.calls[1].messages[-1])["toolu_leap"]
    assert result["is_error"] is True
    assert "February 29" in result["content"]
    (record,) = response.tool_calls
    assert record.output is None
    assert "February 29" in record.error
    assert record.pages == [] and record.source_text is None


def test_an_answer_made_beside_other_tool_calls_waits_for_their_results():
    chat = ScriptedToolModel(
        [
            [LOOKUP_61_90, record_answer(VALID_ANSWER, call_id="toolu_early")],
            [record_answer(VALID_ANSWER)],
        ]
    )

    response = ask(chat)

    results = tool_results(chat.calls[1].messages[-1])
    assert results["toolu_lookup"].get("is_error") is not True
    assert results["toolu_early"]["is_error"] is True
    assert response.turns == 2
    assert [record.name for record in response.tool_calls] == ["lookup_cost_figure"]


def test_the_response_says_a_fallback_answered_when_any_turn_used_one():
    fallback_turn = ToolTurn(
        tool_calls=[LOOKUP_61_90],
        content=chat_content([LOOKUP_61_90]),
        model="backup-model",
        input_tokens=900,
        output_tokens=120,
        latency_ms=10.0,
        stop_reason="tool_use",
        fallback_used=True,
    )
    chat = ScriptedToolModel([fallback_turn, [record_answer(VALID_ANSWER)]])

    assert ask(chat).fallback_used is True
    assert ask(ScriptedToolModel([[record_answer(VALID_ANSWER)]])).fallback_used is False


# --- the trace --------------------------------------------------------------------------------

# 900 input and 120 output tokens at $1 and $2 per million is $0.00114 a turn.
FAKE_PRICES = {
    "scripted-tool-model": Price(
        model="scripted-tool-model",
        input_per_million=1.0,
        output_per_million=2.0,
        per_unit=None,
        unit="tokens",
        source_url="https://prices.example/list.json",
        retrieved_on="2026-10-10",
    )
}


def traced(script: list) -> tuple[TracedChatModel, Tracer, RecordingSink]:
    sink = RecordingSink(trace_id="trace-agent")
    tracer = Tracer(sink)
    return TracedChatModel(ScriptedToolModel(script), tracer, prices=FAKE_PRICES), tracer, sink


def test_the_loop_is_an_agent_span_holding_a_generation_per_turn_and_a_span_per_tool():
    chat, tracer, sink = traced([[LOOKUP_61_90, ENROLLMENT], [record_answer(VALID_ANSWER)]])

    ask(chat, tracer=tracer)

    (root,) = sink.traces
    assert root.name == "agent"
    assert root.input == QUESTION
    assert root.fields["output"] == VALID_ANSWER["answer"]
    (agent,) = root.children
    assert (agent.kind, agent.name) == ("agent", "tools")
    assert [(child.kind, child.name) for child in agent.children] == [
        ("generation", "answer"),
        ("tool", "lookup_cost_figure"),
        ("tool", "check_enrollment_window"),
        ("generation", "answer"),
    ]
    lookup = agent.children[1]
    assert lookup.input == {"figure": "part_a_coinsurance_days_61_90"}
    assert lookup.fields["output"]["amount"] == "434"


def test_a_tool_error_is_recorded_on_its_span():
    leap_day = ToolCall(
        id="toolu_leap", name="check_enrollment_window", arguments={"birth_date": "1960-02-29"}
    )
    chat, tracer, sink = traced([[leap_day], [record_answer(VALID_ANSWER)]])

    response = ask(chat, tracer=tracer)

    tool = next(obs for obs in sink.observations if obs.kind == "tool")
    assert "ToolError" in tool.fields["error"]
    assert "February 29" in tool.fields["error"]
    assert response.answer == VALID_ANSWER["answer"]


def test_cost_and_trace_id_come_from_the_trace_and_cover_every_turn():
    chat, tracer, _ = traced([[LOOKUP_61_90], [record_answer(VALID_ANSWER)]])

    response = ask(chat, tracer=tracer)

    assert response.trace_id == "trace-agent"
    assert response.cost_usd == pytest.approx(2 * 0.00114)


def test_the_agent_joins_a_trace_its_caller_already_opened():
    chat, tracer, sink = traced([[record_answer(VALID_ANSWER)]])

    with tracer.trace("agent", input=QUESTION, tags=["api"]):
        response = ask(chat, tracer=tracer)

    assert len(sink.traces) == 1
    assert sink.traces[0].tags == ["api"]
    assert response.trace_id == "trace-agent"


# --- live -------------------------------------------------------------------------------------

GOLDEN_SET = Path(__file__).resolve().parent.parent / "evals" / "golden_set.jsonl"


@pytest.mark.live
def test_the_agent_looks_up_the_days_61_to_90_coinsurance_for_golden_question_q023():
    """The real agent on q023: 80 days into a hospital stay, what does Dad pay each day?

    Calls Bedrock (an embedding, the answer model's turns, and a re-rank when one is configured)
    and reads the configured chunk set from the database, which must be ingested. Run it with

        AWS_PROFILE=<your-aws-profile> AWS_REGION=us-west-2 uv run pytest -m live -s \\
            tests/test_agent.py
    """
    if boto3.session.Session().get_credentials() is None:
        pytest.skip("no AWS credentials: run with AWS_PROFILE=<your-profile> AWS_REGION=us-west-2")
    items = [json.loads(line) for line in GOLDEN_SET.read_text().splitlines() if line.strip()]
    question = next(item["question"] for item in items if item["id"] == "q023")
    settings = Settings()
    tracer = Tracer()

    with connection_pool(settings.database_url) as pool:
        retriever = Retriever(
            pool,
            get_embedder(settings, tracer),
            settings,
            reranker=get_reranker(settings, tracer),
            tracer=tracer,
        )
        response = answer_with_tools(
            question,
            retriever,
            get_chat_model(settings, tracer),
            max_tool_turns=settings.agent_max_tool_turns,
            tracer=tracer,
        )

    for call in response.tool_calls:
        print(
            f"{call.name}({call.arguments}) -> {call.output if call.error is None else call.error}"
        )
    print(response.answer)
    print(
        f"{response.turns} turns, {response.input_tokens} in / {response.output_tokens} out, "
        f"cost {response.cost_usd}"
    )
    lookups = [call.arguments.get("figure") for call in response.tool_calls]
    assert "part_a_coinsurance_days_61_90" in lookups
    assert "434" in response.answer
