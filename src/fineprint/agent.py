"""The agent: an answer that may call the two handbook tools before it is written.

`answer_question` hands the model the retrieved passages and takes one answer back. That is
enough when the answer is a sentence the handbook prints. It is not enough when the answer has
to be worked out: what someone pays on day 80 of a hospital stay, or the last day of their
Initial Enrollment Period when they were born on May 10. A model doing that arithmetic in its
head, or remembering last year's amount, writes a figure that looks right and is not.

So the agent offers the model the two tools in `fineprint.tools` — `lookup_cost_figure`, which
returns a 2026 dollar amount exactly as the handbook prints it, and `check_enrollment_window`,
which works out enrollment dates from the handbook's rules — plus a third tool, `record_answer`,
whose input is the same `DraftAnswer` the plain answer path validates. Every turn the model must
call at least one of them, and may call several at once. The loop runs the tool calls, sends
their results back, and stops when the model records its answer:

1. Retrieve once, or take the passages the caller already retrieved. The first message gives
   today's date before the question, so "my mom turns 65 on March 12" has a year to work from.
2. Ask for a turn. Run each tool call it makes, each in a `tool` span of its own, and send the
   results back; an error a tool raises goes back too, marked as one, so the model can say the
   handbook does not cover that case.
3. After `max_tool_turns` turns of tool calls, the next turn must be `record_answer`.
4. An answer that does not validate is sent back once with the validation error, as in
   `answer_question`; a second one that does not validate is an error.

The answer is then checked exactly as `answer_question` checks it: citations that name a passage
the model was not given are dropped, and pages come from the database, never from the model.
"""

import json
import time
from collections.abc import Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, ValidationError

from fineprint.answer import (
    SYSTEM_PROMPT,
    AnswerResponse,
    DraftAnswer,
    build_user_prompt,
    check_citations,
)
from fineprint.providers.base import ToolCall, ToolCallingModel, ToolSpec, ToolTurn
from fineprint.providers.bedrock_chat import (
    ChatModelError,
    InvalidToolInputError,
    NoToolCallError,
)
from fineprint.retrieval import RetrievedChunk, Retriever, SearchMode
from fineprint.tools import TOOL_SPECS, ToolError, run_tool
from fineprint.tracing import Tracer

# The tool the model gives its final answer through. The same name as the plain answer path's.
RECORD_ANSWER = "record_answer"

# What the model is told: part 1's rules, which still hold, and what the tools change about them.
AGENT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Tools: you also have two tools, and what they return comes from the same handbook, so it is \
evidence just as the excerpts are. Call lookup_cost_figure for every dollar amount your answer \
states that its list of figures holds, even when an excerpt prints the same amount, and when \
the answer does arithmetic on such an amount (a daily coinsurance over a number of days, say), \
do the arithmetic on the exact amount the tool returned. Call check_enrollment_window whenever \
the answer gives a date worked out from someone's birthday, such as when their Initial \
Enrollment Period starts or ends or when their coverage would start, and take every such date \
from its result instead of working it out yourself. The message gives today's date: when the \
question gives a birthday without a year, or speaks of now or this year, work the year out from \
today's date, and when the question sets a different now (such as "It's now February"), pass \
that date to check_enrollment_window as as_of. Any other amount must come from the \
excerpts, copied as rule 2 says; an amount that is in neither a tool result nor an excerpt is \
not stated. If a tool returns an error, the handbook does not cover that case: say so in the \
answer and send the reader somewhere that does know, as rule 4 says, rather than working the \
answer out another way. The one exception is an error saying an argument was written wrongly: \
correct the argument and call the tool again. You may call several tools in one turn. Always \
give your final answer by calling record_answer, and only after you have read the result of \
every tool you called. Cite the excerpts that support the rules you state. A tool result has no \
chunk id and is never cited, so whenever the answer states an amount or a date a tool returned, \
name the handbook page the tool gave for it in the answer text, written as "(page N)". Set \
found_in_handbook to true when the excerpts or the tool results answer the question.\
"""
)

RECORD_ANSWER_SPEC = ToolSpec(
    name=RECORD_ANSWER,
    description=(
        "Record your final answer, once you have every tool result you need. Every field is "
        "required. Cite the excerpts by chunk id; tool results are never cited."
    ),
    input_schema=DraftAnswer.model_json_schema(),
)

# What a `record_answer` made in the same turn as other tool calls is told: it was written
# before their results existed, so it cannot rest on them.
ANSWERED_TOO_EARLY = (
    "Not recorded: this answer was given in the same turn as other tool calls, before their "
    "results came back. Read the results, then call record_answer again."
)
# What a tool call made in a turn that had to be the answer is told.
NOT_RUN_ANSWER_DUE = "Not run: this turn had to record the answer."


class ToolCallRecord(BaseModel):
    """One tool call the agent ran: what the model asked for and what it got back.

    `output` is what the model received, or None when the tool raised; `error` is then the
    message the model received instead. `pages` and `source_text` say where in the handbook
    the result came from.
    """

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None
    pages: list[int]
    source_text: str | None
    latency_ms: float


class AgentResponse(AnswerResponse):
    """An answer from the agent: everything `/ask` returns, plus the tool calls behind it.

    Tokens, cost and latency cover every model turn and every tool call. `turns` counts the
    model turns, the final `record_answer` included. `fallback_used` is true when any turn was
    answered by the fallback model.
    """

    tool_calls: list[ToolCallRecord]
    turns: int


def answer_with_tools(
    question: str,
    retriever: Retriever,
    chat: ToolCallingModel,
    top_k: int | None = None,
    *,
    mode: SearchMode = "hybrid",
    chunks: Sequence[RetrievedChunk] | None = None,
    today: date | None = None,
    max_tool_turns: int = 4,
    tracer: Tracer | None = None,
) -> AgentResponse:
    """Answer `question` from the handbook, letting the model call the tools first.

    `chunks`, `mode` and `top_k` work as in `answer_question`. `today` is the date the enrollment
    tool checks against when the model gives none; the service passes nothing and gets the real
    date, the eval runner pins one. With a `tracer`, everything runs inside an "agent" trace, or
    inside the one the caller already opened, and the response carries the trace's id and cost.
    """
    today = today if today is not None else date.today()
    # The tool spans need a tracer to open them, so an untraced call gets a private one, but
    # only a tracer the caller gave says what the question cost: a private one sees no
    # generation spans, and would report the model's turns as free.
    own_tracer = tracer if tracer is not None else Tracer()
    with own_tracer.trace("agent", input=question) as trace:
        if chunks is None:
            chunks = retriever.search(question, mode=mode, top_k=top_k)
        chunks = list(chunks)
        with own_tracer.span("agent", "tools", input=question) as span:
            response = _loop(question, chunks, chat, today, max_tool_turns, own_tracer)
            span.end(
                output={"turns": response.turns, "tool_calls": len(response.tool_calls)},
                units=len(response.tool_calls),
            )
        trace.set_output(response.answer)
        if tracer is None:
            return response
        return response.model_copy(
            update={"trace_id": trace.trace_id, "cost_usd": trace.record.cost_usd}
        )


def _loop(
    question: str,
    chunks: list[RetrievedChunk],
    chat: ToolCallingModel,
    today: date,
    max_tool_turns: int,
    tracer: Tracer,
) -> AgentResponse:
    """Ask for turns until the model records an answer that validates."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Today's date: {today.isoformat()}.\n\n"
            + build_user_prompt(question, chunks),
        }
    ]
    tools = [*TOOL_SPECS, RECORD_ANSWER_SPEC]
    turns: list[ToolTurn] = []
    records: list[ToolCallRecord] = []
    tool_turns = 0
    correcting = False  # True once the one correction turn has been granted
    started = time.perf_counter()

    while True:
        must_answer = correcting or tool_turns >= max_tool_turns
        try:
            turn = chat.complete_with_tools(
                AGENT_SYSTEM_PROMPT,
                messages,
                tools,
                force_tool=RECORD_ANSWER if must_answer else None,
            )
        except ChatModelError as error:
            _with_usage(error, turns)
            raise
        turns.append(turn)

        answers = [call for call in turn.tool_calls if call.name == RECORD_ANSWER]
        others = [call for call in turn.tool_calls if call.name != RECORD_ANSWER]
        results: dict[str, dict[str, Any]] = {}

        if others and not must_answer:
            # A turn of tool calls. An answer made alongside them was written before their
            # results existed, so it is sent back to be made again.
            tool_turns += 1
            for call in others:
                record, results[call.id] = _run_tool_call(call, today, tracer)
                records.append(record)
            for call in answers:
                results[call.id] = _error_result(call.id, ANSWERED_TOO_EARLY)
        elif not answers:
            raise _with_usage(
                NoToolCallError(
                    f"{turn.model} did not call {RECORD_ANSWER} when it had to: it called "
                    f"{', '.join(call.name for call in turn.tool_calls) or 'nothing'}."
                ),
                turns,
                raised_here=True,
            )
        else:
            draft, failure = _first_valid(answers)
            if draft is not None:
                return _response(question, chunks, draft, turns, records, started)
            if correcting:
                raise _with_usage(
                    InvalidToolInputError(
                        f"{turn.model} twice recorded an answer that did not validate:\n{failure}"
                    ),
                    turns,
                    raised_here=True,
                )
            correcting = True
            for call in others:
                results[call.id] = _error_result(call.id, NOT_RUN_ANSWER_DUE)
            for position, call in enumerate(answers):
                results[call.id] = _error_result(
                    call.id,
                    f"That did not fit the schema. Fix these problems and call {RECORD_ANSWER} "
                    f"again:\n{failure}"
                    if position == 0
                    else f"This did not fit the schema either, and only one call to "
                    f"{RECORD_ANSWER} is expected.",
                )

        # The API refuses a turn that leaves any tool call of the reply before it unanswered,
        # so every call gets a result, in the order the model made them.
        messages = [
            *messages,
            {"role": "assistant", "content": turn.content},
            {"role": "user", "content": [results[call.id] for call in turn.tool_calls]},
        ]


def _run_tool_call(
    call: ToolCall, today: date, tracer: Tracer
) -> tuple[ToolCallRecord, dict[str, Any]]:
    """Run one tool call in a `tool` span; return its record and the result block for the model."""
    started = time.perf_counter()
    with tracer.span("tool", call.name, input=call.arguments) as span:
        try:
            result = run_tool(call.name, call.arguments, today=today)
        except ToolError as error:
            span.end(error=error, units=1)
            record = ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                output=None,
                error=str(error),
                pages=[],
                source_text=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return record, _error_result(call.id, str(error))
        span.end(output=result.output, units=1)
    record = ToolCallRecord(
        name=call.name,
        arguments=call.arguments,
        output=result.output,
        error=None,
        pages=result.pages,
        source_text=result.source_text,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    block = {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": json.dumps(result.output, ensure_ascii=False),
    }
    return record, block


def _error_result(call_id: str, message: str) -> dict[str, Any]:
    """A `tool_result` block telling the model its call failed, and why."""
    return {"type": "tool_result", "tool_use_id": call_id, "is_error": True, "content": message}


def _first_valid(answers: list[ToolCall]) -> tuple[DraftAnswer | None, ValidationError | None]:
    """The first `record_answer` call that validates, or the first call's validation error."""
    first_failure: ValidationError | None = None
    for call in answers:
        try:
            return DraftAnswer.model_validate(call.arguments), None
        except ValidationError as error:
            first_failure = first_failure or error
    return None, first_failure


def _with_usage[E: ChatModelError](
    error: E, turns: list[ToolTurn], *, raised_here: bool = False
) -> E:
    """`error`, carrying the tokens of every turn so far, as `ChatModelError` promises.

    `turns` are the turns that came back. An error the loop raises itself (`raised_here`) is
    about them, so it carries exactly their tokens. An error from the model already counts its
    own request, which is not in `turns`, so theirs are added to it; one that came without a
    count keeps None, since a partial sum would read as the whole bill.
    """
    spent_in = sum(turn.input_tokens for turn in turns)
    spent_out = sum(turn.output_tokens for turn in turns)
    if raised_here:
        error.input_tokens, error.output_tokens = spent_in, spent_out
    elif error.input_tokens is not None and error.output_tokens is not None:
        error.input_tokens += spent_in
        error.output_tokens += spent_out
    return error


def _response(
    question: str,
    chunks: list[RetrievedChunk],
    draft: DraftAnswer,
    turns: list[ToolTurn],
    records: list[ToolCallRecord],
    started: float,
) -> AgentResponse:
    """The checked answer, with tokens summed over the turns and every tool call listed."""
    citations, dropped = check_citations(draft.citations, chunks)
    return AgentResponse(
        question=question,
        answer=draft.answer,
        found_in_handbook=draft.found_in_handbook,
        citations=citations,
        dropped_citations=dropped,
        confidence=draft.confidence,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
        model=turns[-1].model,
        input_tokens=sum(turn.input_tokens for turn in turns),
        output_tokens=sum(turn.output_tokens for turn in turns),
        latency_ms=(time.perf_counter() - started) * 1000,
        fallback_used=any(turn.fallback_used for turn in turns),
        tool_calls=records,
        turns=len(turns),
    )
