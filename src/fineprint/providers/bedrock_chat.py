"""Claude Opus 5 through Bedrock, answering with an object the service has validated.

Free text would let the model invent a page number or a dollar amount and leave the service
no way to tell. So the model is given one tool whose input schema is the Pydantic model the
caller asked for, and its answer is the tool's arguments, which Pydantic then validates. A
draft that does not validate is sent back once with the validation error attached. A reply can
hold several calls to the tool at once; the first that validates is the answer.

Why tool use and not the API's own structured output, and why these exact request fields,
was settled by live probes against Bedrock in us-west-2 on 2026-09-21:

- `client.messages.parse(..., output_format=...)` -> 400 `output_config.format: Extra inputs
  are not permitted`.
- a tool with `strict: true` -> 400 `tools.0.custom.strict: Extra inputs are not permitted`,
  with either tool_choice.
- a plain tool with `tool_choice={"type": "tool", ...}` -> works, `stop_reason="tool_use"`.
- the bare model id `anthropic.claude-opus-5` is refused ("on-demand throughput isn't
  supported"); the `us.` inference profile is what works.
- `temperature` and the other sampling parameters are refused by this model, so none is sent.
"""

import time
from typing import Any

from anthropic import AnthropicBedrock
from pydantic import BaseModel, ValidationError

from fineprint.providers.base import LLMResult

# The single tool the model answers through. The name shows up in its refusals, so it reads
# like something a careful assistant would call.
TOOL_NAME = "record_answer"

# A ceiling, not a target: large enough that a long cited answer is never cut off.
MAX_TOKENS = 16_000

# The SDK retries throttling and 5xx with exponential backoff; Bedrock throttles readily
# during an eval run, so allow a few more attempts than the SDK's default of 2.
MAX_RETRIES = 4


class ChatModelError(RuntimeError):
    """The model did not give back an answer this service can use."""


class ModelRefusedError(ChatModelError):
    """The model declined to answer at all."""


class ResponseTruncatedError(ChatModelError):
    """The model ran into `max_tokens` before it finished its answer."""


class NoToolCallError(ChatModelError):
    """The model replied with something other than a call to the answer tool."""


class InvalidToolInputError(ChatModelError):
    """The model's answer did not fit the schema, twice."""


class BedrockChatModel:
    """Asks `us.anthropic.claude-opus-5` for an answer shaped like a Pydantic model."""

    def __init__(
        self,
        model: str,
        region: str,
        *,
        client: Any | None = None,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = MAX_RETRIES,
    ):
        """Take the model and region from settings; take a client only in tests."""
        self.model = model
        self.region = region
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = client

    @property
    def client(self) -> Any:
        """The Anthropic client for Bedrock, built on first use.

        `AnthropicBedrock` is the standard runtime endpoint. `AnthropicBedrockMantle` serves
        Haiku 4.5 in us-west-2 but 404s for Opus 5, which is why it is not used here.
        Credentials come from the usual AWS chain, so only the region is passed in.
        """
        if self._client is None:
            self._client = AnthropicBedrock(aws_region=self.region, max_retries=self.max_retries)
        return self._client

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        """Answer `user` under the rules in `system`, as a validated instance of `schema`."""
        tool = {
            "name": TOOL_NAME,
            "description": f"Record your answer as a {schema.__name__}. Every field is required.",
            "input_schema": schema.model_json_schema(),
        }
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        started = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        failure: ValidationError | None = None

        # Two attempts: the first ask, then one retry carrying the validation error back.
        for _ in range(2):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                tools=[tool],
                # Forced, so the model cannot answer in prose. Accepted on this model even
                # though it thinks by default; probed on 2026-09-21.
                tool_choice={"type": "tool", "name": TOOL_NAME},
            )
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            # Why the answer stopped comes before what the answer says: a refused or
            # truncated response still carries content, and it must not be trusted.
            self._check_stop_reason(response)
            calls = self._tool_calls(response)

            try:
                parsed = self._first_valid(calls, schema)
            except ValidationError as error:
                failure = error
                messages = [
                    messages[0],
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": self._tool_results(response, calls[0], error)},
                ]
                continue

            return LLMResult(
                parsed=parsed,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        raise InvalidToolInputError(
            f"{self.model} twice returned a {schema.__name__} that did not validate:\n{failure}"
        )

    def _check_stop_reason(self, response: Any) -> None:
        """Raise when the model stopped for a reason that makes its content unusable."""
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            reason = f" ({details.category}): {details.explanation}" if details else "."
            raise ModelRefusedError(f"{self.model} refused to answer{reason}")
        if response.stop_reason == "max_tokens":
            raise ResponseTruncatedError(
                f"{self.model} hit max_tokens ({self.max_tokens}) before finishing its answer. "
                "Send fewer excerpts or raise max_tokens."
            )

    def _tool_calls(self, response: Any) -> list[Any]:
        """The `tool_use` blocks calling the answer tool, in order, or a clear error if none came.

        Usually there is one. The model can also make several calls at once, each carrying a
        whole answer of its own.
        """
        calls = [
            block
            for block in response.content
            if block.type == "tool_use" and block.name == TOOL_NAME
        ]
        if calls:
            return calls
        blocks = [block.type for block in response.content] or ["nothing"]
        raise NoToolCallError(
            f"{self.model} did not call {TOOL_NAME}: it stopped with "
            f"stop_reason={response.stop_reason!r} and returned {', '.join(blocks)}."
        )

    def _first_valid[T: BaseModel](self, calls: list[Any], schema: type[T]) -> T:
        """Parse the first call whose input fits `schema`, or raise the first call's error.

        A call that validates is kept even when an earlier call in the same reply did not:
        sending the reply back would cost a second request for an answer already in hand.
        """
        errors: list[ValidationError] = []
        for call in calls:
            try:
                return schema.model_validate(call.input)
            except ValidationError as error:
                errors.append(error)
        raise errors[0]

    def _tool_results(
        self, response: Any, failed: Any, error: ValidationError
    ) -> list[dict[str, Any]]:
        """A `tool_result` for each `tool_use` block in `response`, to go in the retry turn.

        The API refuses that turn with a 400, "tool_use ids were found without tool_result
        blocks", if any `tool_use` block in the reply before it goes unanswered. The call that
        failed validation gets its error back; every other call is told it was ignored, since
        one call is expected.
        """
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.id == failed.id:
                content = (
                    f"That did not fit the schema. Fix these problems and call "
                    f"{TOOL_NAME} again:\n{error}"
                )
            else:
                content = f"Only one call to {TOOL_NAME} is expected, so this one was ignored."
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": content,
                }
            )
        return results
