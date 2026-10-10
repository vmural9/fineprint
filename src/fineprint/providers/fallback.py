"""A second answer model for when the first is not there.

Claude Opus 5 answers every question. When Bedrock cannot serve it (throttled past the SDK's
retries, overloaded, unreachable, or too slow) a person asking about their Part B premium is
better served by a second model's answer, marked as such, than by an error page.

`FallbackChatModel` wraps two models and sends a call to the second only for those availability
failures: `ProviderTimeoutError` and `ProviderUnavailableError`. Every other failure goes to the
caller untouched. A refusal, a cut-off reply or a malformed tool call would happen again on the
second model, and a 4xx means the request itself is wrong, so trying again would only spend
money to learn the same thing.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from fineprint.providers.base import (
    ChatModel,
    LLMResult,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCallingModel,
    ToolSpec,
    ToolTurn,
)

# The failures that mean "this model is not available right now", and nothing else.
FALLBACK_ON = (ProviderTimeoutError, ProviderUnavailableError)


class FallbackChatModel:
    """Asks `primary`; when it is unavailable, asks `fallback` instead.

    The reply from the fallback says `fallback_used=True` and carries the fallback's own model
    id. If the fallback fails too, its error is raised with the primary's as the cause, so a
    log shows both. A tool turn from the fallback carries the same flag.
    """

    def __init__(
        self, primary: ChatModel | ToolCallingModel, fallback: ChatModel | ToolCallingModel
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.model = getattr(primary, "model", None)

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        return self._first_then_second(
            lambda model: model.complete_structured(system, user, schema)  # type: ignore[union-attr]
        )

    def complete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec],
        *,
        force_tool: str | None = None,
    ) -> ToolTurn:
        return self._first_then_second(
            lambda model: model.complete_with_tools(  # type: ignore[union-attr]
                system, messages, tools, force_tool=force_tool
            )
        )

    def _first_then_second[R](self, call: Callable[[Any], R]) -> R:
        """`call(primary)`, or `call(fallback)` marked as a fallback when the primary is down."""
        try:
            return call(self.primary)
        except FALLBACK_ON as primary_error:
            try:
                result = call(self.fallback)
            except Exception as fallback_error:
                raise fallback_error from primary_error
        return replace(result, fallback_used=True)  # type: ignore[type-var]
