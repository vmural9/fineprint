"""Failures on demand, so the service's way of coping with them can be seen working.

Setting `FAULT_INJECTION` makes one of the three model calls fail the way Bedrock does when it
is struggling: `chat:unavailable` for throttling or an outage, `rerank:timeout` for a call that
never returns. Add `:once` and only the first call fails, which is how a recovery is shown
(`chat:unavailable:once`: the first question falls back, the next is answered normally).

The wrappers raise the same typed `ProviderError`s the Bedrock classes do, before the real model
is called, so nothing is billed. The factory puts them inside the traced wrappers, so an injected
failure shows up in a trace exactly as a real one does. This is a test and demo switch: it is
empty by default and nothing sets it in production.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel

from fineprint.providers.base import (
    ChatModel,
    Embedder,
    EmbeddingResult,
    LLMResult,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Reranker,
    RerankResult,
    ToolCallingModel,
    ToolSpec,
    ToolTurn,
)

Fault = Literal["timeout", "unavailable"]
Target = Literal["chat", "embedding", "rerank"]

ACCEPTED = (
    "FAULT_INJECTION is <target>:<fault> or <target>:<fault>:once, with target one of "
    f"{', '.join(get_args(Target))} and fault one of {', '.join(get_args(Fault))}, "
    "for example chat:unavailable:once. Leave it empty for no faults."
)


@dataclass(frozen=True)
class FaultSpec:
    """Which call fails, how, and whether only the first one does."""

    target: Target
    fault: Fault
    once: bool


def parse_fault(text: str) -> FaultSpec | None:
    """The fault `text` asks for, or None if it is empty. Raises ValueError on anything else."""
    text = text.strip()
    if not text:
        return None
    parts = text.split(":")
    target, fault = parts[0], parts[1] if len(parts) > 1 else ""
    once = len(parts) == 3 and parts[2] == "once"
    well_formed = len(parts) == 2 or once
    if not well_formed or target not in get_args(Target) or fault not in get_args(Fault):
        raise ValueError(f"FAULT_INJECTION={text!r} is not understood. {ACCEPTED}")
    return FaultSpec(target=target, fault=fault, once=once)  # type: ignore[arg-type]


class _Switch:
    """Raises the spec's error when told to, and remembers whether it has already done so."""

    def __init__(self, spec: FaultSpec) -> None:
        self.spec = spec
        self._fired = False

    def check(self) -> None:
        if self.spec.once and self._fired:
            return
        self._fired = True
        description = f"injected {self.spec.fault} fault on {self.spec.target}"
        error: ProviderError = (
            ProviderTimeoutError(description)
            if self.spec.fault == "timeout"
            else ProviderUnavailableError(description)
        )
        raise error


class FaultyChatModel:
    """A chat model that fails as the spec says before it calls the real one."""

    def __init__(self, inner: ChatModel | ToolCallingModel, spec: FaultSpec) -> None:
        self._inner = inner
        self._switch = _Switch(spec)
        self.model = getattr(inner, "model", None)

    def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        self._switch.check()
        return self._inner.complete_structured(system, user, schema)  # type: ignore[union-attr]

    def complete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec],
        *,
        force_tool: str | None = None,
    ) -> ToolTurn:
        self._switch.check()
        return self._inner.complete_with_tools(  # type: ignore[union-attr]
            system, messages, tools, force_tool=force_tool
        )


class FaultyEmbedder:
    """An embedder that fails as the spec says before it calls the real one."""

    def __init__(self, inner: Embedder, spec: FaultSpec) -> None:
        self._inner = inner
        self._switch = _Switch(spec)
        self.model = getattr(inner, "model", None)
        self.dimension = inner.dimension

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self._switch.check()
        return self._inner.embed(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts).vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text]).vectors[0]


class FaultyReranker:
    """A re-ranker that fails as the spec says before it calls the real one."""

    def __init__(self, inner: Reranker, spec: FaultSpec) -> None:
        self._inner = inner
        self._switch = _Switch(spec)
        self.model = inner.model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        self._switch.check()
        return self._inner.rerank(query, documents, top_n)
