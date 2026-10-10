"""Which class each configured provider name means.

Three dictionaries, one per protocol. Each entry is a small function that imports its provider
module and builds the object, rather than the class itself: importing a class here would drag
boto3 and the anthropic SDK into every module that only wants to read the configuration. A
provider's SDK is imported when that provider is actually chosen, and nowhere else.

Each `get_*` builder takes an optional tracer. Given one, it hands back the model wrapped in the
matching traced class, so every call is timed, priced and traced; `get_tracer` builds the tracer.

Two more layers are added from the settings. `FAULT_INJECTION` wraps the one model it names in a
wrapper that fails on demand, inside the traced wrapper so an injected failure is traced like a
real one. `FALLBACK_LLM_MODEL` puts a second answer model behind the first: a call that finds the
first one unavailable is answered by the second, and both attempts are traced. The chat model is
built as `Fallback(Traced(Faulty?(primary)), Traced(fallback))`.
"""

from collections.abc import Callable

from fineprint.config import Settings
from fineprint.providers.base import ChatModel, Embedder, Reranker
from fineprint.providers.fallback import FallbackChatModel
from fineprint.providers.faults import (
    FaultSpec,
    FaultyChatModel,
    FaultyEmbedder,
    FaultyReranker,
    parse_fault,
)
from fineprint.tracing import TracedChatModel, TracedEmbedder, TracedReranker, Tracer


class UnknownProviderError(ValueError):
    """The configuration names a provider this service does not implement."""


def _bedrock_embedder(settings: Settings) -> Embedder:
    from fineprint.providers.bedrock_embedder import BedrockEmbedder

    return BedrockEmbedder(
        model=settings.embedding_model,
        region=settings.aws_region,
        timeout=settings.aws_timeout_seconds,
        max_attempts=settings.aws_max_attempts,
    )


def _bedrock_chat_model(settings: Settings) -> ChatModel:
    from fineprint.providers.bedrock_chat import BedrockChatModel

    return BedrockChatModel(
        model=settings.llm_model,
        region=settings.aws_region,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
    )


def _bedrock_reranker(settings: Settings) -> Reranker:
    from fineprint.providers.bedrock_reranker import BedrockReranker

    return BedrockReranker(
        model=settings.reranker_model,
        region=settings.aws_region,
        timeout=settings.aws_timeout_seconds,
        max_attempts=settings.aws_max_attempts,
    )


EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {"bedrock": _bedrock_embedder}
CHAT_MODELS: dict[str, Callable[[Settings], ChatModel]] = {"bedrock": _bedrock_chat_model}
RERANKERS: dict[str, Callable[[Settings], Reranker]] = {"bedrock": _bedrock_reranker}


def _fault_for(target: str, settings: Settings) -> FaultSpec | None:
    """The injected fault for `target`, if `FAULT_INJECTION` names it.

    Parsed on every call, so a malformed value fails whichever model is built first, with a
    message saying what is accepted, rather than only when its own target is built.
    """
    spec = parse_fault(settings.fault_injection)
    return spec if spec is not None and spec.target == target else None


def get_tracer(settings: Settings) -> Tracer:
    """The tracer named by `TRACING_PROVIDER`.

    With "langfuse" every trace is also sent to Langfuse; `Settings` has already checked that the
    keys and the URL are there. With "none" nothing leaves the process, but cost and latency are
    still measured. The Langfuse module is imported only here, when it is chosen.
    """
    if settings.tracing_provider != "langfuse":
        return Tracer()
    from fineprint.langfuse_sink import LangfuseSink

    assert settings.langfuse_public_key is not None  # Settings refuses to build without them
    assert settings.langfuse_secret_key is not None
    assert settings.langfuse_base_url is not None
    return Tracer(
        LangfuseSink(
            settings.langfuse_public_key,
            settings.langfuse_secret_key.get_secret_value(),
            settings.langfuse_base_url,
        )
    )


def get_embedder(settings: Settings, tracer: Tracer | None = None) -> Embedder:
    """The embedder named by `EMBEDDING_PROVIDER`, traced when a tracer is given."""
    build = EMBEDDERS.get(settings.embedding_provider)
    if build is None:
        raise UnknownProviderError(
            f"EMBEDDING_PROVIDER={settings.embedding_provider!r} is not implemented. "
            f"Known embedding providers: {', '.join(sorted(EMBEDDERS))}."
        )
    embedder = build(settings)
    fault = _fault_for("embedding", settings)
    if fault is not None:
        embedder = FaultyEmbedder(embedder, fault)
    return embedder if tracer is None else TracedEmbedder(embedder, tracer)


def get_chat_model(settings: Settings, tracer: Tracer | None = None) -> ChatModel:
    """The chat model named by `LLM_PROVIDER`, traced when a tracer is given.

    When `FALLBACK_LLM_MODEL` is set, a second model of the same provider stands behind it and
    answers whenever the first is unavailable. The second is traced as a fallback, so a trace
    that used it holds two generation spans: the first model's, with its error, and this one's.
    """
    build = CHAT_MODELS.get(settings.llm_provider)
    if build is None:
        raise UnknownProviderError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not implemented. "
            f"Known chat providers: {', '.join(sorted(CHAT_MODELS))}."
        )
    fault = _fault_for("chat", settings)
    primary = build(settings)
    if fault is not None:
        primary = FaultyChatModel(primary, fault)
    if tracer is not None:
        primary = TracedChatModel(primary, tracer)
    if not settings.fallback_llm_model:
        return primary

    fallback = build(settings.model_copy(update={"llm_model": settings.fallback_llm_model}))
    if tracer is not None:
        fallback = TracedChatModel(fallback, tracer, fallback=True)
    return FallbackChatModel(primary, fallback)


def get_reranker(settings: Settings, tracer: Tracer | None = None) -> Reranker | None:
    """The re-ranker named by `RERANKER_PROVIDER`, or None when that is "none".

    Re-ranking is the one optional step, so it is the one provider with a value that turns it
    off. Callers skip the step when they get None back. Traced when a tracer is given.
    """
    fault = _fault_for("rerank", settings)  # parsed first, so a malformed value always fails
    if settings.reranker_provider == "none":
        return None
    build = RERANKERS.get(settings.reranker_provider)
    if build is None:
        raise UnknownProviderError(
            f"RERANKER_PROVIDER={settings.reranker_provider!r} is not implemented. "
            f"Known re-ranker providers: none, {', '.join(sorted(RERANKERS))}."
        )
    reranker = build(settings)
    if fault is not None:
        reranker = FaultyReranker(reranker, fault)
    return reranker if tracer is None else TracedReranker(reranker, tracer)
