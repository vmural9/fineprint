"""Which class each configured provider name means.

Three dictionaries, one per protocol. Each entry is a small function that imports its provider
module and builds the object, rather than the class itself: importing a class here would drag
boto3 and the anthropic SDK into every module that only wants to read the configuration. A
provider's SDK is imported when that provider is actually chosen, and nowhere else.
"""

from collections.abc import Callable

from fineprint.config import Settings
from fineprint.providers.base import ChatModel, Embedder, Reranker


class UnknownProviderError(ValueError):
    """The configuration names a provider this service does not implement."""


def _bedrock_embedder(settings: Settings) -> Embedder:
    from fineprint.providers.bedrock_embedder import BedrockEmbedder

    return BedrockEmbedder(model=settings.embedding_model, region=settings.aws_region)


def _bedrock_chat_model(settings: Settings) -> ChatModel:
    from fineprint.providers.bedrock_chat import BedrockChatModel

    return BedrockChatModel(model=settings.llm_model, region=settings.aws_region)


def _bedrock_reranker(settings: Settings) -> Reranker:
    from fineprint.providers.bedrock_reranker import BedrockReranker

    return BedrockReranker(model=settings.reranker_model, region=settings.aws_region)


EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {"bedrock": _bedrock_embedder}
CHAT_MODELS: dict[str, Callable[[Settings], ChatModel]] = {"bedrock": _bedrock_chat_model}
RERANKERS: dict[str, Callable[[Settings], Reranker]] = {"bedrock": _bedrock_reranker}


def get_embedder(settings: Settings) -> Embedder:
    """The embedder named by `EMBEDDING_PROVIDER`."""
    build = EMBEDDERS.get(settings.embedding_provider)
    if build is None:
        raise UnknownProviderError(
            f"EMBEDDING_PROVIDER={settings.embedding_provider!r} is not implemented. "
            f"Known embedding providers: {', '.join(sorted(EMBEDDERS))}."
        )
    return build(settings)


def get_chat_model(settings: Settings) -> ChatModel:
    """The chat model named by `LLM_PROVIDER`."""
    build = CHAT_MODELS.get(settings.llm_provider)
    if build is None:
        raise UnknownProviderError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not implemented. "
            f"Known chat providers: {', '.join(sorted(CHAT_MODELS))}."
        )
    return build(settings)


def get_reranker(settings: Settings) -> Reranker | None:
    """The re-ranker named by `RERANKER_PROVIDER`, or None when that is "none".

    Re-ranking is the one optional step, so it is the one provider with a value that turns it
    off. Callers skip the step when they get None back.
    """
    if settings.reranker_provider == "none":
        return None
    build = RERANKERS.get(settings.reranker_provider)
    if build is None:
        raise UnknownProviderError(
            f"RERANKER_PROVIDER={settings.reranker_provider!r} is not implemented. "
            f"Known re-ranker providers: none, {', '.join(sorted(RERANKERS))}."
        )
    return build(settings)
