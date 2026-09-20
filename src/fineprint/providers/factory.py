"""Which class each configured provider name means.

One dictionary per protocol. Each entry is a small function that imports its provider module
and builds the object, rather than the class itself: importing a class here would drag boto3
and the anthropic SDK into every module that only wants to read the configuration. A
provider's SDK is imported when that provider is actually chosen, and nowhere else.
"""

from collections.abc import Callable

from fineprint.config import Settings
from fineprint.providers.base import Embedder


class UnknownProviderError(ValueError):
    """The configuration names a provider this service does not implement."""


def _bedrock_embedder(settings: Settings) -> Embedder:
    from fineprint.providers.bedrock_embedder import BedrockEmbedder

    return BedrockEmbedder(model=settings.embedding_model, region=settings.aws_region)


EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {"bedrock": _bedrock_embedder}


def get_embedder(settings: Settings) -> Embedder:
    """The embedder named by `EMBEDDING_PROVIDER`."""
    build = EMBEDDERS.get(settings.embedding_provider)
    if build is None:
        raise UnknownProviderError(
            f"EMBEDDING_PROVIDER={settings.embedding_provider!r} is not implemented. "
            f"Known embedding providers: {', '.join(sorted(EMBEDDERS))}."
        )
    return build(settings)
