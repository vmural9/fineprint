"""Turns the failures boto3 raises into the service's `ProviderError`s.

The embedder and the re-ranker both call Bedrock through boto3, and both fail in the same ways,
so the mapping lives here once. botocore has already retried by the time one of these arrives,
so a mapped error means the retries ran out.
"""

from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
from botocore.exceptions import ConnectionError as BotocoreConnectionError

from fineprint.providers.base import (
    ProviderError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

# What a caller catches before asking `map_boto_error` what to raise instead. Anything else, such
# as missing credentials or a malformed request, is a bug or a setup problem and is left alone.
BOTO_ERRORS = (ClientError, ReadTimeoutError, BotocoreConnectionError)

# Bedrock error codes that mean "not now" rather than "not this request".
_TIMEOUT_CODES = {"ModelTimeoutException"}
_UNAVAILABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelErrorException",
    "BadGatewayException",
    "DependencyFailedException",
}


def map_boto_error(error: Exception, model: str) -> ProviderError:
    """The `ProviderError` for a boto3 failure, with `model` named in its message.

    The caller raises the result `from` the original, so the AWS detail stays in the traceback.
    """
    # A connect timeout is also a botocore ConnectionError, so timeouts are checked first.
    if isinstance(error, ReadTimeoutError | ConnectTimeoutError):
        return ProviderTimeoutError(f"{model} timed out: {error}")
    if isinstance(error, BotocoreConnectionError):
        return ProviderUnavailableError(f"{model} could not be reached: {error}")
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        if code in _TIMEOUT_CODES:
            return ProviderTimeoutError(f"{model} timed out ({code}): {error}")
        if code in _UNAVAILABLE_CODES:
            return ProviderUnavailableError(f"{model} is unavailable ({code}): {error}")
        return ProviderRequestError(f"{model} refused the request ({code or 'no code'}): {error}")
    raise TypeError(f"not a boto3 failure this module maps: {error!r}")
