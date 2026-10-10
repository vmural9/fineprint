"""How botocore's failures become the service's `ProviderError`s, one case per mapped class."""

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ProxyConnectionError,
    ReadTimeoutError,
)
from botocore.exceptions import ConnectionError as BotocoreConnectionError

from fineprint.providers.base import (
    ProviderError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from fineprint.providers.bedrock_errors import BOTO_ERRORS, map_boto_error

MODEL = "amazon.titan-embed-text-v2:0"


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "from the test"}}, "InvokeModel")


MAPPED = [
    (ReadTimeoutError(endpoint_url="https://bedrock.test"), ProviderTimeoutError),
    (ConnectTimeoutError(endpoint_url="https://bedrock.test"), ProviderTimeoutError),
    (EndpointConnectionError(endpoint_url="https://bedrock.test"), ProviderUnavailableError),
    (ProxyConnectionError(proxy_url="https://proxy.test"), ProviderUnavailableError),
    (BotocoreConnectionError(error="reset"), ProviderUnavailableError),
    (client_error("ModelTimeoutException"), ProviderTimeoutError),
    (client_error("ThrottlingException"), ProviderUnavailableError),
    (client_error("ServiceUnavailableException"), ProviderUnavailableError),
    (client_error("InternalServerException"), ProviderUnavailableError),
    (client_error("ModelNotReadyException"), ProviderUnavailableError),
    (client_error("ModelErrorException"), ProviderUnavailableError),
    (client_error("BadGatewayException"), ProviderUnavailableError),
    (client_error("DependencyFailedException"), ProviderUnavailableError),
    (client_error("ValidationException"), ProviderRequestError),
    (client_error("AccessDeniedException"), ProviderRequestError),
    (client_error("ResourceNotFoundException"), ProviderRequestError),
    (client_error("ServiceQuotaExceededException"), ProviderRequestError),
    (client_error("SomethingNobodyListed"), ProviderRequestError),
]


@pytest.mark.parametrize(("original", "expected"), MAPPED, ids=lambda case: type(case).__name__)
def test_each_botocore_failure_becomes_its_provider_error(original, expected):
    mapped = map_boto_error(original, MODEL)

    # Not isinstance: the three subclasses are siblings, and a sibling must not pass for another.
    assert type(mapped) is expected
    assert isinstance(mapped, ProviderError)
    assert MODEL in str(mapped)


def test_the_client_error_code_is_named_in_the_message():
    assert "ThrottlingException" in str(map_boto_error(client_error("ThrottlingException"), MODEL))


def test_every_mapped_class_is_caught_by_boto_errors():
    for original, _ in MAPPED:
        assert isinstance(original, BOTO_ERRORS)


def test_a_connect_timeout_is_a_timeout_although_it_is_also_a_connection_error():
    # ConnectTimeoutError subclasses botocore's ConnectionError, which maps to "unavailable",
    # so the timeout check has to come first.
    original = ConnectTimeoutError(endpoint_url="https://bedrock.test")

    assert isinstance(original, BotocoreConnectionError)
    assert type(map_boto_error(original, MODEL)) is ProviderTimeoutError
