"""The Titan embedder, exercised against a stub so no test touches AWS.

One test uses botocore's own `Stubber`, which validates the call against the real
`bedrock-runtime` service model: it fails if we misspell a parameter the way a hand-written
fake never would.
"""

import io
import json
import time

import boto3
import pytest
from botocore.stub import Stubber

from fineprint.providers.bedrock_embedder import BedrockEmbedder, EmbeddingResponseError

MODEL = "amazon.titan-embed-text-v2:0"


def titan_body(vector: list[float]) -> bytes:
    """A response body shaped like Titan's."""
    return json.dumps({"embedding": vector, "inputTextTokenCount": 7}).encode("utf-8")


class FakeBedrockRuntime:
    """The one method the embedder calls, plus a record of how it was called."""

    def __init__(self, *, payload: dict | None = None, delays: dict[str, float] | None = None):
        self.payload = payload
        self.delays = delays or {}
        self.calls: list[dict] = []

    def invoke_model(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        text = json.loads(str(kwargs["body"]))["inputText"]
        time.sleep(self.delays.get(text, 0.0))
        if self.payload is not None:
            body = json.dumps(self.payload).encode("utf-8")
        else:
            # A vector whose first number identifies the text, so order is checkable.
            body = titan_body([float(len(text))] * 1024)
        return {"body": io.BytesIO(body), "contentType": "application/json"}


def test_embed_query_returns_a_vector_of_the_declared_dimension():
    embedder = BedrockEmbedder(model=MODEL, region="us-west-2", client=FakeBedrockRuntime())

    vector = embedder.embed_query("What is the Part B premium?")

    assert embedder.dimension == 1024
    assert len(vector) == embedder.dimension


def test_the_request_carries_the_model_id_and_asks_for_normalized_vectors():
    client = FakeBedrockRuntime()
    embedder = BedrockEmbedder(model=MODEL, region="us-west-2", client=client)

    embedder.embed_query("Does Medicare cover dental?")

    (call,) = client.calls
    assert call["modelId"] == MODEL
    assert json.loads(str(call["body"])) == {
        "inputText": "Does Medicare cover dental?",
        "dimensions": 1024,
        "normalize": True,
    }


def test_embed_documents_returns_one_vector_per_text_in_the_input_order():
    # Titan embeds one text per call, so the embedder fans out across threads. The first text
    # is the slowest, so a result list built in completion order would come back reversed.
    texts = ["a", "bb", "ccc", "dddd"]
    delays = {"a": 0.20, "bb": 0.15, "ccc": 0.10, "dddd": 0.0}
    embedder = BedrockEmbedder(
        model=MODEL, region="us-west-2", client=FakeBedrockRuntime(delays=delays)
    )

    vectors = embedder.embed_documents(texts)

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0]


def test_embed_documents_with_no_texts_makes_no_calls():
    client = FakeBedrockRuntime()
    embedder = BedrockEmbedder(model=MODEL, region="us-west-2", client=client)

    assert embedder.embed_documents([]) == []
    assert client.calls == []


def test_a_response_without_an_embedding_says_so_clearly():
    client = FakeBedrockRuntime(payload={"message": "something else entirely"})
    embedder = BedrockEmbedder(model=MODEL, region="us-west-2", client=client)

    with pytest.raises(EmbeddingResponseError) as raised:
        embedder.embed_query("What is the Part B premium?")

    assert MODEL in str(raised.value)
    assert "embedding" in str(raised.value)


def test_a_vector_of_the_wrong_width_says_so_clearly():
    # The chunks.embedding column is vector(1024); a narrower vector must not reach the database.
    client = FakeBedrockRuntime(payload={"embedding": [0.1] * 512})
    embedder = BedrockEmbedder(model=MODEL, region="us-west-2", client=client)

    with pytest.raises(EmbeddingResponseError) as raised:
        embedder.embed_query("What is the Part B premium?")

    assert "512" in str(raised.value)
    assert "1024" in str(raised.value)


def test_the_request_is_valid_for_the_real_bedrock_runtime_api():
    # botocore validates the parameters against the service model, so a typo in `modelId`
    # or `contentType` fails here rather than in production.
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    body = titan_body([0.5] * 1024)
    with Stubber(client) as stubber:
        stubber.add_response(
            "invoke_model",
            {"body": io.BytesIO(body), "contentType": "application/json"},
            {
                "modelId": MODEL,
                "body": json.dumps({"inputText": "hello", "dimensions": 1024, "normalize": True}),
                "accept": "application/json",
                "contentType": "application/json",
            },
        )
        embedder = BedrockEmbedder(model=MODEL, region="us-west-2", client=client)

        assert len(embedder.embed_query("hello")) == 1024
        stubber.assert_no_pending_responses()
