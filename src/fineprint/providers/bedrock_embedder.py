"""Amazon Titan Text Embeddings V2, reached through Bedrock.

An embedding is a list of numbers that stands for a piece of text: texts about the same
thing end up close together, so a question about "the Part B premium" finds the handbook
paragraph that says "$202.90 each month" even though they share few words. Titan V2 returns
1024 numbers, normalized to unit length, which is why `chunks.embedding` is `vector(1024)`
and why cosine distance in Postgres compares them fairly.

Titan embeds one text per call, so `embed_documents` fans the calls out over a small thread
pool. Ingesting a few hundred chunks then takes about a minute instead of ten, and the
results stay in the order they were asked for.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
from botocore.config import Config

from fineprint.providers.base import EmbeddingResult
from fineprint.providers.bedrock_errors import BOTO_ERRORS, map_boto_error

# Titan V2 can return 1024, 512 or 256 numbers. 1024 is the widest, the default, and the
# width `schema.sql` declares; changing it means a fresh database and a full re-ingest.
TITAN_DIMENSION = 1024

# Enough parallelism to keep ingestion to about a minute, low enough not to be throttled.
EMBED_WORKERS = 8

# Seconds one request may wait for Titan's reply, and botocore's attempts per call (the first
# try included). The connect timeout is fixed at 5 seconds. Settings carry the real values.
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_ATTEMPTS = 4
CONNECT_TIMEOUT = 5


class EmbeddingResponseError(RuntimeError):
    """Bedrock answered with something this code cannot read as an embedding."""


class BedrockEmbedder:
    """Embeds text with `amazon.titan-embed-text-v2:0` through `bedrock-runtime`."""

    def __init__(
        self,
        model: str,
        region: str,
        *,
        client: Any | None = None,
        dimension: int = TITAN_DIMENSION,
        workers: int = EMBED_WORKERS,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        """Take the model and region from settings; take a client only in tests.

        `client` is the seam the unit tests use: pass a stub and nothing touches the network.
        """
        self.model = model
        self.region = region
        self.dimension = dimension
        self.workers = workers
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def client(self) -> Any:
        """The `bedrock-runtime` client, built on first use.

        Built late so that constructing an embedder needs no AWS credentials: boto3 spends
        seconds hunting for credentials it will not find, and the factory's tests should not
        pay that. boto3 reads `AWS_PROFILE` and the rest of the credential chain itself;
        only the region comes from our settings. There is no static type for a boto3 client.

        The timeouts and the retry mode are set here so that a hung request fails in seconds
        and is retried by botocore's standard mode before the caller ever sees an error.
        """
        with self._client_lock:
            if self._client is None:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                    config=Config(
                        connect_timeout=CONNECT_TIMEOUT,
                        read_timeout=self.timeout,
                        retries={"max_attempts": self.max_attempts, "mode": "standard"},
                    ),
                )
            return self._client

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """The vectors for `texts`, in order, with the tokens Titan read and the time taken."""
        started = time.perf_counter()
        if not texts:
            return EmbeddingResult(vectors=[], input_tokens=0, latency_ms=0.0)
        if len(texts) == 1:
            embedded = [self._embed(texts[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(texts))) as pool:
                # `map` yields results in the order of the inputs, not of completion, which is
                # what keeps every vector attached to its own chunk.
                embedded = list(pool.map(self._embed, texts))
        counts = [tokens for _, tokens in embedded]
        return EmbeddingResult(
            vectors=[vector for vector, _ in embedded],
            input_tokens=None if None in counts else sum(counts),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def embed_query(self, text: str) -> list[float]:
        """The vector for one question."""
        return self.embed([text]).vectors[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """The vectors for many chunks, in the order the chunks were given."""
        return self.embed(texts).vectors

    def _embed(self, text: str) -> tuple[list[float], int | None]:
        """One `invoke_model` call: the vector read back out of its JSON body, and its tokens."""
        try:
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(
                    {"inputText": text, "dimensions": self.dimension, "normalize": True}
                ),
                accept="application/json",
                contentType="application/json",
            )
        except BOTO_ERRORS as error:
            raise map_boto_error(error, self.model) from error
        # The body is a stream, so reading it can time out or lose the connection just as the
        # call can; that failure is mapped the same way.
        try:
            raw = response["body"].read()
        except BOTO_ERRORS as error:
            raise map_boto_error(error, self.model) from error
        except (KeyError, TypeError) as error:
            raise EmbeddingResponseError(
                f"{self.model} returned a response with no readable body: {error!r}"
            ) from error
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise EmbeddingResponseError(
                f"{self.model} returned a body this code could not read as JSON: {error}"
            ) from error

        vector = payload.get("embedding")
        if not isinstance(vector, list):
            raise EmbeddingResponseError(
                f"{self.model} returned no 'embedding' field; the response held {sorted(payload)}."
            )
        if len(vector) != self.dimension:
            raise EmbeddingResponseError(
                f"{self.model} returned {len(vector)} numbers, not the {self.dimension} that "
                "the chunks.embedding column holds."
            )
        tokens = payload.get("inputTextTokenCount")
        return [float(value) for value in vector], tokens if isinstance(tokens, int) else None
