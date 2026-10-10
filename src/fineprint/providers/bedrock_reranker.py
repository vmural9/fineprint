"""Cohere Rerank 3.5, reached through the Bedrock Rerank API.

A re-ranker reads the question beside each passage and scores how well that passage answers
it. Reading pairs costs far more than comparing vectors, so it is done only for a short list:
retrieval finds candidates cheaply, by shared words and by nearby vectors, and the re-ranker
puts those few in a new order before the top-k cut.

The request below was settled by a live probe against Bedrock in us-west-2 on 2026-09-26. The
operation is `rerank` on the `bedrock-agent-runtime` client, not on `bedrock-runtime` where the
embedding and chat models are called. The model is named by its foundation-model ARN rather
than its bare id. The response lists `{"index", "relevanceScore"}` pairs, best first, with
scores between 0 and 1.
"""

import threading
from typing import Any

import boto3
from botocore.config import Config

from fineprint.providers.base import RerankResult
from fineprint.providers.bedrock_errors import BOTO_ERRORS, map_boto_error

# Seconds one request may wait for the reply, and botocore's attempts per call (the first try
# included). The connect timeout is fixed at 5 seconds. Settings carry the real values.
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_ATTEMPTS = 4
CONNECT_TIMEOUT = 5


class RerankResponseError(RuntimeError):
    """Bedrock answered with something this code cannot read as a ranking."""


class BedrockReranker:
    """Re-ranks passages with `cohere.rerank-v3-5:0` through `bedrock-agent-runtime`."""

    def __init__(
        self,
        model: str,
        region: str,
        *,
        client: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        """Take the model and region from settings; take a client only in tests.

        `client` is the seam the unit tests use: pass a stub and nothing touches the network.
        """
        self.model = model
        self.region = region
        # Foundation models belong to no account, so the account field of the ARN stays empty.
        self.model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model}"
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def client(self) -> Any:
        """The `bedrock-agent-runtime` client, built on first use.

        Built late, as the embedder's is, so that constructing a re-ranker needs no AWS
        credentials; and under a lock, so that two requests arriving together build one client
        between them. There is no static type for a boto3 client.
        """
        with self._client_lock:
            if self._client is None:
                self._client = boto3.client(
                    "bedrock-agent-runtime",
                    region_name=self.region,
                    config=Config(
                        connect_timeout=CONNECT_TIMEOUT,
                        read_timeout=self.timeout,
                        retries={"max_attempts": self.max_attempts, "mode": "standard"},
                    ),
                )
            return self._client

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        """The `top_n` passages that best answer `query`, best first, by position in `documents`."""
        if not documents or top_n < 1:
            # The Rerank API accepts neither zero passages nor zero results, and the answer
            # would be an empty list anyway.
            return []

        try:
            response = self.client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {
                            "type": "TEXT",
                            "textDocument": {"text": document},
                        },
                    }
                    for document in documents
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {"modelArn": self.model_arn},
                        # Asking for more results than there are passages fails the whole call:
                        # "Cannot provide numberOfResults value that exceeds the number of sources."
                        "numberOfResults": min(top_n, len(documents)),
                    },
                },
            )
        except BOTO_ERRORS as error:
            raise map_boto_error(error, self.model) from error
        results = self._read_results(response, len(documents))

        # Bedrock already sends its results best first and no more than were asked for. Sorting
        # and cutting here keeps the Reranker protocol's promise even if it one day does not.
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_n]

    def _read_results(self, response: dict[str, Any], passage_count: int) -> list[RerankResult]:
        """Every `{"index", "relevanceScore"}` pair as a `RerankResult`, each one checked."""
        items = response.get("results")
        if not isinstance(items, list):
            raise RerankResponseError(
                f"{self.model} returned no list of results; 'results' held {items!r}."
            )

        results: list[RerankResult] = []
        seen: set[int] = set()
        for item in items:
            index = item.get("index") if isinstance(item, dict) else None
            score = item.get("relevanceScore") if isinstance(item, dict) else None
            if not isinstance(index, int) or not isinstance(score, int | float):
                raise RerankResponseError(
                    f"{self.model} returned a result without an integer 'index' and a numeric "
                    f"'relevanceScore': {item!r}."
                )
            # An index that points at no passage, or at one already ranked, would attach a
            # score to the wrong chunk further up the line.
            if not 0 <= index < passage_count:
                raise RerankResponseError(
                    f"{self.model} returned index {index} for {passage_count} passages; an "
                    f"index must be between 0 and {passage_count - 1}."
                )
            if index in seen:
                raise RerankResponseError(f"{self.model} returned index {index} twice.")
            seen.add(index)
            results.append(RerankResult(index=index, score=float(score)))
        return results
