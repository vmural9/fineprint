"""The Bedrock re-ranker, exercised against a stub so no test touches AWS.

One test uses botocore's own `Stubber`, which validates the call against the real
`bedrock-agent-runtime` service model: it fails if we misspell a parameter the way a
hand-written fake never would.
"""

import boto3
import pytest
from botocore.stub import Stubber

from fineprint.providers.base import RerankResult
from fineprint.providers.bedrock_reranker import BedrockReranker, RerankResponseError

MODEL = "cohere.rerank-v3-5:0"
MODEL_ARN = "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0"

# Question q004 of the golden set, and three passages copied from the handbook's pages 40, 53
# and 42. Only the last passage answers the question.
QUESTION = (
    "My mother is having trouble hearing. Does Original Medicare pay for hearing aids, or for "
    "the exam to fit them?"
)
DOCUMENTS = [
    "Medicare covers medically necessary items like oxygen and oxygen equipment, walkers, and "
    "hospital beds when a doctor or other health care provider orders them for use in the home.",
    "Medicare may cover medically necessary ambulance transportation to a foreign hospital only "
    "with admission for medically necessary covered inpatient hospital services.",
    "Note: Original Medicare doesn’t cover hearing aids or exams for fitting hearing aids.",
]


def source(text: str) -> dict:
    """One passage as the Rerank API wants it: an inline text document."""
    return {
        "type": "INLINE",
        "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": text}},
    }


def rerank_response(*results: tuple[int, float]) -> dict:
    """A response shaped like the Rerank API's, from (index, relevanceScore) pairs."""
    return {"results": [{"index": index, "relevanceScore": score} for index, score in results]}


class FakeAgentRuntime:
    """The one method the re-ranker calls, plus a record of how it was called."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def rerank(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        return self.response


def test_the_request_sends_the_question_every_passage_and_the_model_arn():
    client = FakeAgentRuntime(rerank_response((2, 0.8803), (0, 0.0325), (1, 0.0216)))
    reranker = BedrockReranker(model=MODEL, region="us-west-2", client=client)

    reranker.rerank(QUESTION, DOCUMENTS, top_n=3)

    (call,) = client.calls
    assert call == {
        "queries": [{"type": "TEXT", "textQuery": {"text": QUESTION}}],
        "sources": [source(document) for document in DOCUMENTS],
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": MODEL_ARN},
                "numberOfResults": 3,
            },
        },
    }


def test_the_model_arn_is_built_from_the_region_and_the_model_id():
    reranker = BedrockReranker(model=MODEL, region="us-east-1")

    assert reranker.model_arn == "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"


def test_each_result_is_a_position_in_the_passages_given_and_its_score():
    client = FakeAgentRuntime(rerank_response((2, 0.8803), (0, 0.0325), (1, 0.0216)))
    reranker = BedrockReranker(model=MODEL, region="us-west-2", client=client)

    results = reranker.rerank(QUESTION, DOCUMENTS, top_n=3)

    assert results == [
        RerankResult(index=2, score=0.8803),
        RerankResult(index=0, score=0.0325),
        RerankResult(index=1, score=0.0216),
    ]


@pytest.mark.parametrize(("top_n", "expected"), [(1, 1), (3, 3), (5, 3)])
def test_number_of_results_is_top_n_but_never_more_than_the_passages_sent(top_n, expected):
    # Bedrock refuses the whole call otherwise: "Cannot provide numberOfResults value that
    # exceeds the number of sources."
    client = FakeAgentRuntime(rerank_response((2, 0.8803)))
    reranker = BedrockReranker(model=MODEL, region="us-west-2", client=client)

    reranker.rerank(QUESTION, DOCUMENTS, top_n=top_n)

    (call,) = client.calls
    configuration = call["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert configuration["numberOfResults"] == expected


@pytest.mark.parametrize(
    ("documents", "top_n"), [([], 5), (DOCUMENTS, 0)], ids=["no passages", "no results asked for"]
)
def test_nothing_to_rank_means_no_call_and_no_results(documents, top_n):
    client = FakeAgentRuntime(rerank_response())
    reranker = BedrockReranker(model=MODEL, region="us-west-2", client=client)

    assert reranker.rerank(QUESTION, documents, top_n=top_n) == []
    assert client.calls == []


def test_the_results_are_best_first_and_at_most_top_n_whatever_order_they_arrive_in():
    # Bedrock sends its results best first and no more than were asked for; the promise the
    # Reranker protocol makes should hold even if it one day does not.
    client = FakeAgentRuntime(rerank_response((0, 0.0325), (2, 0.8803), (1, 0.0216)))
    reranker = BedrockReranker(model=MODEL, region="us-west-2", client=client)

    results = reranker.rerank(QUESTION, DOCUMENTS, top_n=2)

    assert results == [RerankResult(index=2, score=0.8803), RerankResult(index=0, score=0.0325)]


@pytest.mark.parametrize(
    ("response", "complaint"),
    [
        ({"ResponseMetadata": {"HTTPStatusCode": 200}}, "results"),
        ({"results": "no ranking here"}, "results"),
        ({"results": [{"index": 0}]}, "relevanceScore"),
        ({"results": [{"relevanceScore": 0.5}]}, "index"),
        ({"results": [{"index": "2", "relevanceScore": 0.5}]}, "index"),
        ({"results": [{"index": 2, "relevanceScore": "high"}]}, "relevanceScore"),
        ({"results": [{"index": 3, "relevanceScore": 0.5}]}, "index 3"),
        ({"results": [{"index": -1, "relevanceScore": 0.5}]}, "index -1"),
        (rerank_response((2, 0.8803), (2, 0.0325)), "twice"),
    ],
    ids=[
        "no results",
        "results not a list",
        "no score",
        "no index",
        "index not a number",
        "score not a number",
        "index past the end",
        "negative index",
        "same index twice",
    ],
)
def test_a_malformed_response_says_so_clearly(response, complaint):
    reranker = BedrockReranker(model=MODEL, region="us-west-2", client=FakeAgentRuntime(response))

    with pytest.raises(RerankResponseError) as raised:
        reranker.rerank(QUESTION, DOCUMENTS, top_n=3)

    assert MODEL in str(raised.value)
    assert complaint in str(raised.value)


def test_the_request_is_valid_for_the_real_bedrock_agent_runtime_api():
    # botocore validates the parameters against the service model, so a misspelt key or a
    # missing "type" fails here rather than in production.
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="us-west-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(client) as stubber:
        stubber.add_response(
            "rerank",
            rerank_response((1, 0.8803), (0, 0.0325)),
            {
                "queries": [{"type": "TEXT", "textQuery": {"text": QUESTION}}],
                "sources": [source(document) for document in DOCUMENTS[1:]],
                "rerankingConfiguration": {
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {"modelArn": MODEL_ARN},
                        "numberOfResults": 2,
                    },
                },
            },
        )
        reranker = BedrockReranker(model=MODEL, region="us-west-2", client=client)

        assert reranker.rerank(QUESTION, DOCUMENTS[1:], top_n=5) == [
            RerankResult(index=1, score=0.8803),
            RerankResult(index=0, score=0.0325),
        ]
        stubber.assert_no_pending_responses()
