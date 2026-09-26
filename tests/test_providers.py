"""The provider factory, the fakes other tests build on, and what gets imported when."""

import subprocess
import sys

import pytest
from pydantic import BaseModel

from fineprint.config import Settings
from fineprint.providers.base import ChatModel, Embedder, Reranker, RerankResult
from fineprint.providers.bedrock_chat import BedrockChatModel
from fineprint.providers.bedrock_embedder import BedrockEmbedder
from fineprint.providers.bedrock_reranker import BedrockReranker
from fineprint.providers.factory import (
    UnknownProviderError,
    get_chat_model,
    get_embedder,
    get_reranker,
)
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker


class Answer(BaseModel):
    text: str


def test_get_embedder_builds_the_bedrock_embedder_from_settings():
    settings = Settings(
        embedding_provider="bedrock", embedding_model="amazon.titan-embed-text-v2:0"
    )

    embedder = get_embedder(settings)

    assert isinstance(embedder, BedrockEmbedder)
    assert embedder.model == "amazon.titan-embed-text-v2:0"


def test_get_chat_model_builds_the_bedrock_chat_model_from_settings():
    settings = Settings(llm_provider="bedrock", llm_model="us.anthropic.claude-opus-5")

    chat = get_chat_model(settings)

    assert isinstance(chat, BedrockChatModel)
    assert chat.model == "us.anthropic.claude-opus-5"


def test_an_unknown_embedding_provider_names_the_ones_that_exist():
    settings = Settings(embedding_provider="openai")

    with pytest.raises(UnknownProviderError) as raised:
        get_embedder(settings)

    message = str(raised.value)
    assert "openai" in message
    assert "EMBEDDING_PROVIDER" in message
    assert "bedrock" in message


def test_an_unknown_chat_provider_names_the_ones_that_exist():
    settings = Settings(llm_provider="ollama")

    with pytest.raises(UnknownProviderError) as raised:
        get_chat_model(settings)

    message = str(raised.value)
    assert "ollama" in message
    assert "LLM_PROVIDER" in message


def test_get_reranker_returns_none_when_re_ranking_is_off():
    # "none" is the default: retrieval then keeps the fused order, as it did before re-ranking.
    assert get_reranker(Settings(reranker_provider="none")) is None


def test_get_reranker_builds_the_bedrock_reranker_from_settings():
    settings = Settings(
        reranker_provider="bedrock", reranker_model="cohere.rerank-v3-5:0", aws_region="us-west-2"
    )

    reranker = get_reranker(settings)

    assert isinstance(reranker, BedrockReranker)
    assert reranker.model == "cohere.rerank-v3-5:0"
    assert reranker.region == "us-west-2"


def test_an_unknown_reranker_provider_names_the_ones_that_exist():
    settings = Settings(reranker_provider="cohere")

    with pytest.raises(UnknownProviderError) as raised:
        get_reranker(settings)

    message = str(raised.value)
    assert "cohere" in message
    assert "RERANKER_PROVIDER" in message
    assert "bedrock" in message
    assert "none" in message


def test_importing_the_factory_does_not_import_a_provider_sdk():
    # Only a provider's own module may import its SDK, so importing the factory (which the API,
    # the CLI and the eval runner all do) stays fast and works with no AWS libraries loaded.
    program = (
        "import sys; import fineprint.providers.factory; "
        "print('boto3' in sys.modules, 'anthropic' in sys.modules)"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )

    assert finished.stdout.strip() == "False False"


def test_the_fakes_satisfy_the_protocols():
    # A structural check: these objects are what every other test uses in place of Bedrock.
    embedder: Embedder = FakeEmbedder()
    chat: ChatModel = FakeChatModel(Answer(text="hello"))
    reranker: Reranker = FakeReranker()

    assert embedder.dimension == 1024
    assert len(embedder.embed_query("Part B premium")) == embedder.dimension
    assert chat.complete_structured("system", "user", Answer).parsed.text == "hello"
    assert reranker.model == "fake-reranker"
    assert reranker.rerank("premium", ["deductible", "premium"], top_n=1) == [
        RerankResult(index=1, score=1.0)
    ]


def test_the_fake_embedder_gives_the_same_vector_for_the_same_text_twice():
    # This is what keeps ingestion and retrieval repeatable without calling Bedrock.
    embedder = FakeEmbedder()

    assert embedder.embed_query("Part B premium") == embedder.embed_query("Part B premium")
    assert embedder.embed_query("Part B premium") != embedder.embed_query("Part A premium")
    assert embedder.embed_documents(["one", "two"]) == [
        embedder.embed_query("one"),
        embedder.embed_query("two"),
    ]
    assert len(embedder.embed_query("Part B premium")) == embedder.dimension


def test_the_fake_chat_model_returns_the_canned_object_and_records_the_prompt():
    chat = FakeChatModel(Answer(text="the standard premium is $202.90"))

    result = chat.complete_structured("system rules", "the question", Answer)

    assert result.parsed.text == "the standard premium is $202.90"
    assert result.model == "fake-chat-model"
    assert chat.calls == [("system rules", "the question", Answer)]


def test_the_fake_chat_model_refuses_a_schema_it_was_not_given():
    chat = FakeChatModel(Answer(text="hello"))

    class Other(BaseModel):
        value: int

    with pytest.raises(TypeError):
        chat.complete_structured("system", "user", Other)


def test_the_fake_reranker_ranks_by_how_many_query_words_each_passage_contains():
    reranker = FakeReranker()
    passages = [
        "The Part A premium",  # part, premium
        "part b PREMIUM",  # part, b, premium: case does not matter
        "Partial premiums",  # none: "partial" is not "part", nor is "premiums" "premium"
        "Part B",  # part, b: ties with the first passage and loses on position
    ]

    results = reranker.rerank("Part B premium", passages, top_n=3)

    assert results == [
        RerankResult(index=1, score=3.0),
        RerankResult(index=0, score=2.0),
        RerankResult(index=3, score=2.0),
    ]
    assert reranker.calls == [("Part B premium", passages, 3)]


def test_the_fake_reranker_returns_every_passage_when_top_n_is_larger():
    reranker = FakeReranker()

    results = reranker.rerank("Part B", ["Part A", "Part B"], top_n=5)

    assert [result.index for result in results] == [1, 0]
