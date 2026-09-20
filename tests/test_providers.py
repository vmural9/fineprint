"""The provider factory, the fakes other tests build on, and what gets imported when."""

import subprocess
import sys

import pytest
from pydantic import BaseModel

from fineprint.config import Settings
from fineprint.providers.base import ChatModel, Embedder
from fineprint.providers.bedrock_chat import BedrockChatModel
from fineprint.providers.bedrock_embedder import BedrockEmbedder
from fineprint.providers.factory import UnknownProviderError, get_chat_model, get_embedder
from tests.fakes import FakeChatModel, FakeEmbedder


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
    # A structural check: these two objects are what every other test uses in place of Bedrock.
    embedder: Embedder = FakeEmbedder()
    chat: ChatModel = FakeChatModel(Answer(text="hello"))

    assert embedder.dimension == 1024
    assert len(embedder.embed_query("Part B premium")) == embedder.dimension
    assert chat.complete_structured("system", "user", Answer).parsed.text == "hello"


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
