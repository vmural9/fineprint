"""Tests for the settings object: defaults, the environment, and `.env`."""

import pytest
from pydantic import ValidationError

from fineprint.config import Settings

# Every variable `Settings` reads. The fixture below clears them so a developer's own shell
# cannot make the defaults look right (or wrong) by accident.
SETTING_VARIABLES = (
    "DATABASE_URL",
    "CORPUS_EDITION",
    "CHUNK_SET",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "RERANKER_PROVIDER",
    "RERANKER_MODEL",
    "RERANK_CANDIDATES",
    "JUDGE_MODEL",
    "AWS_REGION",
    "RETRIEVAL_CANDIDATES",
    "RETRIEVAL_TOP_K",
    "RRF_K",
)


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    """Run in an empty directory with none of the settings variables set."""
    for name in SETTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_defaults_match_the_documented_configuration(clean_environment):
    settings = Settings()
    assert settings.database_url == "postgresql://fineprint:fineprint@localhost:5432/fineprint"
    assert settings.corpus_edition == 2026
    assert settings.chunk_set == "fixed-220w"
    assert settings.llm_provider == "bedrock"
    assert settings.llm_model == "us.anthropic.claude-opus-5"
    assert settings.embedding_provider == "bedrock"
    assert settings.embedding_model == "amazon.titan-embed-text-v2:0"
    assert settings.reranker_provider == "none"
    assert settings.reranker_model == "cohere.rerank-v3-5:0"
    assert settings.rerank_candidates == 20
    assert settings.judge_model == "us.anthropic.claude-sonnet-5"
    assert settings.aws_region == "us-west-2"
    assert settings.retrieval_candidates == 20
    assert settings.retrieval_top_k == 5
    assert settings.rrf_k == 60


def test_the_environment_wins_over_the_defaults(clean_environment, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://reader@db.example:6543/handbook")
    monkeypatch.setenv("CHUNK_SET", "semantic-v2")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")

    settings = Settings()

    assert settings.database_url == "postgresql://reader@db.example:6543/handbook"
    assert settings.chunk_set == "semantic-v2"
    assert settings.retrieval_top_k == 8, "a numeric setting arrives as text and is parsed"


def test_a_dot_env_file_is_read(clean_environment):
    (clean_environment / ".env").write_text("CHUNK_SET=from-dot-env\n", encoding="utf-8")

    assert Settings().chunk_set == "from-dot-env"


def test_the_environment_wins_over_the_dot_env_file(clean_environment, monkeypatch):
    (clean_environment / ".env").write_text("CHUNK_SET=from-dot-env\n", encoding="utf-8")
    monkeypatch.setenv("CHUNK_SET", "from-the-environment")

    assert Settings().chunk_set == "from-the-environment"


def test_a_setting_that_is_not_a_number_is_rejected(clean_environment, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "several")

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert "retrieval_top_k" in str(excinfo.value)


def test_no_setting_is_a_secret(clean_environment):
    """Secrets come from the environment only and never have a default (project rule 3)."""
    secret_words = ("key", "secret", "token", "password", "credential")
    secrets = [name for name in Settings.model_fields if any(word in name for word in secret_words)]

    assert secrets == [], f"these settings look like secrets: {secrets}"
