"""Tests for the settings object: defaults, the environment, and `.env`."""

import pytest
from pydantic import SecretStr, ValidationError

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
    "TRACING_PROVIDER",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "FALLBACK_LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "AWS_TIMEOUT_SECONDS",
    "AWS_MAX_ATTEMPTS",
    "AGENT_MAX_TOOL_TURNS",
    "FAULT_INJECTION",
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


def test_the_part_3_defaults(clean_environment):
    settings = Settings()
    assert settings.tracing_provider == "none"
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_base_url is None
    assert settings.fallback_llm_model == "us.anthropic.claude-sonnet-5"
    assert settings.llm_timeout_seconds == 60
    assert settings.llm_max_retries == 4
    assert settings.aws_timeout_seconds == 10
    assert settings.aws_max_attempts == 4
    assert settings.agent_max_tool_turns == 4
    assert settings.fault_injection == ""


def test_an_empty_fallback_model_stays_empty_so_fallback_can_be_switched_off(
    clean_environment, monkeypatch
):
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "")

    assert Settings().fallback_llm_model == ""


def test_langfuse_tracing_is_refused_without_all_three_values(clean_environment, monkeypatch):
    monkeypatch.setenv("TRACING_PROVIDER", "langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    message = str(excinfo.value)
    assert "LANGFUSE_SECRET_KEY" in message
    assert "LANGFUSE_BASE_URL" in message
    assert "LANGFUSE_PUBLIC_KEY" not in message, "only the missing values are named"


def test_langfuse_values_left_empty_in_dot_env_count_as_missing(clean_environment):
    (clean_environment / ".env").write_text(
        "TRACING_PROVIDER=langfuse\nLANGFUSE_PUBLIC_KEY=\nLANGFUSE_SECRET_KEY=\n"
        "LANGFUSE_BASE_URL=\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        assert name in str(excinfo.value)


def test_langfuse_tracing_is_accepted_with_all_three_values(clean_environment, monkeypatch):
    monkeypatch.setenv("TRACING_PROVIDER", "langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    settings = Settings()

    assert settings.tracing_provider == "langfuse"
    assert isinstance(settings.langfuse_secret_key, SecretStr)
    assert settings.langfuse_secret_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings), "the secret key never shows in a repr"


def test_the_langfuse_values_are_not_needed_while_tracing_is_off(clean_environment):
    assert Settings().tracing_provider == "none"


def test_an_unknown_tracing_provider_is_rejected(clean_environment, monkeypatch):
    monkeypatch.setenv("TRACING_PROVIDER", "datadog")

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert "tracing_provider" in str(excinfo.value)


def test_no_setting_is_a_secret_with_a_default(clean_environment):
    """Secrets come from the environment only and never have a default (project rule 3)."""
    secret_words = ("key", "secret", "token", "password", "credential")
    secrets = [name for name in Settings.model_fields if any(word in name for word in secret_words)]

    assert sorted(secrets) == ["langfuse_public_key", "langfuse_secret_key"]
    for name in secrets:
        assert Settings.model_fields[name].default is None, f"{name} must not have a default"
