"""Every setting the service reads, in one place.

Values come from the process environment, and from a `.env` file in the working directory for
local development. The environment wins over the file, and both win over the defaults below.
`.env` is never committed; `.env.example` lists the variables with safe values.

Two AWS variables are deliberately missing from this class. `AWS_PROFILE` and the access keys
behind it are read by boto3 itself from the environment, so naming them here would only invite
someone to give a credential a default value. The one secret this class does carry, the
Langfuse secret key, has no default and is held as a `SecretStr`, so it never shows in a repr.
"""

from typing import Any, Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """The service's configuration, read once and passed around explicitly.

    Each field is filled from the upper-case form of its name, so `chunk_set` comes from
    `CHUNK_SET`.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Where the handbook, its pages, and its chunks live. Matches docker-compose.yml.
    database_url: str = "postgresql://fineprint:fineprint@localhost:5432/fineprint"

    # Which handbook edition queries run against. Part 4 loads a second edition beside it.
    corpus_edition: int = 2026
    # Which set of chunks queries run against. Part 2 adds more sets beside it.
    chunk_set: str = "fixed-220w"

    # The model that writes answers, reached through AWS Bedrock.
    llm_provider: str = "bedrock"
    llm_model: str = "us.anthropic.claude-opus-5"

    # The model that turns text into vectors. Its output dimension must equal the width of the
    # `chunks.embedding` column in schema.sql.
    embedding_provider: str = "bedrock"
    embedding_model: str = "amazon.titan-embed-text-v2:0"

    # An optional step between fusion and the top-k cut. "none" skips it. "bedrock" sends the
    # question and the best `rerank_candidates` fused chunks to `reranker_model`, which scores
    # each chunk against the question, and the top k are kept in that new order.
    reranker_provider: str = "none"
    reranker_model: str = "cohere.rerank-v3-5:0"
    rerank_candidates: int = 20

    # The judge that scores the four RAGAS metrics in the evals (`python -m evals.score`); the
    # service never calls it. It is not the model that writes the answers, so no answer is scored
    # by its own author.
    judge_model: str = "us.anthropic.claude-sonnet-5"

    # The region every Bedrock model is called in.
    aws_region: str = "us-west-2"

    # Retrieval sizes: how many rows each retriever returns, how many survive fusion, and the
    # constant in the reciprocal rank fusion formula.
    retrieval_candidates: int = 20
    retrieval_top_k: int = 5
    rrf_k: int = 60

    # Where the answer model's calls are timed and costed. "none" keeps the measurements in the
    # process; "langfuse" also sends every trace to Langfuse, which needs the three values below.
    tracing_provider: Literal["none", "langfuse"] = "none"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    # The region URL from the Langfuse project's settings page.
    langfuse_base_url: str | None = None

    # The model that answers when `llm_model` is unavailable. An empty string turns it off.
    fallback_llm_model: str = "us.anthropic.claude-sonnet-5"

    # How long one request to the answer or fallback model may take, and how many times the SDK
    # retries throttling and server errors before the service gives up on it.
    llm_timeout_seconds: float = 60
    llm_max_retries: int = 4

    # The same two limits for the embedder and the re-ranker: a read timeout in seconds, and
    # botocore's `max_attempts` (the first try included).
    aws_timeout_seconds: float = 10
    aws_max_attempts: int = 4

    # How many tool turns the agent may take before it is made to record its answer.
    agent_max_tool_turns: int = 4

    # A test and demo switch that makes one call fail on purpose, for example
    # `chat:unavailable:once` or `rerank:timeout`. Empty means no faults.
    fault_injection: str = ""

    @field_validator(
        "langfuse_public_key", "langfuse_secret_key", "langfuse_base_url", mode="before"
    )
    @classmethod
    def empty_means_unset(cls, value: Any) -> Any:
        """Read `LANGFUSE_PUBLIC_KEY=` in a `.env` file as "not set" rather than as an empty key."""
        return None if value == "" else value

    @model_validator(mode="after")
    def langfuse_needs_its_three_values(self) -> "Settings":
        """Refuse `TRACING_PROVIDER=langfuse` unless the keys and URL are there.

        The Langfuse client does not raise for a missing key: it logs a line and quietly
        discards every trace. Failing here, naming what is missing, is the louder choice.
        """
        if self.tracing_provider != "langfuse":
            return self
        values = {
            "LANGFUSE_PUBLIC_KEY": self.langfuse_public_key,
            "LANGFUSE_SECRET_KEY": self.langfuse_secret_key,
            "LANGFUSE_BASE_URL": self.langfuse_base_url,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                f"TRACING_PROVIDER=langfuse needs {', '.join(missing)} to be set; "
                "set them in the environment or in .env, or use TRACING_PROVIDER=none."
            )
        return self
