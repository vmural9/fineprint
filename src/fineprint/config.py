"""Every setting the service reads, in one place.

Values come from the process environment, and from a `.env` file in the working directory for
local development. The environment wins over the file, and both win over the defaults below.
`.env` is never committed; `.env.example` lists the variables with safe values.

Two AWS variables are deliberately missing from this class. `AWS_PROFILE` and the access keys
behind it are read by boto3 itself from the environment, so naming them here would only invite
someone to give a credential a default value. No setting in this file is a secret.
"""

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
