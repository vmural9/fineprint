"""The named configurations part 2 measures: which chunk set, which retrieval mode, which
re-ranker.

A results file records a `RunConfig` — what a run actually did. `EvalConfig` is what tells
`run_golden_set` what to do before it does it: which of the two chunk sets to search, which
retrieval mode to run in, and whether Cohere Rerank 3.5 re-orders the fused list before the
top-k cut. The six entries in `CONFIGS` are the six scoreboard rows part 2 adds: the three part 1
kept unchanged, plus a chunk set x re-ranker 2x2 run in hybrid mode.

Part 1's three names — `hybrid`, `vector-only`, `lexical-only` — are kept exactly so their
committed results files still key to the same rows (decision D6).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fineprint.config import Settings

if TYPE_CHECKING:
    # Only for the type checker: `evals.run_golden_set` and `evals.candidates` both import this
    # module at their own top level, and each stays free of `fineprint.retrieval` — and everything
    # that pulls in — until a real retriever is actually built, so this cannot be a real import.
    from fineprint.retrieval import SearchMode

# Verified live 2026-09-26 (decision D3): the Bedrock Rerank API model every re-ranking
# configuration below uses. It is also `Settings.reranker_model`'s own default, so a plain
# `Settings()` and a re-ranking `EvalConfig` never name a different model by accident.
RERANKER_MODEL = "cohere.rerank-v3-5:0"


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """One named point in the chunk set x retrieval mode x re-ranker space.

    `settings_for` is the only thing that turns one of these into calls: nothing here touches
    a database or a model on its own.
    """

    name: str
    chunk_set: str
    mode: "SearchMode"
    reranker: str | None  # a Bedrock re-ranker model id, or None for no re-ranking step


CONFIGS: dict[str, EvalConfig] = {
    config.name: config
    for config in (
        EvalConfig(name="hybrid", chunk_set="fixed-220w", mode="hybrid", reranker=None),
        EvalConfig(name="vector-only", chunk_set="fixed-220w", mode="vector", reranker=None),
        EvalConfig(name="lexical-only", chunk_set="fixed-220w", mode="lexical", reranker=None),
        EvalConfig(
            name="hybrid+rerank", chunk_set="fixed-220w", mode="hybrid", reranker=RERANKER_MODEL
        ),
        EvalConfig(name="sections", chunk_set="sections", mode="hybrid", reranker=None),
        EvalConfig(
            name="sections+rerank", chunk_set="sections", mode="hybrid", reranker=RERANKER_MODEL
        ),
    )
}


def settings_for(config: EvalConfig, base: Settings) -> Settings:
    """The `Settings` one run of `config` makes its calls with.

    `base` is ordinarily a bare `Settings()`, read fresh so a run picks up whatever the
    environment and `.env` say; a test passes in a `Settings` it built itself to hold everything
    else fixed. Only the fields a configuration controls are overridden — the chunk set to
    search, and whether a re-ranker runs and which model it calls. Everything else, including
    `rerank_candidates`, stays whatever `base` already had.
    """
    update: dict[str, str] = {
        "chunk_set": config.chunk_set,
        "reranker_provider": "bedrock" if config.reranker else "none",
    }
    if config.reranker:
        update["reranker_model"] = config.reranker
    return base.model_copy(update=update)
