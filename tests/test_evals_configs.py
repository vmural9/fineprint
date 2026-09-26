"""Tests for the six named configurations and how each becomes a `Settings`."""

import subprocess
import sys

import pytest

from evals.configs import CONFIGS, RERANKER_MODEL, EvalConfig, settings_for
from fineprint.config import Settings

# (chunk_set, mode, reranker) for every row the part 2 spec asks for.
EXPECTED = {
    "hybrid": ("fixed-220w", "hybrid", None),
    "vector-only": ("fixed-220w", "vector", None),
    "lexical-only": ("fixed-220w", "lexical", None),
    "hybrid+rerank": ("fixed-220w", "hybrid", RERANKER_MODEL),
    "sections": ("sections", "hybrid", None),
    "sections+rerank": ("sections", "hybrid", RERANKER_MODEL),
}


def test_the_registry_has_exactly_the_six_scoreboard_configurations():
    assert set(CONFIGS) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_configuration_matches_its_row_in_the_part_2_spec(name):
    chunk_set, mode, reranker = EXPECTED[name]

    config = CONFIGS[name]

    assert (config.name, config.chunk_set, config.mode, config.reranker) == (
        name,
        chunk_set,
        mode,
        reranker,
    )


def test_importing_this_module_does_not_load_the_retriever():
    # `mode`'s type is `fineprint.retrieval.SearchMode`, imported only under TYPE_CHECKING: this
    # module is the one every eval script imports for --config, so it must not need Postgres,
    # boto3 or anthropic just to describe the command line.
    program = "import sys; import evals.configs; print('fineprint.retrieval' in sys.modules)"
    finished = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )

    assert finished.stdout.strip() == "False"


def test_part_1s_three_names_are_kept_so_their_files_still_key_to_their_rows():
    # D6: renaming these would orphan the committed hybrid/vector-only/lexical-only results.
    assert {"hybrid", "vector-only", "lexical-only"} <= set(CONFIGS)


def test_eval_config_is_frozen():
    config = CONFIGS["hybrid"]

    with pytest.raises(AttributeError):
        config.name = "renamed"  # type: ignore[misc]


def test_a_config_can_be_built_directly_without_the_registry():
    # The dataclass itself is part of the interface, not just the six registered rows.
    config = EvalConfig(name="custom", chunk_set="fixed-220w", mode="vector", reranker=None)

    assert config.reranker is None


# --- settings_for -------------------------------------------------------------------


def test_settings_for_sets_the_chunk_set_from_the_configuration():
    settings = settings_for(CONFIGS["sections"], Settings())

    assert settings.chunk_set == "sections"


def test_settings_for_turns_off_the_reranker_when_the_configuration_has_none():
    settings = settings_for(CONFIGS["hybrid"], Settings())

    assert settings.reranker_provider == "none"


def test_settings_for_turns_on_the_bedrock_reranker_with_the_configurations_model():
    settings = settings_for(CONFIGS["hybrid+rerank"], Settings())

    assert settings.reranker_provider == "bedrock"
    assert settings.reranker_model == RERANKER_MODEL


def test_settings_for_turns_on_the_reranker_for_sections_plus_rerank_too():
    settings = settings_for(CONFIGS["sections+rerank"], Settings())

    assert settings.chunk_set == "sections"
    assert settings.reranker_provider == "bedrock"
    assert settings.reranker_model == RERANKER_MODEL


def test_settings_for_leaves_everything_else_from_the_base_settings_alone():
    base = Settings(rerank_candidates=40, retrieval_top_k=8)

    settings = settings_for(CONFIGS["sections+rerank"], base)

    assert settings.rerank_candidates == 40
    assert settings.retrieval_top_k == 8
    assert settings.database_url == base.database_url
    assert settings.embedding_model == base.embedding_model


def test_settings_for_does_not_mutate_the_base_settings_object():
    base = Settings()

    settings_for(CONFIGS["hybrid+rerank"], base)

    assert base.chunk_set == "fixed-220w"
    assert base.reranker_provider == "none"
