"""Tests for the scoreboard generator.

The fixture results files under `tests/fixtures/results/` are hand-written, so every number in
`tests/fixtures/expected_scoreboard.md` can be worked out on paper. For `hybrid`: three
answerable questions of which two found an expected page (page_hit 2/3 = 66.7%); expected pages
found 1 + 0.5 + 0 out of 3 (page_recall 50.0%); first hits at ranks 1 and 2 and nowhere
(mrr (1 + 0.5 + 0)/3 = 0.50); citations landing on an expected page once in three (33.3%); the
right call on found_in_handbook on all four questions (100.0%); and two answers reviewed of
which one passed (50.0%).

For `hybrid+rerank`: the same four questions, re-ranked. q001's only covering chunk sat at
`fused_rank` 9 before re-ranking put it at `rank` 1 — lifted into the top 5. q002's covering
chunk was already at `fused_rank` 2 — already in the top 5. q005's one stored chunk (pages
60-61) never covers its expected page 42 at all — no expected page in the top 5. That is one
question in each of the three movement buckets, out of three answerable questions. The four
RAGAS metrics: context_recall (1.0, 0.5, 0.0) means 0.50; context_precision (1.0, 0.75, 0.25)
means 0.67; faithfulness (1.0, 0.6, and `None` for q005, which had no claims to check) means
(1.0 + 0.6) / 2 = 0.80; answer_relevance (0.9, 0.7, 0.0) means 0.53.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from evals.golden import Evidence, GoldenItem
from evals.results import latest_per_config
from evals.scoreboard import (
    BLOCK_END,
    BLOCK_START,
    ScoreboardError,
    main,
    render,
    render_readme_block,
    replace_block,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "results"
STAMP = "2026-09-21 03:15 UTC"

# A small, self-contained stand-in for the committed golden set, covering only the four
# questions the fixtures use, so the re-ranker movement table's tests never depend on
# `evals/golden_set.jsonl` staying the way it is today. None of the movement scenarios below
# needs an `alt_pages` entry to resolve correctly, so none is given one.
GOLDEN = {
    item.id: item
    for item in (
        GoldenItem(
            id="q001",
            question="How much will her Part B premium be each month in 2026?",
            expected_answer="The standard Part B premium in 2026 is $202.90 a month.",
            expected_pages=(23,),
            category="costs",
            difficulty="easy",
            type="lookup",
            evidence=(
                Evidence(page=23, quote="The standard Part B premium amount in 2026 is $202.90."),
            ),
        ),
        GoldenItem(
            id="q002",
            question=(
                "What will 5 hospital days and 30 skilled nursing facility days cost in 2026?"
            ),
            expected_answer="You pay the $1,736 Part A deductible for the benefit period.",
            expected_pages=(27, 29),
            category="costs",
            difficulty="hard",
            type="multi_section",
            evidence=(
                Evidence(
                    page=27,
                    quote=(
                        "Each time you start a new benefit period, you must pay $1,736 (in "
                        "2026) before Medicare starts to pay."
                    ),
                ),
            ),
        ),
        GoldenItem(
            id="q003",
            question="What are the 2026 income and resource limits for Extra Help?",
            expected_answer=(
                "The handbook does not give the 2026 limits; it points to Medicare.gov."
            ),
            expected_pages=(),
            category="costs",
            difficulty="medium",
            type="unanswerable",
            evidence=(
                Evidence(
                    page=92, quote="You can find 2026 income and resource limits on Medicare.gov."
                ),
            ),
        ),
        GoldenItem(
            id="q005",
            question="What does the comparison table say about seeing a specialist?",
            expected_answer="Medicare Advantage Plans may require a referral to see a specialist.",
            expected_pages=(42,),
            category="original_vs_advantage",
            difficulty="medium",
            type="table",
            evidence=(Evidence(page=42, quote="you may need a referral"),),
        ),
    )
}

README = f"""# fineprint

Some prose above the block.

{BLOCK_START}
No eval has been run yet.
{BLOCK_END}

Some prose below the block.
"""

EXPECTED_SCOREBOARD = (FIXTURES.parent / "expected_scoreboard.md").read_text(encoding="utf-8")
# The block has no trailing newline of its own; the fixture file, like every text file, does.
EXPECTED_README_BLOCK = (
    (FIXTURES.parent / "expected_readme_block.md").read_text(encoding="utf-8").rstrip("\n")
)


@pytest.fixture
def fixture_results():
    """The latest results file per configuration, from the committed fixtures."""
    return latest_per_config(FIXTURES)


@pytest.fixture
def workspace(tmp_path: Path):
    """A scoreboard file and a README to generate into, away from the real ones."""
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    scoreboard = tmp_path / "scoreboard.md"
    return scoreboard, readme


def arguments(workspace, *extra: str) -> list[str]:
    """The command-line arguments that point the generator at the fixtures and the workspace."""
    scoreboard, readme = workspace
    return [
        "--results-dir",
        str(FIXTURES),
        "--scoreboard",
        str(scoreboard),
        "--readme",
        str(readme),
        *extra,
    ]


# --- the rendered files -----------------------------------------------------------


def test_the_fixture_results_render_to_exactly_this_markdown(fixture_results):
    assert render(fixture_results, STAMP, golden=GOLDEN) == EXPECTED_SCOREBOARD


def test_the_readme_block_carries_the_headline_table_and_points_at_the_detail(fixture_results):
    assert render_readme_block(fixture_results, STAMP) == EXPECTED_README_BLOCK


def test_only_the_latest_run_of_a_configuration_is_shown(fixture_results):
    # The superseded hybrid run scored 0.0% and reviewed one answer as a fail.
    assert fixture_results["hybrid"].run_id == "20260921T031500Z_hybrid"
    assert "0.0% (1 reviewed)" not in render(fixture_results, STAMP, golden=GOLDEN)


def test_with_no_results_the_table_has_no_rows_and_says_so():
    document = render({}, STAMP)

    assert "No eval has been run yet" in document
    assert "| Configuration | Page hit@5 |" in document
    assert "| hybrid" not in document
    assert "## Detail" not in document


def test_a_row_whose_run_had_failures_says_so_under_the_table(fixture_results):
    fixture_results["hybrid"].questions[0].error = "answering failed: Bedrock said no"

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "`hybrid`: 1 question failed during the run and scored nothing." in document


def test_rows_measured_against_different_corpora_are_flagged_as_not_comparable(fixture_results):
    lexical = fixture_results["lexical-only"]
    lexical.golden_set = replace(lexical.golden_set, sha256="ff" * 32)

    assert "not strictly comparable" in render(fixture_results, STAMP, golden=GOLDEN)


# --- part 2: the RAGAS columns, the by-type tables, and the re-ranker ------------


def test_a_row_with_no_scoring_pass_shows_dashes_for_the_four_ragas_headline_columns(
    fixture_results,
):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| hybrid | 66.7% | 50.0% (2 reviewed) | — | — | — | — | — | — | — | — |" in document
    assert "| lexical-only | 66.7% | — | — | — | — | — | — | — | — | — |" in document


def test_a_scored_row_shows_the_four_ragas_headline_columns_to_two_decimals(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| hybrid+rerank | 66.7% | — | 0.50 | 0.67 | 0.80 | 0.53 | — | — | — | — |" in document


def test_page_recall_context_recall_and_faithfulness_get_their_own_by_type_tables(
    fixture_results,
):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "### page_recall@5 by question type" in document
    assert "### context_recall by question type" in document
    assert "### faithfulness by question type" in document


def test_by_type_tables_are_all_dashes_for_a_row_with_no_scoring_pass(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    context_recall_section = document.split("### context_recall by question type")[1]
    assert "| hybrid | — | — | — |" in context_recall_section
    assert "| lexical-only | — | — | — |" in context_recall_section


def test_faithfulness_by_type_is_a_dash_for_a_type_with_no_claims_to_check(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    faithfulness_section = document.split("### faithfulness by question type")[1]
    # q005 is hybrid+rerank's only `table` question, and its faithfulness is None (no claims).
    assert "| hybrid+rerank | 1.00 (1) | — | 0.60 (1) |" in faithfulness_section


def test_the_reranker_movement_table_lists_only_configurations_with_a_reranker(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    movement_section = document.split("### Re-ranker movement")[1].split(
        "### What produced each row"
    )[0]
    assert "| hybrid+rerank |" in movement_section
    assert "| hybrid |" not in movement_section
    assert "lexical-only" not in movement_section


def test_the_movement_counts_match_the_fixtures_fused_ranks(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    # q001 lifted (fused_rank 9 > 5), q002 already there (fused_rank 2), q005 has no covering
    # chunk among its stored top 5 at all; q003 is unanswerable and does not count.
    assert "| hybrid+rerank | 1 | 1 | 1 | 3 |" in document


def test_the_movement_table_is_left_out_when_no_row_has_a_reranker():
    unreranked = {"lexical-only": latest_per_config(FIXTURES)["lexical-only"]}

    document = render(unreranked, STAMP)

    assert "Re-ranker movement" not in document


def test_the_provenance_table_shows_the_reranker_model_and_a_dash_when_there_is_none(
    fixture_results,
):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| hybrid | hybrid | fixed-220w | 5 | 20 | 60 | — | " in document
    assert (
        "| hybrid+rerank | hybrid | fixed-220w | 5 | 20 | 60 | cohere.rerank-v3-5:0 | " in document
    )


def test_the_provenance_table_shows_the_judge_model_and_a_dash_when_unscored(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert (
        "us.anthropic.claude-opus-5 | — | 1f4e9c2 | 20260921T031500Z_hybrid" in document
    )  # hybrid: answered but never scored
    assert (
        "us.anthropic.claude-sonnet-5 | 1f4e9c2 | 20260921T050000Z_hybrid+rerank" in document
    )  # hybrid+rerank: scored by Sonnet 5


def test_an_unknown_configuration_sorts_alphabetically_after_the_known_ones(fixture_results):
    extra = replace(fixture_results["lexical-only"])
    extra.config = replace(extra.config, name="zzz-experimental")
    fixture_results["zzz-experimental"] = extra

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert (
        document.index("| hybrid |")
        < document.index("| hybrid+rerank |")
        < document.index("| lexical-only |")
        < document.index("| zzz-experimental |")
    )


# --- the README block -------------------------------------------------------------


def test_replacing_the_block_leaves_the_rest_of_the_readme_alone():
    replaced = replace_block(README, f"{BLOCK_START}\nnew\n{BLOCK_END}")

    assert "Some prose above the block." in replaced
    assert "Some prose below the block." in replaced
    assert "No eval has been run yet." not in replaced
    assert f"{BLOCK_START}\nnew\n{BLOCK_END}" in replaced


def test_a_readme_without_the_markers_is_refused():
    with pytest.raises(ScoreboardError) as error:
        replace_block("# fineprint\n", "block")

    assert "scoreboard:start" in str(error.value)


# --- the command ------------------------------------------------------------------


def test_the_command_writes_both_files(workspace, capsys):
    scoreboard, readme = workspace

    exit_code = main(arguments(workspace))

    assert exit_code == 0
    assert scoreboard.read_text(encoding="utf-8").startswith("# Scoreboard")
    assert "| hybrid | 66.7% |" in readme.read_text(encoding="utf-8")
    assert "Some prose below the block." in readme.read_text(encoding="utf-8")
    assert "hybrid, hybrid+rerank, lexical-only" in capsys.readouterr().out


def test_check_passes_right_after_generating(workspace):
    main(arguments(workspace))

    assert main(arguments(workspace, "--check")) == 0


def test_check_ignores_the_generation_time(workspace):
    scoreboard, _ = workspace
    main(arguments(workspace))
    aged = scoreboard.read_text(encoding="utf-8").replace(
        "Generated at", "Generated at an hour ago,", 1
    )
    scoreboard.write_text(aged, encoding="utf-8")

    assert main(arguments(workspace, "--check")) == 0


def test_check_fails_after_a_number_is_edited_by_hand(workspace, capsys):
    scoreboard, _ = workspace
    main(arguments(workspace))
    edited = scoreboard.read_text(encoding="utf-8").replace("66.7%", "99.9%")
    scoreboard.write_text(edited, encoding="utf-8")

    exit_code = main(arguments(workspace, "--check"))

    assert exit_code == 1
    assert "is out of date" in capsys.readouterr().out


def test_check_fails_when_the_readme_block_is_edited_by_hand(workspace):
    _, readme = workspace
    main(arguments(workspace))
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("66.7%", "99.9%"), encoding="utf-8"
    )

    assert main(arguments(workspace, "--check")) == 1


def test_check_fails_when_the_scoreboard_has_never_been_generated(workspace):
    assert main(arguments(workspace, "--check")) == 1


def test_a_missing_readme_is_an_error_not_a_stale_file(tmp_path, capsys):
    exit_code = main(
        [
            "--results-dir",
            str(FIXTURES),
            "--scoreboard",
            str(tmp_path / "scoreboard.md"),
            "--readme",
            str(tmp_path / "nowhere.md"),
        ]
    )

    assert exit_code == 2
    assert "does not exist" in capsys.readouterr().err
