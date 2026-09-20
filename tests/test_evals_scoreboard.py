"""Tests for the scoreboard generator.

The fixture results files under `tests/fixtures/results/` are hand-written, so every number in
`tests/fixtures/expected_scoreboard.md` can be worked out on paper. For `hybrid`: three
answerable questions of which two found an expected page (page_hit 2/3 = 66.7%); expected pages
found 1 + 0.5 + 0 out of 3 (page_recall 50.0%); first hits at ranks 1 and 2 and nowhere
(mrr (1 + 0.5 + 0)/3 = 0.50); citations landing on an expected page once in three (33.3%); the
right call on found_in_handbook on all four questions (100.0%); and two answers reviewed of
which one passed (50.0%).
"""

from dataclasses import replace
from pathlib import Path

import pytest

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
    assert render(fixture_results, STAMP) == EXPECTED_SCOREBOARD


def test_the_readme_block_carries_the_headline_table_and_points_at_the_detail(fixture_results):
    assert render_readme_block(fixture_results, STAMP) == EXPECTED_README_BLOCK


def test_only_the_latest_run_of_a_configuration_is_shown(fixture_results):
    # The superseded hybrid run scored 0.0% and reviewed one answer as a fail.
    assert fixture_results["hybrid"].run_id == "20260921T031500Z_hybrid"
    assert "0.0% (1 reviewed)" not in render(fixture_results, STAMP)


def test_with_no_results_the_table_has_no_rows_and_says_so():
    document = render({}, STAMP)

    assert "No eval has been run yet" in document
    assert "| Configuration | Page hit@5 |" in document
    assert "| hybrid" not in document
    assert "Part 1 detail" not in document


def test_a_row_whose_run_had_failures_says_so_under_the_table(fixture_results):
    fixture_results["hybrid"].questions[0].error = "answering failed: Bedrock said no"

    document = render(fixture_results, STAMP)

    assert "`hybrid`: 1 question failed during the run and scored nothing." in document


def test_rows_measured_against_different_corpora_are_flagged_as_not_comparable(fixture_results):
    lexical = fixture_results["lexical-only"]
    lexical.golden_set = replace(lexical.golden_set, sha256="ff" * 32)

    assert "not strictly comparable" in render(fixture_results, STAMP)


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
    assert "hybrid, lexical-only" in capsys.readouterr().out


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
