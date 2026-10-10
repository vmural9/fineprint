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
        "us.anthropic.claude-opus-5 | — | — | — | 1f4e9c2 | 20260921T031500Z_hybrid" in document
    )  # hybrid: answered but never scored
    assert (
        "us.anthropic.claude-opus-5 | — | us.anthropic.claude-sonnet-5 | — | 1f4e9c2 | "
        "20260921T050000Z_hybrid+rerank" in document
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


# --- part 3: cost, latency, tool calls, and what the traced run was made with ----
#
# The `agent` fixture is traced and offers the two tools. Its four questions, by hand:
#   q001  8 s,  $0.04: embedding, 2 generations, 1 tool call
#   q002 12 s,  $0.06: embedding, 3 generations (the primary timed out, so one ran on the
#                      fallback model), 2 tool calls, one of which the tool refused
#   q003  4 s,  $0.02: embedding, 1 generation (unanswerable; it still costs)
#   q005 20 s,  $0.08: embedding, 2 generations, 2 tool calls
# Mean cost (0.04 + 0.06 + 0.02 + 0.08) / 4 = $0.0500. Sorted latencies 4, 8, 12, 20 s: p50 is
# rank 2 (8.0 s), p95 is rank ceil(0.95 * 4) = 4 (20.0 s), max 20.0 s. Generations (2 + 3 + 1 + 2)
# / 4 = 2.00; tool calls (1 + 2 + 0 + 2) / 4 = 1.25; one question had a fallback generation.
# Tool calls: lookup_cost_figure in q001, q002 and q005 (3 questions) made 1 + 1 + 2 = 4 calls,
# none refused; check_enrollment_window in q002 only, 1 call, refused.


def test_the_traced_row_shows_its_cost_per_query_and_p95_latency_in_the_headline(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | 66.7% | — | — | — | — | — | $0.0500 | 20.0 s | — | — |" in document


def test_the_legend_says_where_cost_and_p95_come_from(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert (
        "Cost per query and p95 latency come from the trace stored with each question, at AWS "
        "list prices in us-west-2, with p95 taken by nearest rank over the run's questions."
    ) in document
    assert "AWS list prices" in render_readme_block(fixture_results, STAMP)


def test_the_agent_row_sits_right_after_the_reranked_rows_and_before_the_rest(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert (
        document.index("| hybrid+rerank |")
        < document.index("| agent |")
        < document.index("| lexical-only |")
    )


def test_the_cost_and_latency_table_lists_the_traced_row_only(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    section = document.split("### Cost and latency")[1].split("### Tool calls")[0]
    assert (
        "| Configuration | Mean cost | p50 latency | p95 latency | Max latency | "
        "Generations per query | Tool calls per query | Fallbacks | Re-ranks skipped |"
    ) in section
    assert "| agent | $0.0500 | 8.0 s | 20.0 s | 20.0 s | 2.00 | 1.25 | 1 | — |" in section
    assert "| hybrid |" not in section
    assert "lexical-only" not in section


def test_the_cost_and_latency_table_is_left_out_while_no_row_has_a_trace():
    untraced = {
        name: result for name, result in latest_per_config(FIXTURES).items() if name != "agent"
    }

    document = render(untraced, STAMP, golden=GOLDEN)

    assert "Cost and latency" not in document
    assert "$" not in document


def test_one_question_without_a_trace_turns_the_traced_rows_numbers_into_dashes(fixture_results):
    fixture_results["agent"].questions[2].trace = None

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | 66.7% | — | — | — | — | — | — | — | — | — |" in document
    # The row had traces for three questions, so it stays in the table, with dashes.
    assert "| agent | — | — | — | — | — | — | — |" in document


def test_one_trace_without_a_cost_turns_the_traced_rows_numbers_into_dashes(fixture_results):
    fixture_results["agent"].questions[1].trace = replace(
        fixture_results["agent"].questions[1].trace, cost_usd=None
    )

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | 66.7% | — | — | — | — | — | — | — | — | — |" in document
    assert "| agent | — | — | — | — | — | — | — |" in document


def test_the_tool_calls_table_counts_questions_calls_and_refused_calls_per_tool(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    section = document.split("### Tool calls")[1].split("### What produced each row")[0]
    assert (
        "| Configuration | Tool | Questions that called it | Which questions | Calls | Errors |"
        in section
    )
    assert "| agent | check_enrollment_window | 1 | q002 | 1 | 1 |" in section
    assert "| agent | lookup_cost_figure | 3 | q001, q002, q005 | 4 | 0 |" in section
    assert "| hybrid |" not in section


def test_the_tool_calls_table_lists_the_questions_that_called_neither_tool(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    section = document.split("### Tool calls")[1].split("### What produced each row")[0]
    assert "| agent | (neither tool) | 1 | q003 | — | — |" in section
    assert "a question without a trace is in no line" in section


def test_a_question_that_called_both_tools_is_listed_under_each_once(fixture_results):
    # q002 called lookup_cost_figure and check_enrollment_window; q005 called the lookup twice.
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | lookup_cost_figure | 3 | q001, q002, q005 |" in document
    assert "| agent | check_enrollment_window | 1 | q002 |" in document


def test_a_row_that_offered_the_tools_but_never_used_them_shows_zeros_not_nothing(
    fixture_results,
):
    for question in fixture_results["agent"].questions:
        question.trace = replace(
            question.trace,
            calls=[call for call in question.trace.calls if call.kind != "tool"],
        )

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | check_enrollment_window | 0 | — | 0 | 0 |" in document
    assert "| agent | lookup_cost_figure | 0 | — | 0 | 0 |" in document
    assert "| agent | (neither tool) | 4 | q001, q002, q003, q005 | — | — |" in document


def test_the_tool_calls_come_from_the_traces_so_a_failed_answer_does_not_change_them(
    fixture_results,
):
    # q001 has no stored answer but its trace still shows the lookup it made, so the table and
    # "tool calls per query" (1.25) agree. The answer's own tool_calls list is not read.
    fixture_results["agent"].questions[0].answer = None

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | lookup_cost_figure | 3 | q001, q002, q005 | 4 | 0 |" in document
    assert "| agent | $0.0500 | 8.0 s | 20.0 s | 20.0 s | 2.00 | 1.25 | 1 | — |" in document


def test_a_question_without_a_trace_adds_no_tool_calls(fixture_results):
    fixture_results["agent"].questions[0].trace = None

    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert "| agent | lookup_cost_figure | 2 | q002, q005 | 3 | 0 |" in document
    assert "| agent | (neither tool) | 1 | q003 | — | — |" in document, "q001 is in no line"


def test_the_tool_calls_table_is_left_out_while_no_row_offered_the_tools():
    untooled = {
        name: result
        for name, result in latest_per_config(FIXTURES).items()
        if not result.config.tools
    }

    assert "Tool calls" not in render(untooled, STAMP, golden=GOLDEN)


def test_the_provenance_table_names_the_fallback_model_and_the_tracing_provider(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert (
        "| agent | hybrid | sections | 5 | 20 | 60 | — | amazon.titan-embed-text-v2:0 | "
        "us.anthropic.claude-opus-5 | us.anthropic.claude-sonnet-5 | — | langfuse | 1f4e9c2 | "
        "20260921T060000Z_agent |"
    ) in document
    # An older run has neither: both cells are dashes.
    assert "| Answer model | Fallback model | Judge model | Tracing | Commit | Run |" in document


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
    assert "hybrid, hybrid+rerank, agent, lexical-only" in capsys.readouterr().out


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


# --- part 3: where the time went --------------------------------------------------
#
# Per question, from the agent fixture's calls (seconds):
#   q001  retriever 1.0; generations 3.0 + 2.5 = 5.5; tool 0.4;           of 8 s    -> 68.75%
#   q002  retriever 1.5; generations 2.0 + 3.0 + 3.5 = 8.5; tools 0.3 + 0.2 = 0.5; of 12 s -> 70.83%
#   q003  retriever 0.5; generation 3.0; no tool (0);                      of 4 s    -> 75%
#   q005  retriever 2.0; generations 6.0 + 9.0 = 15.0; tools 0.25 + 0.25 = 0.5; of 20 s -> 75%
# Nearest rank over four questions: p50 is the 2nd smallest, p95 the 4th (the largest).
#   retriever  sorted 0.5 1.0 1.5 2.0   -> p50 1.0 s,  p95 2.0 s
#   generation sorted 3.0 5.5 8.5 15.0  -> p50 5.5 s,  p95 15.0 s
#   tools      sorted 0 0.4 0.5 0.5     -> p50 0.4 s,  p95 0.5 s
#   share      sorted 68.75 70.83 75 75 -> median (2nd smallest) 70.8%


def test_the_time_by_step_table_lists_the_traced_row_with_hand_checked_numbers(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    section = document.split("### Time by step")[1].split("### Tool calls")[0]
    assert (
        "| Configuration | Retriever p50 | Retriever p95 | Generation p50 | Generation p95 | "
        "Tools p50 | Tools p95 | Generation share (median) |"
    ) in section
    assert "| agent | 1.0 s | 2.0 s | 5.5 s | 15.0 s | 0.4 s | 0.5 s | 70.8% |" in section
    assert "| hybrid |" not in section
    assert "nearest rank" in section


def test_the_time_by_step_table_sits_after_cost_and_latency(fixture_results):
    document = render(fixture_results, STAMP, golden=GOLDEN)

    assert (
        document.index("### Cost and latency")
        < document.index("### Time by step")
        < document.index("### Tool calls")
    )


def test_a_row_without_tools_has_dashes_for_the_tool_columns(fixture_results):
    sections = replace(
        fixture_results["agent"],
        config=replace(fixture_results["agent"].config, name="sections", tools=False),
    )
    fixture_results["sections"] = sections

    section = render(fixture_results, STAMP, golden=GOLDEN).split("### Time by step")[1]

    assert "| sections | 1.0 s | 2.0 s | 5.5 s | 15.0 s | — | — | 70.8% |" in section
    assert "| agent | 1.0 s | 2.0 s | 5.5 s | 15.0 s | 0.4 s | 0.5 s | 70.8% |" in section


def test_a_row_with_any_question_lacking_a_trace_is_left_out_of_the_time_table(fixture_results):
    fixture_results["sections"] = replace(
        fixture_results["agent"],
        config=replace(fixture_results["agent"].config, name="sections", tools=False),
        questions=[replace(question) for question in fixture_results["agent"].questions],
    )
    fixture_results["sections"].questions[2].trace = None

    section = render(fixture_results, STAMP, golden=GOLDEN).split("### Time by step")[1]

    assert "| agent |" in section.split("### Tool calls")[0]
    assert "| sections |" not in section.split("### Tool calls")[0]


def test_the_time_by_step_table_is_left_out_while_no_row_is_fully_traced(fixture_results):
    fixture_results["agent"].questions[0].trace = None

    assert "Time by step" not in render(fixture_results, STAMP, golden=GOLDEN)


def test_the_time_by_step_table_is_left_out_while_no_row_has_a_trace():
    untraced = {
        name: result for name, result in latest_per_config(FIXTURES).items() if name != "agent"
    }

    assert "Time by step" not in render(untraced, STAMP, golden=GOLDEN)


def test_a_partial_run_never_becomes_a_row_even_when_it_is_the_latest_file(tmp_path):
    from evals.results import load, save

    full = load(FIXTURES / "20260921T031500Z_hybrid.json")
    partial = replace(
        full,
        run_id="20260922T000000Z_hybrid",
        created_at="2026-09-22T00:00:00Z",
        config=replace(full.config, only=["q001"]),
        questions=full.questions[:1],
    )
    save(full, tmp_path / "full.json")
    save(partial, tmp_path / "partial.json")

    latest = latest_per_config(tmp_path)

    assert latest["hybrid"].run_id == full.run_id
    assert len(latest["hybrid"].questions) == len(full.questions)


def test_a_directory_holding_only_partial_runs_has_no_rows(tmp_path):
    from evals.results import load, save

    full = load(FIXTURES / "20260921T031500Z_hybrid.json")
    save(replace(full, config=replace(full.config, only=["q001"])), tmp_path / "partial.json")

    assert latest_per_config(tmp_path) == {}


def rerank_call(error: str | None):
    from fineprint.tracing import CallRecord

    return CallRecord(
        kind="rerank",
        name="rerank",
        model="cohere.rerank-v3-5:0",
        input_tokens=None,
        output_tokens=None,
        units=1,
        latency_ms=100.0,
        cost_usd=0.002,
        error=error,
    )


def with_reranker(result, failures: tuple[bool, ...]):
    """The agent fixture as a re-ranking row whose questions' re-ranks failed as `failures` says."""
    questions = [
        replace(
            question,
            trace=replace(
                question.trace,
                calls=[*question.trace.calls, rerank_call("throttled" if failed else None)],
            ),
        )
        for question, failed in zip(result.questions, failures, strict=True)
    ]
    return replace(
        result, config=replace(result.config, reranker="cohere.rerank-v3-5:0"), questions=questions
    )


def test_the_re_ranks_skipped_column_counts_questions_whose_rerank_failed(fixture_results):
    fixture_results["agent"] = with_reranker(fixture_results["agent"], (True, False, True, False))

    section = render(fixture_results, STAMP, golden=GOLDEN).split("### Cost and latency")[1]

    assert "| agent | $0.0500 | 8.0 s | 20.0 s | 20.0 s | 2.00 | 1.25 | 1 | 2 |" in section


def test_a_re_ranking_row_that_skipped_nothing_shows_zero_not_a_dash(fixture_results):
    fixture_results["agent"] = with_reranker(fixture_results["agent"], (False,) * 4)

    section = render(fixture_results, STAMP, golden=GOLDEN).split("### Cost and latency")[1]

    assert "| 1 | 0 |" in section


def test_the_re_ranks_skipped_column_is_a_dash_when_a_question_has_no_trace(fixture_results):
    fixture_results["agent"] = with_reranker(fixture_results["agent"], (True,) * 4)
    fixture_results["agent"].questions[2].trace = None

    section = render(fixture_results, STAMP, golden=GOLDEN).split("### Cost and latency")[1]

    assert "| agent | — | — | — | — | — | — | — | — |" in section
