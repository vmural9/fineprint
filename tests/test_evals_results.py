"""Tests for the results file: what a run records, and how it is read back."""

import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.metrics import Aggregates, ManualPass, QuestionMetrics
from evals.results import (
    RESULTS_DIR,
    Corpus,
    GitState,
    GoldenSetInfo,
    QuestionResult,
    ResultsError,
    RetrievedChunk,
    Review,
    RunConfig,
    RunResult,
    Scoring,
    git_state,
    latest_per_config,
    load,
    load_all,
    make_run_id,
    results_path,
    save,
    sha256_of,
)

HYBRID = RunConfig(
    name="hybrid",
    mode="hybrid",
    chunk_set="fixed-220w",
    top_k=5,
    candidates=20,
    rrf_k=60,
    embedding_model="amazon.titan-embed-text-v2:0",
    llm_model="us.anthropic.claude-opus-5",
)

# The three files part 1 committed, written before any part 2 field existed.
PART_1_FILES = (
    "20260921T011412Z_lexical-only.json",
    "20260921T011428Z_vector-only.json",
    "20260921T012044Z_hybrid.json",
)


def question(item_id: str = "q001", *, verdict: str | None = None) -> QuestionResult:
    """One answered question's row, with a hit on its single expected page."""
    return QuestionResult(
        id=item_id,
        question="How much is the Part B premium?",
        type="lookup",
        category="costs",
        difficulty="easy",
        answerable=True,
        expected_pages=[23],
        retrieved=[
            RetrievedChunk(
                chunk_id=141,
                page_start=22,
                page_end=23,
                rank=1,
                score=0.0328,
                lexical_rank=1,
                vector_rank=3,
            )
        ],
        metrics=QuestionMetrics(1.0, 1.0, 1.0, 1.0, 1.0),
        answer={"answer": "$202.90 a month.", "found_in_handbook": True},
        review=Review(verdict=verdict, note="", reviewer=None, reviewed_at=None),
    )


def run(config_name: str = "hybrid", *, run_id: str = "20260921T031500Z_hybrid") -> RunResult:
    """A one-question run, ready to save."""
    return RunResult(
        run_id=run_id,
        created_at="2026-09-21T03:15:00Z",
        git=GitState(commit="0123456789abcdef0123456789abcdef01234567", dirty=False),
        config=replace(HYBRID, name=config_name),
        corpus=Corpus(edition=2026, sha256="d7a341bc3d2d3dab59af746a0875752761e6c2f2"),
        golden_set=GoldenSetInfo(sha256="ab" * 32, count=15),
        questions=[question()],
    )


# --- run ids and paths ------------------------------------------------------------


def test_a_run_id_is_a_utc_timestamp_and_the_configuration_name():
    when = datetime(2026, 9, 21, 3, 15, 0, tzinfo=UTC)

    assert make_run_id("vector-only", when) == "20260921T031500Z_vector-only"


def test_a_run_id_allows_a_plus_for_a_rerank_configuration():
    when = datetime(2026, 9, 21, 3, 15, 0, tzinfo=UTC)

    assert make_run_id("hybrid+rerank", when) == "20260921T031500Z_hybrid+rerank"


def test_a_results_file_is_named_after_its_run(tmp_path: Path):
    assert results_path("20260921T031500Z_hybrid", tmp_path).name == "20260921T031500Z_hybrid.json"


# --- saving and loading -----------------------------------------------------------


def test_a_run_survives_being_written_and_read_back(tmp_path: Path):
    path = save(run(), results_path(run().run_id, tmp_path))

    assert load(path) == run()


def test_a_run_with_every_new_field_filled_survives_being_written_and_read_back(tmp_path: Path):
    """Every field part 2 adds — the re-ranker knobs, the retrieved text and ranks, the
    per-question scoring detail, and the scoring block itself — round-trips unchanged."""
    filled = replace(
        run(),
        config=replace(
            HYBRID, name="hybrid+rerank", reranker="cohere.rerank-v3-5:0", rerank_candidates=20
        ),
        questions=[
            replace(
                question(),
                metrics=QuestionMetrics(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    context_recall=0.9,
                    context_precision=0.75,
                    faithfulness=1.0,
                    answer_relevance=0.85,
                ),
                retrieved=[
                    RetrievedChunk(
                        chunk_id=141,
                        page_start=22,
                        page_end=23,
                        rank=1,
                        score=0.91,
                        lexical_rank=1,
                        vector_rank=3,
                        ordinal=42,
                        section="Part B costs",
                        text="The standard Part B premium amount in 2026 is $202.90.",
                        fused_rank=4,
                        rerank_score=0.91,
                    )
                ],
                scoring_detail={"sentences": [{"text": "$202.90 a month.", "supported": True}]},
            )
        ],
        scoring=Scoring(
            judge_model="us.anthropic.claude-sonnet-5",
            embedding_model="amazon.titan-embed-text-v2:0",
            prompts_sha256="ab" * 32,
            scored_at="2026-09-27T00:00:00Z",
            judge_input_tokens=1200,
            judge_output_tokens=300,
        ),
    )

    reloaded = load(save(filled, results_path(filled.run_id, tmp_path)))

    assert reloaded == filled


def test_saving_creates_the_results_directory_when_it_is_missing(tmp_path: Path):
    directory = tmp_path / "results"

    save(run(), results_path(run().run_id, directory))

    assert (directory / "20260921T031500Z_hybrid.json").is_file()


def test_a_saved_run_is_readable_json_with_the_documented_keys(tmp_path: Path):
    path = save(run(), results_path(run().run_id, tmp_path))

    record = json.loads(path.read_text(encoding="utf-8"))

    assert set(record) == {
        "run_id",
        "created_at",
        "git",
        "config",
        "corpus",
        "golden_set",
        "questions",
        "scoring",
    }
    assert record["git"] == {"commit": "0123456789abcdef0123456789abcdef01234567", "dirty": False}
    assert record["config"]["llm_model"] == "us.anthropic.claude-opus-5"
    assert record["config"]["reranker"] is None
    assert record["scoring"] is None
    assert record["questions"][0]["retrieved"][0]["rank"] == 1
    assert record["questions"][0]["scoring_detail"] is None
    assert record["questions"][0]["review"] == {
        "verdict": None,
        "note": "",
        "reviewer": None,
        "reviewed_at": None,
    }


def test_a_retrieval_only_run_records_no_answer_and_no_llm_model(tmp_path: Path):
    retrieval_only = run()
    retrieval_only.config = replace(HYBRID, llm_model=None)
    retrieval_only.questions[0].answer = None

    reloaded = load(save(retrieval_only, results_path(retrieval_only.run_id, tmp_path)))

    assert reloaded.config.llm_model is None
    assert reloaded.questions[0].answer is None
    assert reloaded.questions[0].answered is False


def test_a_file_that_is_not_a_results_file_is_refused_by_name(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text('{"run_id": "x"}', encoding="utf-8")

    with pytest.raises(ResultsError) as error:
        load(path)

    assert "broken.json" in str(error.value)
    assert "created_at" in str(error.value)


def test_a_file_that_is_not_json_at_all_is_refused_by_name(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(ResultsError) as error:
        load(path)

    assert "broken.json" in str(error.value)


# --- part 1's committed files still load ------------------------------------------


@pytest.mark.parametrize("filename", PART_1_FILES)
def test_part_1s_committed_results_files_still_load_with_every_new_field_defaulting_to_none(
    filename: str,
):
    """Every field part 2 adds defaults to `None`, so a file written before any of them existed
    still loads — this is what lets the scoreboard go on reading part 1's three runs unchanged."""
    result = load(RESULTS_DIR / filename)

    assert result.config.reranker is None
    assert result.config.rerank_candidates is None
    assert result.scoring is None
    assert result.questions  # sanity: the file was not empty
    for row in result.questions:
        assert row.scoring_detail is None
        for chunk in row.retrieved:
            assert chunk.ordinal is None
            assert chunk.section is None
            assert chunk.text is None
            assert chunk.fused_rank is None
            assert chunk.rerank_score is None


# --- finding the latest run per configuration -------------------------------------


def test_the_latest_run_of_each_configuration_wins(tmp_path: Path):
    for run_id in ("20260920T090000Z_hybrid", "20260921T031500Z_hybrid"):
        save(run("hybrid", run_id=run_id), results_path(run_id, tmp_path))
    save(
        run("lexical-only", run_id="20260919T080000Z_lexical-only"),
        results_path("20260919T080000Z_lexical-only", tmp_path),
    )

    latest = latest_per_config(tmp_path)

    assert set(latest) == {"hybrid", "lexical-only"}
    assert latest["hybrid"].run_id == "20260921T031500Z_hybrid"
    assert latest["lexical-only"].run_id == "20260919T080000Z_lexical-only"


def test_the_configuration_name_inside_the_file_decides_the_row_not_the_file_name(tmp_path: Path):
    save(
        run("vector-only", run_id="20260921T031500Z_hybrid"),
        results_path("20260921T031500Z_hybrid", tmp_path),
    )

    assert set(latest_per_config(tmp_path)) == {"vector-only"}


def test_nothing_to_load_is_not_an_error(tmp_path: Path):
    assert load_all(tmp_path) == []
    assert latest_per_config(tmp_path) == {}
    assert latest_per_config(tmp_path / "never-created") == {}


def test_files_that_are_not_results_are_ignored(tmp_path: Path):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# not a run", encoding="utf-8")
    save(run(), results_path(run().run_id, tmp_path))

    assert [result.run_id for result in load_all(tmp_path)] == ["20260921T031500Z_hybrid"]


def test_a_json_file_in_a_subdirectory_is_not_a_run_and_is_ignored(tmp_path: Path):
    """`candidates/`, where the candidate-rank diagnostic writes its own differently shaped
    files, is the motivating case: this file has no `config` field, so if `load_all` ever read
    it, it would fail the whole scoreboard with "not a results file", not just skip one row."""
    subdirectory = tmp_path / "candidates"
    subdirectory.mkdir()
    (subdirectory / "20260926T145549Z_candidates-fixed-220w-hybrid.json").write_text(
        json.dumps({"run_id": "not-a-run", "chunk_set": "fixed-220w"}), encoding="utf-8"
    )
    save(run(), results_path(run().run_id, tmp_path))

    assert [result.run_id for result in load_all(tmp_path)] == ["20260921T031500Z_hybrid"]
    assert set(latest_per_config(tmp_path)) == {"hybrid"}


# --- aggregates come from the per-question rows -----------------------------------


def test_a_run_recomputes_its_aggregates_from_its_questions():
    result = run()
    result.questions = [question("q001", verdict="pass"), question("q002", verdict="fail")]

    assert result.aggregates == Aggregates(
        questions=2,
        answerable=2,
        answered=2,
        page_hit=1.0,
        page_recall=1.0,
        mrr=1.0,
        cited_page_hit=1.0,
        abstention_accuracy=1.0,
        manual=ManualPass(passes=1, reviewed=2),
        context_recall=None,
        context_precision=None,
        faithfulness=None,
        answer_relevance=None,
    )


# --- provenance -------------------------------------------------------------------


def test_the_sha256_of_a_file_is_the_sha256_of_its_bytes(tmp_path: Path):
    path = tmp_path / "golden_set.jsonl"
    path.write_bytes(b"")

    assert sha256_of(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_git_state_reports_the_commit_and_whether_the_tree_was_dirty(tmp_path: Path):
    git = ["git", "-c", "user.email=t@example.com", "-c", "user.name=test"]
    subprocess.run([*git, "init", "-q", "-b", "main", str(tmp_path)], check=True)
    (tmp_path / "file.txt").write_text("one", encoding="utf-8")
    subprocess.run([*git, "-C", str(tmp_path), "add", "file.txt"], check=True)
    subprocess.run([*git, "-C", str(tmp_path), "commit", "-q", "-m", "first"], check=True)

    clean = git_state(tmp_path)
    (tmp_path / "file.txt").write_text("two", encoding="utf-8")
    dirty = git_state(tmp_path)

    assert clean.commit is not None and len(clean.commit) == 40
    assert clean.dirty is False
    assert dirty == GitState(commit=clean.commit, dirty=True)


def test_git_state_outside_a_repository_says_so_rather_than_failing(tmp_path: Path):
    assert git_state(tmp_path) == GitState(commit=None, dirty=False)
