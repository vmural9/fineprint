"""Tests for the candidate-rank diagnostic, driven with a fake retriever.

The real `Retriever.candidates()` talks to Postgres and, in hybrid and vector mode, to Bedrock.
Everything under test here takes a retriever as an argument precisely so these tests can hand it
a stand-in instead, and never call either.
"""

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from evals.candidates import (
    QuestionRank,
    bucket_for,
    build_parser,
    first_hit_rank,
    main,
    rank_question,
    rank_questions,
    render_outliers,
    render_table,
)
from evals.golden import GoldenItem
from evals.results import Corpus


@dataclass(frozen=True)
class FakeChunk:
    """A stand-in for `fineprint.retrieval.RetrievedChunk`: only what this diagnostic reads."""

    page_start: int
    page_end: int
    rank: int


@dataclass
class FakeRetriever:
    """Returns a canned pool per question and remembers how it was called."""

    pools: dict[str, list[FakeChunk]]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def candidates(self, question: str, mode: str = "hybrid") -> list[FakeChunk]:
        self.calls.append((question, mode))
        return self.pools.get(question, [])


def golden(
    item_id: str = "q001",
    *,
    question: str | None = None,
    item_type: str = "lookup",
    expected_pages: tuple[int, ...] = (23,),
    alt_pages: dict[int, tuple[int, ...]] | None = None,
) -> GoldenItem:
    """A golden item with only the fields this diagnostic reads spelled out."""
    return GoldenItem(
        id=item_id,
        question=question or f"question for {item_id}",
        expected_answer="an answer with exact figures.",
        expected_pages=expected_pages,
        category="costs",
        difficulty="easy",
        type=item_type,
        evidence=(),
        alt_pages=alt_pages or {},
    )


# --- first_hit_rank and bucket_for -------------------------------------------------


def test_a_chunk_covering_the_expected_page_at_rank_3_lands_in_bucket_1_to_5():
    item = golden(expected_pages=(23,))
    pool = [FakeChunk(page_start=22, page_end=23, rank=3)]

    assert first_hit_rank(item, pool) == 3
    assert bucket_for(3) == "1-5"


def test_a_hit_at_rank_12_lands_in_bucket_6_to_20():
    assert bucket_for(12) == "6-20"


def test_a_hit_at_rank_35_lands_in_bucket_21_plus():
    assert bucket_for(35) == "21+"


def test_no_covering_chunk_in_the_pool_is_absent():
    item = golden(expected_pages=(23,))
    pool = [FakeChunk(page_start=90, page_end=91, rank=1)]

    assert first_hit_rank(item, pool) is None
    assert bucket_for(None) == "absent"


def test_the_minimum_rank_wins_regardless_of_the_pool_s_list_order():
    """`candidates()` returns the pool in rank order, but the rule must not depend on that: it
    reads each chunk's own `.rank`, so the lowest one wins even out of list order."""
    item = golden(expected_pages=(23,))
    pool = [
        FakeChunk(page_start=90, page_end=91, rank=9),
        FakeChunk(page_start=22, page_end=23, rank=2),
    ]

    assert first_hit_rank(item, pool) == 2


def test_a_chunk_on_an_alternate_page_counts_as_covering_the_expected_page():
    item = golden(expected_pages=(30,), alt_pages={30: (23,)})
    pool = [FakeChunk(page_start=23, page_end=23, rank=1)]

    assert first_hit_rank(item, pool) == 1


def test_a_rank_deeper_than_40_is_still_found_not_mislabelled_absent():
    assert bucket_for(41) == "21+"


# --- rank_questions: the loop, and skipping unanswerable questions -----------------


def test_rank_questions_covers_one_case_per_bucket_and_skips_unanswerable():
    at_rank_3 = golden("q001", expected_pages=(23,))
    at_rank_12 = golden("q002", expected_pages=(50,))
    at_rank_35 = golden("q003", expected_pages=(70,))
    missing = golden("q004", expected_pages=(99,))
    unanswerable = golden("q005", item_type="unanswerable", expected_pages=())

    retriever = FakeRetriever(
        {
            at_rank_3.question: [FakeChunk(page_start=22, page_end=23, rank=3)],
            at_rank_12.question: [FakeChunk(page_start=50, page_end=50, rank=12)],
            at_rank_35.question: [FakeChunk(page_start=70, page_end=70, rank=35)],
            missing.question: [FakeChunk(page_start=1, page_end=2, rank=1)],
        }
    )

    rows = rank_questions(
        [at_rank_3, at_rank_12, at_rank_35, missing, unanswerable], retriever, mode="hybrid"
    )

    assert [(row.id, row.first_hit_rank, row.bucket) for row in rows] == [
        ("q001", 3, "1-5"),
        ("q002", 12, "6-20"),
        ("q003", 35, "21+"),
        ("q004", None, "absent"),
    ]
    # The unanswerable question was never even sent to the retriever.
    assert unanswerable.question not in [call[0] for call in retriever.calls]


def test_rank_questions_passes_the_mode_through_to_candidates():
    item = golden(expected_pages=(23,))
    retriever = FakeRetriever({item.question: []})

    rank_questions([item], retriever, mode="lexical")

    assert retriever.calls == [(item.question, "lexical")]


def test_rank_question_builds_one_row_from_one_item_and_its_pool():
    item = golden("q009", item_type="table", expected_pages=(11, 12))

    row = rank_question(item, [FakeChunk(page_start=11, page_end=11, rank=6)])

    assert row == QuestionRank(
        id="q009", type="table", expected_pages=[11, 12], first_hit_rank=6, bucket="6-20"
    )


# --- render_table -------------------------------------------------------------------


def test_render_table_has_one_row_per_question_type_plus_all():
    rows = [
        QuestionRank(id="q001", type="lookup", expected_pages=[23], first_hit_rank=3, bucket="1-5"),
        QuestionRank(
            id="q002", type="lookup", expected_pages=[50], first_hit_rank=12, bucket="6-20"
        ),
        QuestionRank(
            id="q003", type="table", expected_pages=[11, 12], first_hit_rank=None, bucket="absent"
        ),
    ]

    lines = render_table(rows)
    by_type = {line.split()[0]: line for line in lines[1:]}

    assert set(by_type) == {"lookup", "table", "multi_section", "all"}
    # lookup: one in 1-5, one in 6-20, none in 21+ or absent.
    assert by_type["lookup"].split()[1:] == ["1", "1", "0", "0", "2"]
    # table: one absent.
    assert by_type["table"].split()[1:] == ["0", "0", "0", "1", "1"]
    # multi_section had no questions at all: an all-zero row, not a missing one.
    assert by_type["multi_section"].split()[1:] == ["0", "0", "0", "0", "0"]
    # all: every question, in the same four buckets.
    assert by_type["all"].split()[1:] == ["1", "1", "0", "1", "3"]


def test_render_table_header_names_all_four_buckets():
    header = render_table([])[0]

    assert header.split()[:4] == ["type", "1-5", "6-20", "21+"]
    assert "absent" in header


# --- render_outliers -----------------------------------------------------------------


def test_render_outliers_skips_rank_1_to_5_and_names_rank_and_expected_pages():
    rows = [
        QuestionRank(id="q001", type="lookup", expected_pages=[23], first_hit_rank=3, bucket="1-5"),
        QuestionRank(
            id="q034", type="lookup", expected_pages=[82], first_hit_rank=6, bucket="6-20"
        ),
        QuestionRank(
            id="q019",
            type="table",
            expected_pages=[11, 12],
            first_hit_rank=None,
            bucket="absent",
        ),
    ]

    lines = render_outliers(rows)

    assert len(lines) == 2
    assert lines[0].startswith("q034") and "lookup" in lines[0]
    assert "rank 6" in lines[0] and "expected [82]" in lines[0]
    assert "rank absent" in lines[1] and "expected [11, 12]" in lines[1]


def test_render_outliers_is_empty_when_everything_is_in_the_top_5():
    rows = [
        QuestionRank(id="q001", type="lookup", expected_pages=[23], first_hit_rank=1, bucket="1-5")
    ]

    assert render_outliers(rows) == []


# --- the command ----------------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch):
    """Replace the database and the retriever so `main` can run offline."""

    def wire(items, retriever):
        @contextmanager
        def fake_pool(database_url: str):
            yield object()

        monkeypatch.setattr("evals.candidates.load_golden_set", lambda path: items)
        monkeypatch.setattr("evals.candidates.connection_pool", fake_pool)
        monkeypatch.setattr(
            "evals.candidates.corpus_from_database",
            lambda pool, edition: Corpus(edition=edition, sha256="d7a341bc3d2d"),
        )
        monkeypatch.setattr("evals.candidates.build_retriever", lambda settings, pool: retriever)

    return wire


def test_the_command_refuses_a_mode_it_does_not_know(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--mode", "magic"])

    assert exit_info.value.code == 2
    assert "hybrid" in capsys.readouterr().err


def test_the_command_writes_a_json_file_with_the_documented_keys(wired, tmp_path):
    item = golden("q001", expected_pages=(23,))
    retriever = FakeRetriever({item.question: [FakeChunk(page_start=22, page_end=23, rank=3)]})
    wired([item], retriever)

    exit_code = main(
        ["--chunk-set", "fixed-220w", "--mode", "hybrid", "--results-dir", str(tmp_path)]
    )

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    assert "candidates-fixed-220w-hybrid" in written.name

    record = json.loads(written.read_text(encoding="utf-8"))
    assert set(record) == {
        "run_id",
        "created_at",
        "git",
        "chunk_set",
        "mode",
        "pool_size",
        "corpus",
        "golden_set",
        "questions",
    }
    assert record["chunk_set"] == "fixed-220w"
    assert record["mode"] == "hybrid"
    assert record["pool_size"] == 20
    assert record["corpus"] == {"edition": 2026, "sha256": "d7a341bc3d2d"}
    assert record["golden_set"]["count"] == 1
    assert record["questions"] == [
        {
            "id": "q001",
            "type": "lookup",
            "expected_pages": [23],
            "first_hit_rank": 3,
            "bucket": "1-5",
        }
    ]
    assert retriever.calls == [(item.question, "hybrid")]


def test_the_command_uses_settings_chunk_set_when_none_is_given(wired, tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNK_SET", "sections")
    item = golden("q001", expected_pages=(23,))
    wired([item], FakeRetriever({item.question: []}))

    exit_code = main(["--results-dir", str(tmp_path)])

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    assert "candidates-sections-hybrid" in written.name
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["chunk_set"] == "sections"


def test_the_command_prints_the_table_and_an_outlier_line(wired, tmp_path, capsys):
    item = golden("q034", expected_pages=(82,))
    retriever = FakeRetriever({item.question: [FakeChunk(page_start=82, page_end=82, rank=6)]})
    wired([item], retriever)

    main(["--chunk-set", "fixed-220w", "--mode", "hybrid", "--results-dir", str(tmp_path)])

    printed = capsys.readouterr().out
    assert "type " in printed and "absent" in printed
    assert "q034" in printed and "rank 6" in printed and "expected [82]" in printed
    assert "wrote " in printed


def test_the_results_directory_is_created_when_it_is_missing(wired, tmp_path):
    item = golden("q001", expected_pages=(23,))
    wired([item], FakeRetriever({item.question: []}))
    missing = tmp_path / "results"

    exit_code = main(["--chunk-set", "fixed-220w", "--results-dir", str(missing)])

    assert exit_code == 0
    assert len(list(missing.glob("*.json"))) == 1
