"""Tests for the golden-set runner, driven with a fake retriever and a fake answer function.

The real retriever and the real answer function talk to Postgres and to Bedrock. The loop under
test takes both as arguments precisely so that these tests can hand it stand-ins instead.
"""

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from evals.configs import CONFIGS as SHARED_CONFIGS
from evals.configs import settings_for
from evals.golden import Evidence, GoldenItem
from evals.results import Corpus
from evals.run_golden_set import CONFIGS, build_parser, build_retriever, main, run_questions
from fineprint.answer import Citation, DraftAnswer, answer_question
from fineprint.config import Settings
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker


@dataclass(frozen=True)
class FakeChunk:
    """A stand-in for `fineprint.retrieval.RetrievedChunk`."""

    chunk_id: int
    page_start: int
    page_end: int
    rank: int
    text: str = "..."
    score: float = 0.03
    lexical_rank: int | None = 1
    vector_rank: int | None = 2
    ordinal: int | None = 1
    section: str | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class FakeCitation:
    """A stand-in for `fineprint.answer.CitedChunk`."""

    chunk_id: int
    page_start: int
    page_end: int
    quote: str = "The standard Part B premium amount in 2026 is $202.90."
    quote_verified: bool = True


@dataclass(frozen=True)
class FakeAnswer:
    """A stand-in for `fineprint.answer.AnswerResponse`."""

    question: str
    answer: str
    found_in_handbook: bool
    citations: list[FakeCitation]
    confidence: str = "high"


@dataclass
class FakeRetriever:
    """Returns canned chunks per question and remembers how it was called."""

    chunks: dict[str, list[FakeChunk]]
    calls: list[tuple[str, str, int | None]] = field(default_factory=list)
    fails_on: str | None = None

    def search(
        self, question: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[FakeChunk]:
        self.calls.append((question, mode, top_k))
        if question == self.fails_on:
            raise RuntimeError('relation "chunks" does not exist')
        return self.chunks.get(question, [])


def golden(
    item_id: str = "q001",
    *,
    question: str = "How much is the Part B premium?",
    item_type: str = "lookup",
    expected_pages: tuple[int, ...] = (23,),
) -> GoldenItem:
    """A golden item with only the fields the runner reads spelled out."""
    return GoldenItem(
        id=item_id,
        question=question,
        expected_answer="$202.90 a month in 2026.",
        expected_pages=expected_pages,
        category="costs",
        difficulty="easy",
        type=item_type,
        evidence=(Evidence(page=23, quote="The standard Part B premium amount in 2026"),),
    )


ON_PAGE_23 = [FakeChunk(chunk_id=141, page_start=22, page_end=23, rank=1)]
ELSEWHERE = [FakeChunk(chunk_id=9, page_start=90, page_end=91, rank=1)]


def answered(question: str, *, found: bool = True, pages: int = 23) -> FakeAnswer:
    """An answer that cites one chunk on the given page."""
    return FakeAnswer(
        question=question,
        answer="The standard Part B premium in 2026 is $202.90 a month.",
        found_in_handbook=found,
        citations=[FakeCitation(chunk_id=141, page_start=pages, page_end=pages)],
    )


# --- the loop ---------------------------------------------------------------------


def test_the_loop_records_the_retrieved_chunks_and_scores_the_question():
    item = golden()
    retriever = FakeRetriever({item.question: ON_PAGE_23})

    rows = run_questions([item], retriever, mode="hybrid", top_k=5, report=lambda line: None)

    (row,) = rows
    assert row.id == "q001"
    assert [(chunk.chunk_id, chunk.rank, chunk.page_end) for chunk in row.retrieved] == [
        (141, 1, 23)
    ]
    assert row.metrics.page_hit == 1.0
    assert row.metrics.reciprocal_rank == 1.0
    assert row.error is None


def test_the_loop_carries_the_new_retrieval_fields_into_the_results_file():
    """D5: from part 2 on, a results file stores each retrieved chunk's text, ordinal, section
    and both ranks, not just where it sits and how it ranked."""
    item = golden()
    chunk = FakeChunk(
        chunk_id=141,
        page_start=22,
        page_end=23,
        rank=1,
        text="The standard Part B premium amount in 2026 is $202.90.",
        ordinal=7,
        section="Part B costs",
        fused_rank=4,
        rerank_score=0.91,
    )
    retriever = FakeRetriever({item.question: [chunk]})

    (row,) = run_questions([item], retriever, mode="hybrid", top_k=5, report=lambda line: None)

    (retrieved,) = row.retrieved
    assert retrieved.text == chunk.text
    assert retrieved.ordinal == 7
    assert retrieved.section == "Part B costs"
    assert retrieved.fused_rank == 4
    assert retrieved.rerank_score == 0.91


def test_the_loop_searches_in_the_configured_mode_at_the_configured_k():
    item = golden()
    retriever = FakeRetriever({item.question: ON_PAGE_23})

    run_questions([item], retriever, mode="lexical", top_k=5, report=lambda line: None)

    assert retriever.calls == [(item.question, "lexical", 5)]


def test_a_retrieval_only_run_records_no_answer_and_no_answer_metrics():
    item = golden()

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ON_PAGE_23}),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
    )

    assert row.answer is None
    assert row.answered is False
    assert row.metrics.cited_page_hit is None
    assert row.metrics.abstention_correct is None


def test_an_answer_is_stored_whole_and_scored_on_its_citations():
    item = golden()

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ON_PAGE_23}),
        lambda question, chunks: answered(question),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
    )

    assert row.answer["answer"].startswith("The standard Part B premium")
    assert row.answer["citations"][0]["page_start"] == 23
    assert row.metrics.cited_page_hit == 1.0
    assert row.metrics.abstention_correct == 1.0


def test_an_answer_that_cites_the_wrong_page_scores_zero_on_cited_page_hit():
    item = golden()

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ON_PAGE_23}),
        lambda question, chunks: answered(question, pages=90),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
    )

    assert row.metrics.cited_page_hit == 0.0


def test_an_unanswerable_question_is_scored_on_its_abstention_alone():
    item = golden(
        "q003",
        question="What are the 2026 Extra Help limits?",
        item_type="unanswerable",
        expected_pages=(),
    )

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ELSEWHERE}),
        lambda question, chunks: answered(question, found=False),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
    )

    assert row.answerable is False
    assert (row.metrics.page_hit, row.metrics.page_recall, row.metrics.reciprocal_rank) == (
        None,
        None,
        None,
    )
    assert row.metrics.abstention_correct == 1.0


# --- one failure does not stop the run --------------------------------------------


def test_a_question_whose_retrieval_fails_is_recorded_and_the_run_carries_on():
    first, second = golden("q001"), golden("q002", question="What does Part D cover?")
    retriever = FakeRetriever({second.question: ON_PAGE_23}, fails_on=first.question)

    rows = run_questions(
        [first, second], retriever, mode="hybrid", top_k=5, report=lambda line: None
    )

    assert [row.id for row in rows] == ["q001", "q002"]
    assert "relation" in rows[0].error
    assert rows[0].retrieved == [] and rows[0].metrics.page_hit is None
    assert rows[1].error is None and rows[1].metrics.page_hit == 1.0


def test_a_question_whose_answer_fails_keeps_its_retrieval_metrics():
    item = golden()

    def explode(question: str, chunks) -> FakeAnswer:
        raise RuntimeError("Bedrock said no")

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ON_PAGE_23}),
        explode,
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
    )

    assert row.metrics.page_hit == 1.0
    assert row.answer is None
    assert "Bedrock said no" in row.error


# --- retrieve once, and the answer describes what was retrieved -------------------


def test_the_loop_retrieves_once_and_the_answer_describes_the_same_chunks():
    """The bug this fixes: `answer_question` used to search again on its own, so the answer's
    evidence and the retrieval metrics could silently describe different chunks."""
    item = golden()
    retriever = FakeRetriever({item.question: ON_PAGE_23})
    chat = FakeChatModel(
        DraftAnswer(
            answer="The standard Part B premium in 2026 is $202.90 a month.",
            found_in_handbook=True,
            citations=[Citation(chunk_id=141, quote="The standard Part B premium amount in 2026")],
            confidence="high",
        )
    )

    def answer_fn(question: str, chunks):
        return answer_question(question, retriever, chat, chunks=chunks)

    (row,) = run_questions(
        [item], retriever, answer_fn, mode="hybrid", top_k=5, report=lambda line: None
    )

    assert retriever.calls == [(item.question, "hybrid", 5)], "retrieval happens exactly once"
    assert row.answer["retrieved_chunk_ids"] == [chunk.chunk_id for chunk in row.retrieved]


# --- what the operator sees -------------------------------------------------------


def test_one_progress_line_per_question_naming_the_question_and_its_hit():
    items = [golden("q001"), golden("q002", question="What does Part D cover?")]
    lines: list[str] = []

    run_questions(
        items,
        FakeRetriever({items[0].question: ON_PAGE_23}),
        mode="hybrid",
        top_k=5,
        report=lines.append,
    )

    assert len(lines) == 2
    assert lines[0].startswith("q001")
    assert "hit 1" in lines[0] and "hit 0" in lines[1]


# --- the six configurations --------------------------------------------------------


def test_the_runners_configs_are_the_shared_registry():
    """`run_golden_set` names the one registry in `evals.configs`, not a copy of its own, so
    a new row there needs no change here."""
    assert CONFIGS is SHARED_CONFIGS


def test_the_configurations_cover_the_six_scoreboard_rows():
    assert sorted(CONFIGS) == [
        "hybrid",
        "hybrid+rerank",
        "lexical-only",
        "sections",
        "sections+rerank",
        "vector-only",
    ]


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_build_retriever_wires_each_configurations_own_settings_and_reranker(name, monkeypatch):
    """`build_retriever` builds the `Settings` this configuration means and passes whatever
    `get_reranker` hands back straight into the `Retriever`.

    `get_reranker` is monkeypatched here in `evals.run_golden_set` — not in
    `fineprint.providers.factory` — because the runner imports it at module level precisely so
    a test can replace it, the same way the `wired` fixture below replaces this module's other
    neighbours. `get_embedder` stays a lazy import inside `build_retriever`, so it is replaced at
    its own source instead.
    """
    config = CONFIGS[name]
    fake_reranker = FakeReranker()
    seen: list[Settings] = []

    def fake_get_reranker(settings: Settings):
        seen.append(settings)
        return fake_reranker if config.reranker else None

    monkeypatch.setattr("fineprint.providers.factory.get_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr("evals.run_golden_set.get_reranker", fake_get_reranker)

    settings = settings_for(config, Settings())
    retriever = build_retriever(settings, pool=object())

    assert seen == [settings]
    assert retriever.settings.chunk_set == config.chunk_set
    assert retriever.settings.reranker_provider == ("bedrock" if config.reranker else "none")
    if config.reranker:
        assert retriever.settings.reranker_model == config.reranker
        assert retriever.reranker is fake_reranker
    else:
        assert retriever.reranker is None


# --- the command ------------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch):
    """Replace the database, the embedder and the answer model so `main` can run offline."""

    def wire(items, retriever, answer_fn=None):
        @contextmanager
        def fake_pool(database_url: str):
            yield object()

        def build_answer_function(settings, built_retriever):
            if answer_fn is None:
                raise AssertionError("a --retrieval-only run must not build the answer model")
            return answer_fn

        monkeypatch.setattr("evals.run_golden_set.load_golden_set", lambda path: items)
        monkeypatch.setattr("evals.run_golden_set.connection_pool", fake_pool)
        monkeypatch.setattr(
            "evals.run_golden_set.corpus_from_database",
            lambda pool, edition: Corpus(edition=edition, sha256="d7a341bc3d2d"),
        )
        monkeypatch.setattr(
            "evals.run_golden_set.build_retriever", lambda settings, pool: retriever
        )
        monkeypatch.setattr("evals.run_golden_set.build_answer_function", build_answer_function)

    return wire


def test_the_command_refuses_a_configuration_it_does_not_know(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--config", "magic"])

    assert exit_info.value.code == 2
    assert "hybrid" in capsys.readouterr().err


def test_the_command_writes_a_results_file_and_prints_the_summary(wired, tmp_path, capsys):
    items = [golden("q001"), golden("q002", question="What does Part D cover?")]
    wired(
        items,
        FakeRetriever({items[0].question: ON_PAGE_23, items[1].question: ELSEWHERE}),
        answer_fn=lambda question, chunks: answered(question),
    )

    exit_code = main(["--config", "hybrid", "--results-dir", str(tmp_path)])

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    assert written.name.endswith("_hybrid.json")
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["config"] == {
        "name": "hybrid",
        "mode": "hybrid",
        "chunk_set": "fixed-220w",
        "top_k": 5,
        "candidates": 20,
        "rrf_k": 60,
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "llm_model": "us.anthropic.claude-opus-5",
        "reranker": None,
        "rerank_candidates": None,
    }
    assert record["corpus"] == {"edition": 2026, "sha256": "d7a341bc3d2d"}
    assert record["golden_set"]["count"] == 2
    assert [row["id"] for row in record["questions"]] == ["q001", "q002"]
    assert record["questions"][0]["answer"]["found_in_handbook"] is True

    printed = capsys.readouterr().out
    # One question of the two found its expected page.
    assert "page_hit@5" in printed and "50.0%" in printed
    assert written.name in printed


def test_a_reranking_configuration_records_its_model_and_candidate_pool(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    exit_code = main(["--config", "hybrid+rerank", "--results-dir", str(tmp_path)])

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["config"]["reranker"] == "cohere.rerank-v3-5:0"
    assert record["config"]["rerank_candidates"] == 20
    assert record["config"]["chunk_set"] == "fixed-220w"


def test_a_sections_configuration_records_the_sections_chunk_set(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}))

    exit_code = main(["--config", "sections", "--retrieval-only", "--results-dir", str(tmp_path)])

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["config"]["chunk_set"] == "sections"
    assert record["config"]["reranker"] is None
    assert record["config"]["rerank_candidates"] is None


def test_a_retrieval_only_run_never_builds_the_answer_model(wired, tmp_path):
    items = [golden("q001")]
    retriever = FakeRetriever({items[0].question: ON_PAGE_23})
    wired(items, retriever)

    exit_code = main(
        ["--config", "lexical-only", "--retrieval-only", "--results-dir", str(tmp_path)]
    )

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["config"]["llm_model"] is None
    assert record["questions"][0]["answer"] is None
    assert retriever.calls == [(items[0].question, "lexical", 5)]


def test_the_command_exits_1_when_a_question_failed(wired, tmp_path):
    items = [golden("q001"), golden("q002", question="What does Part D cover?")]
    retriever = FakeRetriever({items[0].question: ON_PAGE_23, items[1].question: ON_PAGE_23})

    def explode(question: str, chunks) -> FakeAnswer:
        if question == items[0].question:
            raise RuntimeError("Bedrock said no")
        return answered(question)

    wired(items, retriever, answer_fn=explode)

    exit_code = main(["--config", "hybrid", "--results-dir", str(tmp_path)])

    assert exit_code == 1


def test_the_results_directory_is_created_when_it_is_missing(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}))
    missing = tmp_path / "results"

    main(["--config", "hybrid", "--retrieval-only", "--results-dir", str(missing)])

    assert len(list(missing.glob("*.json"))) == 1
