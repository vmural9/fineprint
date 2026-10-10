"""Tests for the golden-set runner, driven with a fake retriever and a fake answer function.

The real retriever and the real answer function talk to Postgres and to Bedrock. The loop under
test takes both as arguments precisely so that these tests can hand it stand-ins instead.
"""

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date

import pytest

from evals.configs import CONFIGS as SHARED_CONFIGS
from evals.configs import settings_for
from evals.golden import Evidence, GoldenItem
from evals.metrics import QuestionMetrics
from evals.results import Corpus, GitState, GoldenSetInfo, QuestionResult, RunConfig, RunResult
from evals.run_golden_set import (
    CONFIGS,
    build_answer_function,
    build_parser,
    build_retriever,
    main,
    run_questions,
    summary_lines,
)
from evals.score import tool_contexts
from fineprint.answer import Citation, DraftAnswer, answer_question
from fineprint.config import Settings
from fineprint.pricing import Cost
from fineprint.providers.base import ToolCall
from fineprint.retrieval import RetrievedChunk
from fineprint.tracing import CallRecord, Tracer, TraceRecord
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    RecordingSink,
    ScriptedToolModel,
)


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


def test_the_configurations_cover_the_seven_scoreboard_rows():
    assert sorted(CONFIGS) == [
        "agent",
        "hybrid",
        "hybrid+rerank",
        "lexical-only",
        "sections",
        "sections+rerank",
        "vector-only",
    ]


def test_importing_the_runner_does_not_load_the_retriever_or_a_provider_sdk():
    # evals.candidates imports build_retriever from this module at its own top level, so this
    # module — and evals.configs, which it also imports at its top level — staying free of
    # fineprint.retrieval (and everything that pulls in) is what keeps describing --config from
    # requiring Postgres, boto3 or anthropic to even be installed.
    program = (
        "import sys; import evals.run_golden_set; "
        "print('fineprint.retrieval' in sys.modules, 'boto3' in sys.modules, "
        "'anthropic' in sys.modules)"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )

    assert finished.stdout.strip() == "False False False"


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
    tracer = Tracer()
    seen: list[tuple[Settings, Tracer | None]] = []
    embedders: list[Tracer | None] = []

    def fake_get_reranker(settings: Settings, tracer: Tracer | None = None):
        seen.append((settings, tracer))
        return fake_reranker if config.reranker else None

    def fake_get_embedder(settings: Settings, tracer: Tracer | None = None):
        embedders.append(tracer)
        return FakeEmbedder()

    monkeypatch.setattr("fineprint.providers.factory.get_embedder", fake_get_embedder)
    monkeypatch.setattr("evals.run_golden_set.get_reranker", fake_get_reranker)

    settings = settings_for(config, Settings())
    retriever = build_retriever(settings, pool=object(), tracer=tracer)

    assert seen == [(settings, tracer)]
    assert embedders == [tracer]
    assert retriever.tracer is tracer
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
    """Replace the database, the embedder and the answer model so `main` can run offline.

    `wire(...)` returns the list `build_retriever` is called into: `main` calls it exactly once,
    with the `Settings` it built for this run, so a test can check what that run actually wired
    up (its chunk set, its re-ranker) without a real database or a real retriever.

    The tracer `main` builds is a `Tracer` over a `RecordingSink`, kept as `wired.sink`, and what
    `main` passes to `build_answer_function` is kept as `wired.answer_arguments`.
    """
    sink = RecordingSink()
    tracer = Tracer(sink)
    answer_arguments: list[dict] = []

    def wire(items, retriever, answer_fn=None):
        seen_settings: list[Settings] = []

        @contextmanager
        def fake_pool(database_url: str):
            yield object()

        def build_answer_function(settings, built_retriever, **arguments):
            if answer_fn is None:
                raise AssertionError("a --retrieval-only run must not build the answer model")
            answer_arguments.append(arguments)
            return answer_fn

        def fake_build_retriever(settings, pool, built_tracer=None):
            seen_settings.append(settings)
            assert built_tracer is tracer, "the retriever shares the run's tracer"
            return retriever

        monkeypatch.setattr("evals.run_golden_set.load_golden_set", lambda path: items)
        monkeypatch.setattr("evals.run_golden_set.connection_pool", fake_pool)
        monkeypatch.setattr(
            "evals.run_golden_set.corpus_from_database",
            lambda pool, edition: Corpus(edition=edition, sha256="d7a341bc3d2d"),
        )
        monkeypatch.setattr("evals.run_golden_set.get_tracer", lambda settings: tracer)
        monkeypatch.setattr("evals.run_golden_set.build_retriever", fake_build_retriever)
        monkeypatch.setattr("evals.run_golden_set.build_answer_function", build_answer_function)

        return seen_settings

    wire.sink = sink
    wire.tracer = tracer
    wire.answer_arguments = answer_arguments
    return wire


def test_the_command_refuses_a_configuration_it_does_not_know(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--config", "magic"])

    assert exit_info.value.code == 2
    assert "hybrid" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_main_wires_each_configurations_settings_and_searches_in_its_mode(name, wired, tmp_path):
    """For every configuration, `main` builds the retriever with the `Settings` that
    configuration means, and searches in the mode it names.

    This drives `main` end to end through the `wired` fixture, which is what actually exercises
    `--config`, `CONFIGS[name]` and `settings_for` together; `build_retriever` itself is replaced,
    so this does not also cover wiring `get_reranker` into a real `Retriever` — that is the unit
    test above.
    """
    config = CONFIGS[name]
    items = [golden("q001")]
    retriever = FakeRetriever({items[0].question: ON_PAGE_23})
    seen_settings = wired(items, retriever, answer_fn=lambda question, chunks: answered(question))

    exit_code = main(["--config", name, "--results-dir", str(tmp_path)])

    assert exit_code == 0
    (settings,) = seen_settings
    assert settings.chunk_set == config.chunk_set
    assert settings.reranker_provider == ("bedrock" if config.reranker else "none")
    if config.reranker:
        assert settings.reranker_model == config.reranker
    assert retriever.calls == [(items[0].question, config.mode, settings.retrieval_top_k)]


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
        "tools": False,
        "tools_today": None,
        "fallback_model": "us.anthropic.claude-sonnet-5",
        "tracing": "none",
        "langfuse_base_url": None,
        "agent_max_tool_turns": None,
        "only": None,
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


# --- tracing: one trace per question ------------------------------------------------


def priced_answer(tracer: Tracer):
    """An answer function that makes one generation call inside the question's trace."""

    def answer(question: str, chunks):
        with tracer.span("generation", "answer", model="us.anthropic.claude-opus-5") as span:
            span.end(
                input_tokens=1000,
                output_tokens=100,
                model="us.anthropic.claude-opus-5",
                cost=Cost(input_usd=0.005, output_usd=0.0025),
            )
        return answered(question)

    return answer


def test_each_question_is_a_trace_in_the_runs_session_tagged_with_its_configuration():
    sink = RecordingSink()
    items = [golden("q001"), golden("q002", question="What does Part D cover?")]
    retriever = FakeRetriever({items[0].question: ON_PAGE_23})
    tracer = Tracer(sink)

    run_questions(
        items,
        retriever,
        priced_answer(tracer),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
        tracer=tracer,
        run_id="20261010T100000Z_sections",
        config_name="sections",
    )

    first, second = sink.traces
    assert first.session_id == second.session_id == "20261010T100000Z_sections"
    assert first.tags == second.tags == ["eval", "sections"]
    assert (first.metadata, second.metadata) == (
        {"question_id": "q001"},
        {"question_id": "q002"},
    )
    assert first.input == items[0].question
    assert first.fields["output"] == answered(items[0].question).answer


def test_the_trace_record_is_stored_on_the_question_with_its_calls_and_cost():
    sink = RecordingSink(trace_id="trace-42")
    item = golden()
    tracer = Tracer(sink)

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ON_PAGE_23}),
        priced_answer(tracer),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
        tracer=tracer,
        run_id="r",
        config_name="hybrid",
    )

    assert isinstance(row.trace, TraceRecord)
    assert row.trace.trace_id == "trace-42"
    assert row.trace.name == "eval-question"
    assert [(call.kind, call.input_tokens, call.output_tokens) for call in row.trace.calls] == [
        ("generation", 1000, 100)
    ]
    assert row.trace.cost_usd == pytest.approx(0.0075)
    assert row.trace.latency_ms > 0


def test_a_question_that_failed_still_has_its_trace():
    sink = RecordingSink()
    item = golden()
    retriever = FakeRetriever({}, fails_on=item.question)

    (row,) = run_questions(
        [item],
        retriever,
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
        tracer=Tracer(sink),
        run_id="r",
        config_name="hybrid",
    )

    assert "retrieval failed" in row.error
    assert row.trace is not None and row.trace.trace_id == "trace-1"


def test_without_a_tracer_the_question_has_no_trace_and_nothing_is_traced():
    item = golden()

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: ON_PAGE_23}),
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
    )

    assert row.trace is None


# --- tracing: what the command records ---------------------------------------------


def test_the_command_stores_each_trace_in_the_file_and_closes_the_tracer(wired, tmp_path):
    items = [golden("q001")]
    wired(
        items,
        FakeRetriever({items[0].question: ON_PAGE_23}),
        answer_fn=priced_answer(wired.tracer),
    )

    exit_code = main(["--config", "sections", "--results-dir", str(tmp_path)])

    assert exit_code == 0
    (written,) = list(tmp_path.glob("*.json"))
    record = json.loads(written.read_text(encoding="utf-8"))
    (trace,) = [row["trace"] for row in record["questions"]]
    assert trace["trace_id"] == "trace-1"
    assert trace["cost_usd"] == pytest.approx(0.0075)
    (session,) = {root.session_id for root in wired.sink.traces}
    assert session == record["run_id"]
    assert wired.sink.traces[0].tags == ["eval", "sections"]
    assert wired.sink.flushes == 1 and wired.sink.shutdowns == 1


def test_the_tracer_is_closed_even_when_the_run_cannot_start(wired, tmp_path, monkeypatch):
    items = [golden("q001")]
    wired(items, FakeRetriever({}))

    def refuse(pool, edition):
        raise RuntimeError("not ingested")

    monkeypatch.setattr("evals.run_golden_set.corpus_from_database", refuse)

    with pytest.raises(RuntimeError):
        main(["--config", "sections", "--results-dir", str(tmp_path)])

    assert wired.sink.flushes == 1 and wired.sink.shutdowns == 1


def test_the_command_records_the_fallback_model_and_that_nothing_was_sent_to_langfuse(
    wired, tmp_path, monkeypatch
):
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "us.anthropic.claude-sonnet-5")
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "hybrid", "--results-dir", str(tmp_path)])

    (written,) = list(tmp_path.glob("*.json"))
    config = json.loads(written.read_text(encoding="utf-8"))["config"]
    assert config["fallback_model"] == "us.anthropic.claude-sonnet-5"
    assert config["tracing"] == "none"
    assert config["langfuse_base_url"] is None
    assert config["tools"] is False and config["tools_today"] is None


def test_a_run_without_a_fallback_model_records_none(wired, tmp_path, monkeypatch):
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "")
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "hybrid", "--results-dir", str(tmp_path)])

    (written,) = list(tmp_path.glob("*.json"))
    assert json.loads(written.read_text(encoding="utf-8"))["config"]["fallback_model"] is None


def test_a_retrieval_only_run_records_no_fallback_model(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}))

    main(["--config", "sections", "--retrieval-only", "--results-dir", str(tmp_path)])

    (written,) = list(tmp_path.glob("*.json"))
    assert json.loads(written.read_text(encoding="utf-8"))["config"]["fallback_model"] is None


def test_a_langfuse_run_records_the_provider_and_where_the_traces_went(
    wired, tmp_path, monkeypatch
):
    monkeypatch.setenv("TRACING_PROVIDER", "langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "hybrid", "--results-dir", str(tmp_path)])

    (written,) = list(tmp_path.glob("*.json"))
    config = json.loads(written.read_text(encoding="utf-8"))["config"]
    assert config["tracing"] == "langfuse"
    assert config["langfuse_base_url"] == "https://langfuse.example"
    assert "sk-test" not in written.read_text(encoding="utf-8")


def test_the_command_hands_the_tracer_to_the_answer_function(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "sections", "--results-dir", str(tmp_path)])

    (arguments,) = wired.answer_arguments
    assert arguments["tracer"] is wired.tracer
    assert arguments["tools"] is False


def test_the_agent_configuration_offers_the_tools_and_records_the_day_it_ran(
    wired, tmp_path, monkeypatch
):
    monkeypatch.setattr("evals.run_golden_set.run_date", lambda: date(2026, 10, 12))
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "agent", "--results-dir", str(tmp_path)])

    (arguments,) = wired.answer_arguments
    assert arguments["tools"] is True
    assert arguments["today"] == date(2026, 10, 12)
    (written,) = list(tmp_path.glob("*_agent.json"))
    config = json.loads(written.read_text(encoding="utf-8"))["config"]
    assert config["tools"] is True
    assert config["tools_today"] == "2026-10-12"
    assert config["chunk_set"] == "sections" and config["reranker"] == "cohere.rerank-v3-5:0"


def test_a_run_that_does_not_use_tools_records_no_date(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "sections", "--results-dir", str(tmp_path)])

    (written,) = list(tmp_path.glob("*.json"))
    assert json.loads(written.read_text(encoding="utf-8"))["config"]["tools_today"] is None


# --- the summary: cost, latency, fallbacks ------------------------------------------


def traced_row(
    question_id: str, *, cost: float | None, latency_ms: float, fallback: bool = False
) -> QuestionResult:
    calls = [
        CallRecord(
            kind="generation",
            name="answer",
            model="m",
            input_tokens=1,
            output_tokens=1,
            units=None,
            latency_ms=latency_ms,
            cost_usd=cost,
            error=None,
            fallback=fallback,
        )
    ]
    return QuestionResult(
        id=question_id,
        question="?",
        type="lookup",
        category="costs",
        difficulty="easy",
        answerable=True,
        expected_pages=[23],
        retrieved=[],
        metrics=QuestionMetrics(),
        trace=TraceRecord(
            trace_id=None, name="eval-question", latency_ms=latency_ms, cost_usd=cost, calls=calls
        ),
    )


def run_of(*rows: QuestionResult, fallback_model: str | None = None) -> RunResult:
    return RunResult(
        run_id="20261010T100000Z_sections",
        created_at="2026-10-10T10:00:00Z",
        git=GitState(commit=None, dirty=False),
        config=RunConfig(
            name="sections",
            mode="hybrid",
            chunk_set="sections",
            top_k=5,
            candidates=20,
            rrf_k=60,
            embedding_model="amazon.titan-embed-text-v2:0",
            llm_model="us.anthropic.claude-opus-5",
            fallback_model=fallback_model,
        ),
        corpus=Corpus(edition=2026, sha256="abc"),
        golden_set=GoldenSetInfo(sha256="def", count=len(rows)),
        questions=list(rows),
    )


def test_the_summary_reports_total_cost_cost_per_query_p95_and_fallbacks():
    rows = [
        traced_row("q001", cost=0.25, latency_ms=2000),
        traced_row("q002", cost=0.75, latency_ms=4000, fallback=True),
    ]

    lines = summary_lines(run_of(*rows, fallback_model="us.anthropic.claude-sonnet-5"))

    assert "  cost                 $1.0000 in all, $0.5000 a question" in lines
    assert "  p95 latency          4.0 s" in lines
    assert "  fallbacks            1" in lines
    assert "  answered by us.anthropic.claude-sonnet-5: q002" in lines


def test_the_summary_names_no_fallback_questions_when_there_were_none():
    lines = summary_lines(run_of(traced_row("q001", cost=0.1, latency_ms=1000)))

    assert "  fallbacks            0" in lines
    assert not any(line.startswith("  answered by") for line in lines)


def test_the_summary_shows_dashes_when_a_question_has_no_cost():
    lines = summary_lines(
        run_of(
            traced_row("q001", cost=0.1, latency_ms=1000),
            traced_row("q002", cost=None, latency_ms=1000),
        )
    )

    assert "  cost                 — in all, — a question" in lines
    assert "  p95 latency          —" in lines
    assert "  fallbacks            —" in lines


# --- the real answer function -------------------------------------------------------


def test_the_answer_function_traces_the_chat_model_and_joins_the_questions_trace(monkeypatch):
    """`build_answer_function` builds the chat model with the run's tracer and answers inside
    the question's own trace, so the response carries that trace's id."""
    item = golden()
    chat = FakeChatModel(
        DraftAnswer(
            answer="The standard Part B premium in 2026 is $202.90 a month.",
            found_in_handbook=True,
            citations=[],
            confidence="high",
        )
    )
    built: list[Tracer | None] = []

    def fake_get_chat_model(settings, tracer=None):
        built.append(tracer)
        return chat

    monkeypatch.setattr("fineprint.providers.factory.get_chat_model", fake_get_chat_model)
    sink = RecordingSink(trace_id="trace-7")
    tracer = Tracer(sink)
    answer_fn = build_answer_function(
        Settings(), FakeRetriever({}), tracer=tracer, today=date(2026, 10, 12)
    )

    with tracer.trace("eval-question"):
        response = answer_fn(item.question, [])

    assert built == [tracer]
    assert response.trace_id == "trace-7"
    assert [root.name for root in sink.traces] == ["eval-question"], "one trace, not two"


def test_the_tools_answer_function_answers_through_the_agent_and_stores_its_tool_calls(
    monkeypatch,
):
    """With `tools`, a question is answered by `answer_with_tools` on the run's date and tracer.
    What `_as_json` stores is what `evals.score` reads its tool contexts from."""
    hospital = RetrievedChunk(
        chunk_id=74,
        ordinal=73,
        section="Inpatient hospital care",
        page_start=27,
        page_end=27,
        text=(
            "Days 61–90: You pay $434 each day. Days 91–150: You pay $868 each day while using "
            "your 60 lifetime reserve days."
        ),
        score=0.0325,
        rank=1,
        lexical_rank=1,
        vector_rank=1,
        fused_rank=1,
        rerank_score=None,
    )
    lookup = ToolCall(
        id="toolu_lookup",
        name="lookup_cost_figure",
        arguments={"figure": "part_a_coinsurance_days_61_90"},
    )
    record = ToolCall(
        id="toolu_answer",
        name="record_answer",
        arguments={
            "answer": "On day 80 he pays $434 each day (page 27).",
            "found_in_handbook": True,
            "citations": [{"chunk_id": 74, "quote": "Days 61–90: You pay $434 each day."}],
            "confidence": "high",
        },
    )
    chat = ScriptedToolModel([[lookup], [record]])
    monkeypatch.setattr(
        "fineprint.providers.factory.get_chat_model", lambda settings, tracer=None: chat
    )
    sink = RecordingSink(trace_id="trace-9")
    tracer = Tracer(sink)
    settings = Settings(agent_max_tool_turns=3)
    answer_fn = build_answer_function(
        settings, FakeRetriever({}), tools=True, today=date(2026, 10, 12), tracer=tracer
    )
    item = golden("q023", question="What does he pay a day after 80 days in the hospital?")

    (row,) = run_questions(
        [item],
        FakeRetriever({item.question: [hospital]}),
        answer_fn,
        mode="hybrid",
        top_k=5,
        report=lambda line: None,
        tracer=tracer,
        run_id="r",
        config_name="agent",
    )

    assert row.error is None
    assert "Today's date: 2026-10-12." in chat.calls[0].messages[0]["content"]
    (tool_call,) = row.answer["tool_calls"]
    assert tool_call["name"] == "lookup_cost_figure" and tool_call["error"] is None
    assert tool_call["source_text"]
    assert row.answer["turns"] == 2
    assert tool_contexts(row) == [tool_call["source_text"]], "scoring reads the stored shape"
    assert row.answer["trace_id"] == "trace-9"
    (root,) = sink.traces
    assert root.name == "eval-question", "the agent joins the runner's trace"
    assert [child.kind for child in root.children] == ["agent"]
    assert row.trace is not None and ("agent", "tools") in [
        (call.kind, call.name) for call in row.trace.calls
    ]


# --- --only: a few questions as a smoke test ----------------------------------------


def three_golden_items():
    return [
        golden("q001"),
        golden("q006", question="What does Part D cover?"),
        golden("q009", question="What is a Medigap policy?"),
    ]


def test_only_runs_the_named_questions_in_golden_set_order(wired, tmp_path):
    items = three_golden_items()
    retriever = FakeRetriever({item.question: ON_PAGE_23 for item in items})
    wired(items, retriever, answer_fn=lambda q, c: answered(q))

    exit_code = main(
        ["--config", "sections", "--only", "q009,q001", "--results-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert [call[0] for call in retriever.calls] == [items[0].question, items[2].question]
    (written,) = list(tmp_path.glob("*.json"))
    record = json.loads(written.read_text(encoding="utf-8"))
    assert [row["id"] for row in record["questions"]] == ["q001", "q009"]
    assert record["config"]["only"] == ["q001", "q009"]
    assert record["golden_set"]["count"] == 3, "the golden set is still the whole file"


def test_a_full_run_records_no_subset(wired, tmp_path):
    items = three_golden_items()
    wired(items, FakeRetriever({}), answer_fn=lambda q, c: answered(q))

    main(["--config", "sections", "--results-dir", str(tmp_path)])

    (written,) = list(tmp_path.glob("*.json"))
    assert json.loads(written.read_text(encoding="utf-8"))["config"]["only"] is None


def test_only_refuses_an_id_the_golden_set_does_not_have(wired, tmp_path, capsys):
    items = three_golden_items()
    retriever = FakeRetriever({})
    wired(items, retriever, answer_fn=lambda q, c: answered(q))

    exit_code = main(
        ["--config", "sections", "--only", "q001,q999", "--results-dir", str(tmp_path)]
    )

    assert exit_code == 1
    assert "q999" in capsys.readouterr().err
    assert retriever.calls == [] and list(tmp_path.glob("*.json")) == []


def test_only_refuses_an_empty_list(wired, tmp_path, capsys):
    wired(three_golden_items(), FakeRetriever({}))

    assert main(["--config", "sections", "--only", " , ", "--results-dir", str(tmp_path)]) == 1
    assert "at least one question id" in capsys.readouterr().err


@pytest.mark.parametrize("spelling", ["default", "dotted"])
def test_only_refuses_the_default_results_directory(spelling, wired, capsys):
    from evals.results import RESULTS_DIR

    items = three_golden_items()
    retriever = FakeRetriever({})
    wired(items, retriever, answer_fn=lambda q, c: answered(q))
    directory = RESULTS_DIR if spelling == "default" else RESULTS_DIR / ".." / "results"
    before = sorted(RESULTS_DIR.glob("*.json"))

    exit_code = main(["--config", "sections", "--only", "q001", "--results-dir", str(directory)])

    assert exit_code == 1
    assert "other than evals/results" in capsys.readouterr().err
    assert retriever.calls == []
    assert sorted(RESULTS_DIR.glob("*.json")) == before, "nothing was written"


def test_only_without_a_results_dir_is_refused(wired, capsys):
    wired(three_golden_items(), FakeRetriever({}))

    assert main(["--config", "sections", "--only", "q001"]) == 1
    assert "other than evals/results" in capsys.readouterr().err


# --- a run that would be spoiled is refused before any model call ------------------


def refused_run(wired, tmp_path, *arguments) -> tuple[int, FakeRetriever]:
    items = [golden("q001")]
    retriever = FakeRetriever({items[0].question: ON_PAGE_23})
    wired(items, retriever, answer_fn=lambda q, c: answered(q))
    exit_code = main([*arguments, "--results-dir", str(tmp_path)])
    return exit_code, retriever


@pytest.mark.parametrize("fault", ["chat:timeout", "embedding:unavailable:once", "rerank:timeout"])
def test_a_run_with_a_fault_switch_set_exits_2_before_any_search(
    fault, wired, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("FAULT_INJECTION", fault)

    exit_code, retriever = refused_run(wired, tmp_path, "--config", "sections")

    assert exit_code == 2
    assert f"FAULT_INJECTION={fault}" in capsys.readouterr().err
    assert retriever.calls == [] and list(tmp_path.glob("*.json")) == []


def test_the_fault_switch_is_refused_with_only_too(wired, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAULT_INJECTION", "chat:timeout")

    exit_code, retriever = refused_run(wired, tmp_path, "--config", "sections", "--only", "q001")

    assert exit_code == 2
    assert "FAULT_INJECTION" in capsys.readouterr().err
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("variable", "arguments"),
    [
        ("LLM_MODEL", ["--config", "sections"]),
        ("FALLBACK_LLM_MODEL", ["--config", "sections"]),
        ("EMBEDDING_MODEL", ["--config", "sections", "--retrieval-only"]),
    ],
)
def test_a_model_with_no_price_exits_2_naming_the_variable_and_the_model(
    variable, arguments, wired, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv(variable, "some.unpriced-model")

    exit_code, retriever = refused_run(wired, tmp_path, *arguments)

    assert exit_code == 2
    error = capsys.readouterr().err
    assert variable in error and "some.unpriced-model" in error and "PRICES" in error
    assert retriever.calls == []


def test_an_unpriced_reranker_is_refused_only_when_the_configuration_reranks(
    wired, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("RERANKER_MODEL", "some.unpriced-reranker")

    exit_code, retriever = refused_run(wired, tmp_path, "--config", "sections+rerank")
    assert exit_code == 0, "settings_for pins the configuration's own re-ranker model"

    # A configuration whose re-ranker is not a priced model is refused.
    monkeypatch.setitem(
        CONFIGS, "odd", replace(CONFIGS["sections+rerank"], name="odd", reranker="some.unpriced")
    )
    exit_code, _ = refused_run(wired, tmp_path, "--config", "odd")
    assert exit_code == 2
    assert "RERANKER_MODEL=some.unpriced" in capsys.readouterr().err


def test_answer_models_need_no_price_on_a_retrieval_only_run(wired, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some.unpriced-model")
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "some.unpriced-fallback")
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}))

    assert main(["--config", "sections", "--retrieval-only", "--results-dir", str(tmp_path)]) == 0


def test_an_empty_fallback_model_needs_no_price(wired, tmp_path, monkeypatch):
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "")

    exit_code, _ = refused_run(wired, tmp_path, "--config", "sections")

    assert exit_code == 0


def test_the_agent_turn_limit_is_recorded_for_tool_configurations_only(wired, tmp_path):
    items = [golden("q001")]
    wired(items, FakeRetriever({items[0].question: ON_PAGE_23}), answer_fn=lambda q, c: answered(q))

    main(["--config", "agent", "--results-dir", str(tmp_path)])
    main(["--config", "sections", "--results-dir", str(tmp_path)])

    configs = {
        path.name.split("_", 1)[1]: json.loads(path.read_text(encoding="utf-8"))["config"]
        for path in tmp_path.glob("*.json")
    }
    assert configs["agent.json"]["agent_max_tool_turns"] == 4
    assert configs["sections.json"]["agent_max_tool_turns"] is None


# --- a re-rank that failed is counted, not hidden -----------------------------------


def reranked_trace(*, failed: bool) -> TraceRecord:
    call = CallRecord(
        kind="rerank",
        name="rerank",
        model="cohere.rerank-v3-5:0",
        input_tokens=None,
        output_tokens=None,
        units=1,
        latency_ms=50.0,
        cost_usd=0.0 if failed else 0.002,
        error="throttled" if failed else None,
    )
    return TraceRecord(
        trace_id=None, name="eval-question", latency_ms=1000.0, cost_usd=call.cost_usd, calls=[call]
    )


def test_the_summary_names_the_questions_whose_rerank_failed():
    rows = [traced_row(f"q{n:03d}", cost=0.1, latency_ms=1000) for n in (1, 2, 17)]
    for row, failed in zip(rows, (True, False, True), strict=True):
        row.trace = reranked_trace(failed=failed)

    lines = summary_lines(run_of(*rows))

    assert "  re-rank skipped: q001, q017" in lines


def test_the_summary_says_nothing_about_reranks_when_none_failed():
    rows = [traced_row("q001", cost=0.1, latency_ms=1000)]
    rows[0].trace = reranked_trace(failed=False)

    assert not any("re-rank" in line for line in summary_lines(run_of(*rows)))


def test_a_failed_rerank_does_not_make_the_question_a_fallback_answer():
    row = traced_row("q001", cost=0.1, latency_ms=1000)
    row.trace = reranked_trace(failed=True)

    lines = summary_lines(run_of(row))

    assert "  fallbacks            0" in lines


def test_the_command_exits_1_and_says_why_when_a_rerank_failed(wired, tmp_path, capsys):
    from fineprint.providers.base import ProviderUnavailableError

    items = [golden("q001"), golden("q002", question="What does Part D cover?")]

    def answer_fn(question: str, chunks):
        with wired.tracer.span("rerank", "rerank", model="cohere.rerank-v3-5:0") as span:
            failed = question == items[0].question
            span.end(units=1, error=ProviderUnavailableError("throttled") if failed else None)
        return answered(question)

    retriever = FakeRetriever({item.question: ON_PAGE_23 for item in items})
    wired(items, retriever, answer_fn=answer_fn)

    exit_code = main(["--config", "sections+rerank", "--results-dir", str(tmp_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "re-rank skipped: q001" in captured.out
    assert "re-rank failed" in captured.err
    assert len(list(tmp_path.glob("*.json"))) == 1, "the file is written all the same"
