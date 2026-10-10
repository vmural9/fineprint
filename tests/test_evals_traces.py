"""Tests for `python -m evals.traces verify`, driven by a fake Langfuse API and no network."""

from types import SimpleNamespace

import pytest

from evals import traces
from evals.metrics import QuestionMetrics
from evals.results import (
    Corpus,
    GitState,
    GoldenSetInfo,
    QuestionResult,
    RunConfig,
    RunResult,
    load,
    results_path,
    save,
)
from fineprint.config import Settings
from fineprint.tracing import CallRecord, TraceRecord


def call(kind, input_tokens, output_tokens, cost, **extra) -> CallRecord:
    return CallRecord(
        kind=kind,
        name=extra.pop("name", kind),
        model=extra.pop("model", None),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        units=extra.pop("units", None),
        latency_ms=10.0,
        cost_usd=cost,
        error=None,
        **extra,
    )


def trace_record(trace_id: str | None, *, cost: float | None = 0.0123) -> TraceRecord:
    """A question that embedded (12 tokens), re-ranked (1 unit) and generated (500 in, 40 out)."""
    return TraceRecord(
        trace_id=trace_id,
        name="eval-question",
        latency_ms=2500.0,
        cost_usd=cost,
        calls=[
            call("retriever", None, None, None),
            call("embedding", 12, None, 0.0000002),
            call("rerank", None, None, 0.0020, units=1),
            call("generation", 500, 40, 0.0102998),
        ],
    )


def question(question_id: str, record: TraceRecord | None) -> QuestionResult:
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
        trace=record,
    )


def results_file(tmp_path, *rows: QuestionResult):
    result = RunResult(
        run_id="20261010T100000Z_agent",
        created_at="2026-10-10T10:00:00Z",
        git=GitState(commit=None, dirty=False),
        config=RunConfig(
            name="agent",
            mode="hybrid",
            chunk_set="sections",
            top_k=5,
            candidates=20,
            rrf_k=60,
            embedding_model="amazon.titan-embed-text-v2:0",
            llm_model="us.anthropic.claude-opus-5",
        ),
        corpus=Corpus(edition=2026, sha256="abc"),
        golden_set=GoldenSetInfo(sha256="def", count=len(rows)),
        questions=list(rows),
    )
    return save(result, results_path(result.run_id, tmp_path))


def observation(type_, input_tokens=None, output_tokens=None, cost=None) -> SimpleNamespace:
    """One observation as the SDK's `ObservationV2` exposes the fields the check reads."""
    usage = None
    if input_tokens is not None or output_tokens is not None:
        usage = {"input": input_tokens or 0, "output": output_tokens or 0}
    return SimpleNamespace(
        type=type_, usage_details=usage, cost_details=None, total_cost=cost, trace_id="t"
    )


def matching() -> list[SimpleNamespace]:
    """What Langfuse holds for `trace_record`: the same spans, as it types them."""
    return [
        observation("SPAN"),
        observation("RETRIEVER"),
        observation("EMBEDDING", 12, 0, 0.0000002),
        SimpleNamespace(  # the re-rank is a generation with search units, not tokens
            type="GENERATION",
            usage_details={"search_units": 1},
            cost_details=None,
            total_cost=0.0020,
            trace_id="t",
        ),
        observation("GENERATION", 500, 40, 0.0102998),
    ]


class FakeApi:
    """The part of `langfuse.api` this command uses: `observations.get_many`."""

    def __init__(self, by_trace: dict[str, list], page_size: int | None = None) -> None:
        self.by_trace = by_trace
        self.page_size = page_size
        self.calls: list[dict] = []
        self.observations = SimpleNamespace(get_many=self.get_many)

    def get_many(self, **arguments):
        self.calls.append(arguments)
        every = self.by_trace.get(arguments["trace_id"], [])
        start = int(arguments["cursor"] or 0)
        size = self.page_size or len(every) or 1
        page = every[start : start + size]
        more = str(start + size) if start + size < len(every) else None
        return SimpleNamespace(data=page, meta=SimpleNamespace(cursor=more))


def verified(tmp_path, api, *rows, **options):
    return traces.verify(load(results_file(tmp_path, *rows)), api, **options)


# --- what is compared ---------------------------------------------------------------


def test_a_trace_that_matches_is_checked_and_reports_nothing(tmp_path):
    api = FakeApi({"t1": matching()})

    outcome = verified(tmp_path, api, question("q001", trace_record("t1")))

    assert outcome.checked == 1
    assert outcome.problems == []


def test_the_request_names_the_trace_and_the_usage_fields(tmp_path):
    api = FakeApi({"t1": matching()})

    verified(tmp_path, api, question("q001", trace_record("t1")))

    (request,) = api.calls
    assert request["trace_id"] == "t1"
    assert "usage" in request["fields"]
    assert request["limit"] == traces.PAGE_SIZE


def test_every_page_of_a_long_trace_is_read(tmp_path):
    api = FakeApi({"t1": matching()}, page_size=2)

    outcome = verified(tmp_path, api, question("q001", trace_record("t1")))

    assert len(api.calls) == 3
    assert outcome.problems == []


def test_a_token_mismatch_names_the_question_the_trace_and_both_numbers(tmp_path):
    seen = matching()
    seen[-1] = observation("GENERATION", 499, 40, 0.0102998)
    api = FakeApi({"t1": seen})

    outcome = verified(tmp_path, api, question("q001", trace_record("t1")))

    assert outcome.problems == ["q001  trace t1  input tokens: stored 512, Langfuse 511"]


def test_output_tokens_are_compared_too(tmp_path):
    seen = matching()
    seen[-1] = observation("GENERATION", 500, 41, 0.0102998)
    api = FakeApi({"t1": seen})

    outcome = verified(tmp_path, api, question("q001", trace_record("t1")))

    assert outcome.problems == ["q001  trace t1  output tokens: stored 40, Langfuse 41"]


def test_a_cost_mismatch_beyond_a_ten_thousandth_of_a_dollar_is_reported(tmp_path):
    seen = matching()
    seen[-1] = observation("GENERATION", 500, 40, 0.0102998 + 0.00011)
    api = FakeApi({"t1": seen})

    outcome = verified(tmp_path, api, question("q001", trace_record("t1")))

    assert outcome.problems == ["q001  trace t1  cost: stored $0.012300, Langfuse $0.012410"]


def test_a_cost_within_a_ten_thousandth_of_a_dollar_matches(tmp_path):
    seen = matching()
    seen[-1] = observation("GENERATION", 500, 40, 0.0102998 + 0.00009)
    api = FakeApi({"t1": seen})

    assert verified(tmp_path, api, question("q001", trace_record("t1"))).problems == []


def test_the_cost_detail_stands_in_when_langfuse_gives_no_total_cost(tmp_path):
    seen = matching()
    seen[-1] = SimpleNamespace(
        type="GENERATION",
        usage_details={"input": 500, "output": 40},
        cost_details={"input": 0.01, "output": 0.0002998, "total": 0.0102998},
        total_cost=None,
        trace_id="t",
    )
    api = FakeApi({"t1": seen})

    assert verified(tmp_path, api, question("q001", trace_record("t1"))).problems == []


def test_a_trace_langfuse_does_not_have_is_a_problem(tmp_path):
    api = FakeApi({})

    outcome = verified(tmp_path, api, question("q001", trace_record("t1")))

    assert outcome.problems == ["q001  trace t1  Langfuse has no such trace"]


def test_a_missing_trace_is_asked_for_again_and_a_late_one_counts_as_found(tmp_path):
    api = FakeApi({})
    pauses: list[float] = []

    def sleep(seconds: float) -> None:
        pauses.append(seconds)
        api.by_trace["t1"] = matching()  # the trace arrives while we wait

    outcome = verified(
        tmp_path, api, question("q001", trace_record("t1")), attempts=3, interval=5, sleep=sleep
    )

    assert outcome.problems == []
    assert pauses == [5]


def test_a_trace_that_never_arrives_is_reported_after_every_attempt(tmp_path):
    api = FakeApi({})
    pauses: list[float] = []

    outcome = verified(
        tmp_path,
        api,
        question("q001", trace_record("t1")),
        attempts=3,
        interval=5,
        sleep=pauses.append,
    )

    assert len(outcome.problems) == 1
    assert pauses == [5, 5]
    assert len(api.calls) == 3


def test_a_question_that_matched_is_not_asked_about_again(tmp_path):
    api = FakeApi({"t1": matching()})

    verified(
        tmp_path,
        api,
        question("q001", trace_record("t1")),
        question("q002", trace_record("t2")),
        attempts=2,
        interval=0,
        sleep=lambda seconds: None,
    )

    assert [request["trace_id"] for request in api.calls] == ["t1", "t2", "t2"]


def test_a_question_without_a_trace_id_is_not_looked_up(tmp_path):
    api = FakeApi({"t1": matching()})

    outcome = verified(
        tmp_path,
        api,
        question("q001", trace_record("t1")),
        question("q002", trace_record(None)),
        question("q003", None),
    )

    assert outcome.checked == 1
    assert len(api.calls) == 1


def test_an_unknown_stored_cost_is_compared_on_tokens_only(tmp_path):
    api = FakeApi({"t1": matching()})

    outcome = verified(tmp_path, api, question("q001", trace_record("t1", cost=None)))

    assert outcome.problems == []
    assert outcome.without_cost == 1
    assert "1 with an unknown stored cost" in traces.summary(outcome, "run")


# --- the command ----------------------------------------------------------------------


class FakeClient:
    def __init__(self, api: FakeApi) -> None:
        self.api = api
        self.closed = False

    def shutdown(self) -> None:
        self.closed = True


@pytest.fixture
def langfuse_api(monkeypatch):
    """Replace the Langfuse client with one that serves `api`; returns the client it built."""

    def serve(api: FakeApi) -> FakeClient:
        client = FakeClient(api)
        monkeypatch.setattr("evals.traces.build_client", lambda settings: client)
        return client

    return serve


def test_the_command_exits_0_and_summarises_when_everything_matches(tmp_path, langfuse_api, capsys):
    path = results_file(tmp_path, question("q001", trace_record("t1")))
    client = langfuse_api(FakeApi({"t1": matching()}))

    exit_code = traces.main(["verify", str(path), "--wait", "0"])

    assert exit_code == 0
    assert "1 traces checked against Langfuse, all match" in capsys.readouterr().out
    assert client.closed


def test_the_command_exits_1_and_prints_a_line_per_mismatch(tmp_path, langfuse_api, capsys):
    seen = matching()
    seen[-1] = observation("GENERATION", 1, 40, 0.0102998)
    path = results_file(
        tmp_path, question("q001", trace_record("t1")), question("q002", trace_record("t2"))
    )
    langfuse_api(FakeApi({"t1": seen, "t2": matching()}))

    exit_code = traces.main(["verify", str(path), "--wait", "0"])

    assert exit_code == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "q001  trace t1  input tokens: stored 512, Langfuse 13"
    assert lines[-1].endswith("2 traces checked against Langfuse, 1 do not match")


def test_the_command_exits_1_for_a_trace_langfuse_does_not_have(tmp_path, langfuse_api, capsys):
    path = results_file(tmp_path, question("q001", trace_record("t1")))
    langfuse_api(FakeApi({}))

    assert traces.main(["verify", str(path), "--wait", "0"]) == 1
    assert "Langfuse has no such trace" in capsys.readouterr().out


def test_missing_langfuse_settings_exit_1_naming_each_one(tmp_path, monkeypatch, capsys):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        traces, "Settings", lambda: Settings(_env_file=None)
    )  # no .env on the developer's machine
    path = results_file(tmp_path, question("q001", trace_record("t1")))

    exit_code = traces.main(["verify", str(path)])

    assert exit_code == 1
    error = capsys.readouterr().err
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        assert name in error


def test_a_file_with_no_trace_ids_exits_1_before_asking_for_credentials(tmp_path, capsys):
    path = results_file(tmp_path, question("q001", trace_record(None)), question("q002", None))

    exit_code = traces.main(["verify", str(path)])

    assert exit_code == 1
    assert "no trace id" in capsys.readouterr().err


def test_a_file_that_cannot_be_read_exits_2(tmp_path, capsys):
    exit_code = traces.main(["verify", str(tmp_path / "missing.json")])

    assert exit_code == 2
    assert "missing.json" in capsys.readouterr().err


def test_the_client_is_built_from_the_three_settings(monkeypatch):
    built: list[dict] = []
    monkeypatch.setattr(traces, "Langfuse", lambda **arguments: built.append(arguments))
    settings = Settings(
        _env_file=None,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url="https://langfuse.example",
    )

    traces.build_client(settings)

    assert built == [
        {
            "public_key": "pk-test",
            "secret_key": "sk-test",
            "base_url": "https://langfuse.example",
        }
    ]
