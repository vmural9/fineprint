"""Tests for the scoring pass, `python -m evals.score`, all of them offline.

`ScriptedChatModel` plays the judge and hands back the verdicts each test wrote; `FakeEmbedder`
stands in for Titan. The questions and reference answers are the golden set's own, and the
passages are its evidence quotes, so every figure below was copied from the handbook.
"""

import re
import shutil
from pathlib import Path

import pytest

from evals.golden import DEFAULT_GOLDEN_SET, load_golden_set
from evals.metrics import QuestionMetrics
from evals.ragas_metrics import (
    PROMPTS_SHA256,
    AnswerClaims,
    ClaimVerdict,
    ClaimVerdicts,
    GeneratedQuestions,
    PassageVerdict,
    PassageVerdicts,
    SentenceVerdict,
    SentenceVerdicts,
    numbered_passages,
    split_sentences,
)
from evals.results import (
    Corpus,
    GitState,
    GoldenSetInfo,
    QuestionResult,
    RetrievedChunk,
    RunConfig,
    RunResult,
    load,
    results_path,
    save,
    sha256_of,
)
from evals.score import build_judge, main
from fineprint.config import Settings
from fineprint.providers.bedrock_chat import BedrockChatModel
from tests.fakes import FakeEmbedder, ScriptedChatModel

GOLDEN = {item.id: item for item in load_golden_set()}
Q001 = GOLDEN["q001"]  # lookup: the Part B premium, and what can make it higher
Q002 = GOLDEN["q002"]  # multi_section: a hospital stay, then a skilled nursing facility
Q003 = GOLDEN["q003"]  # unanswerable: the 2026 Extra Help limits

# The passages: the golden set's evidence quotes, word for word as the handbook prints them.
PREMIUM, IRMAA = (evidence.quote for evidence in Q001.evidence)  # both on page 23
DEDUCTIBLE = Q002.evidence[0].quote  # page 27
SNF_DAYS = Q002.evidence[3].quote  # page 29
EXTRA_HELP = Q003.evidence[1].quote  # page 92

# The claims the judge finds in q001's answer, which is the premium passage itself.
PREMIUM_CLAIMS = split_sentences(PREMIUM)

FOUR = ("context_recall", "context_precision", "faithfulness", "answer_relevance")
DEFAULT_JUDGE = Settings.model_fields["judge_model"].default
TITAN = "amazon.titan-embed-text-v2:0"

FIXTURES = Path(__file__).parent / "fixtures" / "results"
COMMITTED_RESULTS = Path(__file__).parent.parent / "evals" / "results"
# The three runs part 1 committed. They predate chunk text in results files.
PART_1_RUNS = (
    "20260921T011412Z_lexical-only.json",
    "20260921T011428Z_vector-only.json",
    "20260921T012044Z_hybrid.json",
)


# --- a results file to score -------------------------------------------------------


def chunk(rank: int, text: str, page: int) -> RetrievedChunk:
    """One retrieved chunk as a part 2 run records it: with its text."""
    return RetrievedChunk(
        chunk_id=1900 + rank,
        page_start=page,
        page_end=page,
        rank=rank,
        score=0.03,
        lexical_rank=rank,
        vector_rank=rank,
        ordinal=100 + rank,
        text=text,
        fused_rank=rank,
    )


def row(item, retrieved: list[RetrievedChunk], *, answer: str | None = None) -> QuestionResult:
    """One question's row, answered with `answer` when one is given."""
    return QuestionResult(
        id=item.id,
        question=item.question,
        type=item.type,
        category=item.category,
        difficulty=item.difficulty,
        answerable=item.answerable,
        expected_pages=list(item.expected_pages),
        retrieved=retrieved,
        metrics=QuestionMetrics(page_hit=1.0, page_recall=1.0, reciprocal_rank=1.0)
        if item.answerable
        else QuestionMetrics(),
        answer=None
        if answer is None
        else {
            "question": item.question,
            "answer": answer,
            "found_in_handbook": item.answerable,
            "citations": [],
            "dropped_citations": 0,
            "confidence": "high",
        },
    )


def results_file(directory: Path, *rows: QuestionResult, golden_sha256: str | None = None) -> Path:
    """Save a run holding `rows`, made against today's golden set unless told otherwise."""
    result = RunResult(
        run_id="20260926T101500Z_hybrid",
        created_at="2026-09-26T10:15:00Z",
        git=GitState(commit="0123456789abcdef0123456789abcdef01234567", dirty=False),
        config=RunConfig(
            name="hybrid",
            mode="hybrid",
            chunk_set="fixed-220w",
            top_k=5,
            candidates=20,
            rrf_k=60,
            embedding_model=TITAN,
            llm_model="us.anthropic.claude-opus-5",
        ),
        corpus=Corpus(
            edition=2026, sha256="d7a341bc3d2d3dab59af746a0875752761e6c2f2dc107d95e9078a23933513d6"
        ),
        golden_set=GoldenSetInfo(
            sha256=golden_sha256 or sha256_of(DEFAULT_GOLDEN_SET), count=len(GOLDEN)
        ),
        questions=list(rows),
    )
    return save(result, results_path(result.run_id, directory))


def three_questions(directory: Path) -> Path:
    """A run with one question of each kind: answered, not answered, and unanswerable.

    q001's chunks are stored out of rank order on purpose; the judge must still read them best
    ranked first.
    """
    return results_file(
        directory,
        row(Q001, [chunk(2, IRMAA, 23), chunk(1, PREMIUM, 23)], answer=PREMIUM),
        row(Q002, [chunk(1, DEDUCTIBLE, 27), chunk(2, SNF_DAYS, 29)]),
        row(Q003, [chunk(1, EXTRA_HELP, 92)], answer=split_sentences(Q003.expected_answer)[0]),
    )


# --- the judge's replies -----------------------------------------------------------


def recall(item, *supported: bool) -> SentenceVerdicts:
    """The recall judge's reply: one verdict per sentence of `item`'s reference answer."""
    assert len(supported) == len(split_sentences(item.expected_answer)), "one per sentence"
    return SentenceVerdicts(
        verdicts=[
            SentenceVerdict(reason="passage 1" if ok else "in no passage", supported=ok)
            for ok in supported
        ]
    )


def precision(*useful: bool) -> PassageVerdicts:
    """The precision judge's reply: one verdict per passage, in rank order."""
    return PassageVerdicts(
        verdicts=[
            PassageVerdict(reason="gives the amount" if ok else "gives nothing used", useful=ok)
            for ok in useful
        ]
    )


def checked(*supported: bool) -> ClaimVerdicts:
    """The faithfulness judge's second reply: one verdict per claim."""
    return ClaimVerdicts(
        verdicts=[
            ClaimVerdict(reason="passage 1" if ok else "no passage says so", supported=ok)
            for ok in supported
        ]
    )


def full_script() -> dict:
    """Every reply a pass over `three_questions` asks the judge for, q001's first.

    q001: sentences 2 of 3 supported, the passage at rank 1 useful and the one at rank 2 not,
    one of its two claims supported, and each question written back the very one asked.
    q002, which has no answer: sentences 2 of 4 supported, only the passage at rank 2 useful.
    """
    return {
        SentenceVerdicts: [recall(Q001, True, True, False), recall(Q002, True, False, False, True)],
        PassageVerdicts: [precision(True, False), precision(False, True)],
        AnswerClaims: [AnswerClaims(claims=PREMIUM_CLAIMS)],
        ClaimVerdicts: [checked(True, False)],
        GeneratedQuestions: [GeneratedQuestions(questions=[Q001.question] * 3, noncommittal=False)],
    }


def first_question_only(script: dict) -> dict:
    """The replies for q001 alone: the first of each kind."""
    return {schema: replies[:1] for schema, replies in script.items()}


def judge(script: dict | None = None) -> ScriptedChatModel:
    """A judge that charges 1,000 input and 100 output tokens a call."""
    return ScriptedChatModel(
        full_script() if script is None else script, input_tokens=1000, output_tokens=100
    )


class InterruptedJudge(ScriptedChatModel):
    """A judge stopped, as by Ctrl-C, the first time its script has nothing left to give."""

    def complete_structured(self, system, user, schema):
        if not self.script.get(schema):
            raise KeyboardInterrupt
        return super().complete_structured(system, user, schema)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Run `main` offline, with the judge a test hands to `wire` and `FakeEmbedder`.

    The settings are the defaults: no `.env`, and no judge or embedding model from the shell.
    `wire(judge)` returns the list of model ids `build_judge` was asked for, so a test can see
    which judge was built; `wire(None)` makes building one fail the test.
    """
    for name in ("JUDGE_MODEL", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    def wire(judge):
        built: list[str] = []

        def build_judge(settings, model):
            built.append(model)
            if judge is None:
                raise AssertionError("this command must not build a judge")
            return judge

        monkeypatch.setattr("evals.score.build_judge", build_judge)
        monkeypatch.setattr("evals.score.build_embedder", lambda settings: FakeEmbedder())
        return built

    return wire


def flat(text: str) -> str:
    """`text` with every run of whitespace made one space, so wrapped lines read as one."""
    return " ".join(text.split())


# --- a full pass -------------------------------------------------------------------


def test_each_question_is_scored_on_the_metrics_that_apply_to_it(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())

    exit_code = main([str(path)])

    assert exit_code == 0
    answered, unanswered, unanswerable = load(path).questions
    assert answered.metrics.context_recall == pytest.approx(2 / 3)
    assert answered.metrics.context_precision == 1.0  # its one useful passage ranked first
    assert answered.metrics.faithfulness == 0.5
    assert answered.metrics.answer_relevance == pytest.approx(1.0)
    assert answered.metrics.page_hit == 1.0, "the run's own metrics are left alone"
    # No answer: only the two metrics that judge retrieval.
    assert unanswered.metrics.context_recall == 0.5
    assert unanswered.metrics.context_precision == 0.5  # useful at rank 2 only: (1/2) / 1
    assert unanswered.metrics.faithfulness is None
    assert unanswered.metrics.answer_relevance is None
    # Unanswerable: none of the four, even though the run answered it.
    assert all(getattr(unanswerable.metrics, name) is None for name in FOUR)
    assert unanswerable.scoring_detail == {"skipped": "unanswerable"}


def test_each_question_keeps_the_verdicts_behind_its_scores_and_their_tokens(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())

    main([str(path)])

    answered, unanswered, _ = load(path).questions
    detail = answered.scoring_detail
    assert set(detail) == {*FOUR, "judge_input_tokens", "judge_output_tokens"}
    sentences = detail["context_recall"]["sentences"]
    assert [entry["sentence"] for entry in sentences] == split_sentences(Q001.expected_answer)
    assert [entry["supported"] for entry in sentences] == [True, True, False]
    assert [entry["useful"] for entry in detail["context_precision"]["passages"]] == [True, False]
    assert [entry["claim"] for entry in detail["faithfulness"]["claims"]] == PREMIUM_CLAIMS
    assert detail["answer_relevance"]["noncommittal"] is False
    # Five judge calls: recall, precision, two for faithfulness, and relevance.
    assert (detail["judge_input_tokens"], detail["judge_output_tokens"]) == (5000, 500)
    assert set(unanswered.scoring_detail) == {
        "context_recall",
        "context_precision",
        "judge_input_tokens",
        "judge_output_tokens",
    }
    assert unanswered.scoring_detail["judge_input_tokens"] == 2000


def test_the_scoring_block_names_the_judge_the_embedder_and_the_prompts(wired, tmp_path):
    path = three_questions(tmp_path)
    built = wired(judge())

    main([str(path)])

    scoring = load(path).scoring
    assert built == [DEFAULT_JUDGE], "the judge is the JUDGE_MODEL setting unless told otherwise"
    assert scoring.judge_model == DEFAULT_JUDGE
    assert scoring.embedding_model == TITAN
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", scoring.scored_at)


def test_the_file_records_the_sha256_of_the_prompts_that_judged_it(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())

    main([str(path)])

    assert load(path).scoring.prompts_sha256 == PROMPTS_SHA256


def test_the_scoring_block_totals_the_judge_tokens_of_every_question(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())

    main([str(path)])

    scoring = load(path).scoring
    # q001 took five calls and q002 two; q003 took none.
    assert (scoring.judge_input_tokens, scoring.judge_output_tokens) == (7000, 700)


def test_the_judge_reads_the_files_own_chunk_text_best_ranked_first(wired, tmp_path):
    path = three_questions(tmp_path)
    scripted = judge()
    wired(scripted)

    main([str(path)])

    recall_request = scripted.calls[0].user
    assert numbered_passages([PREMIUM, IRMAA]) in recall_request


def test_judge_model_picks_the_judge_and_the_file_records_it(wired, tmp_path):
    path = three_questions(tmp_path)
    built = wired(judge())

    main([str(path), "--judge-model", "us.anthropic.claude-opus-5"])

    assert built == ["us.anthropic.claude-opus-5"]
    assert load(path).scoring.judge_model == "us.anthropic.claude-opus-5"


def test_the_summary_gives_the_counts_the_four_means_and_the_judge_tokens(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    wired(judge())

    main([str(path)])

    printed = capsys.readouterr().out
    assert "3 questions, 2 scored, 1 skipped, 0 failed" in printed
    summary = printed.split("3 questions, 2 scored")[1]
    assert re.search(r"context_recall +0\.58\n", summary)  # the mean of 2/3 and 1/2
    assert re.search(r"context_precision +0\.75\n", summary)  # the mean of 1 and 1/2
    assert re.search(r"faithfulness +0\.50\n", summary)  # q001's alone: q002 has no answer
    assert re.search(r"answer_relevance +1\.00\n", summary)
    assert "7,000 in, 700 out" in summary
    assert str(path) in summary


def test_a_question_whose_retrieval_failed_in_the_run_is_skipped(wired, tmp_path):
    broken = row(Q001, [])
    broken.error = "retrieval failed: the database went away"
    broken.metrics = QuestionMetrics()
    path = results_file(tmp_path, broken)
    idle = ScriptedChatModel({})
    wired(idle)

    exit_code = main([str(path)])

    assert exit_code == 0
    assert idle.calls == []
    assert load(path).questions[0].scoring_detail == {"skipped": "retrieval failed"}


# --- skipping, resuming and rescoring ----------------------------------------------


def test_a_second_pass_leaves_the_scored_questions_alone(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())
    main([str(path)])
    before = path.read_bytes()
    idle = ScriptedChatModel({})  # any call it gets fails the test's expectations below
    wired(idle)

    exit_code = main([str(path)])

    assert exit_code == 0
    assert idle.calls == []
    assert path.read_bytes() == before


def test_an_interrupted_pass_keeps_what_it_scored_and_the_next_one_carries_on(wired, tmp_path):
    path = three_questions(tmp_path)
    script = full_script()
    wired(InterruptedJudge(first_question_only(script), input_tokens=1000, output_tokens=100))

    with pytest.raises(KeyboardInterrupt):
        main([str(path)])

    saved = load(path)
    assert saved.questions[0].metrics.context_recall == pytest.approx(2 / 3)
    assert saved.questions[1].scoring_detail is None
    assert saved.scoring.judge_input_tokens == 5000

    rest = {
        SentenceVerdicts: script[SentenceVerdicts][1:],
        PassageVerdicts: [precision(False, True)],
    }
    resumed = ScriptedChatModel(rest)
    wired(resumed)

    assert main([str(path)]) == 0
    assert len(resumed.calls) == 2, "only q002's recall and precision; q001 is not judged again"
    assert load(path).questions[1].metrics.context_recall == 0.5


def test_rescore_scores_every_question_again_and_recounts_the_tokens(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())
    main([str(path)])
    script = full_script()
    script[SentenceVerdicts] = [recall(Q001, True, True, True), recall(Q002, *[True] * 4)]
    wired(ScriptedChatModel(script, input_tokens=10, output_tokens=1))

    exit_code = main([str(path), "--rescore"])

    assert exit_code == 0
    rescored = load(path)
    assert [question.metrics.context_recall for question in rescored.questions] == [1.0, 1.0, None]
    # Seven calls at the new price: the totals are recounted, not added to the last pass's.
    assert (rescored.scoring.judge_input_tokens, rescored.scoring.judge_output_tokens) == (70, 7)


def test_an_interrupted_rescore_leaves_no_score_from_the_pass_before(wired, tmp_path):
    path = three_questions(tmp_path)
    wired(judge())
    main([str(path)])
    wired(InterruptedJudge(first_question_only(full_script())))

    with pytest.raises(KeyboardInterrupt):
        main([str(path), "--rescore"])

    first, second, _ = load(path).questions
    assert first.scoring_detail is not None
    assert second.scoring_detail is None, "cleared, so the next pass scores it with this judge"
    assert second.metrics.context_recall is None


def test_a_file_scored_by_other_prompts_is_not_topped_up_without_rescore(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    wired(judge())
    main([str(path)])
    earlier = load(path)
    earlier.scoring.prompts_sha256 = "0" * 64  # as if the prompts had changed since
    save(earlier, path)
    before = path.read_bytes()
    built = wired(None)

    exit_code = main([str(path)])

    assert exit_code == 2
    assert built == []
    assert path.read_bytes() == before
    assert "--rescore" in capsys.readouterr().err

    wired(judge())
    assert main([str(path), "--rescore"]) == 0
    assert load(path).scoring.prompts_sha256 == PROMPTS_SHA256


# --- judge failures ----------------------------------------------------------------


def test_a_judge_failure_is_recorded_on_its_question_and_the_pass_goes_on(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    script = full_script()
    del script[AnswerClaims]  # q001's faithfulness finds nothing left in the script
    wired(judge(script))

    exit_code = main([str(path)])

    assert exit_code == 1
    failed, unanswered, _ = load(path).questions
    error = failed.scoring_detail["error"]
    assert error["type"] == "AssertionError"
    assert "AnswerClaims" in error["message"]
    # Recall and precision were worked out before the failure; a half-scored question keeps none.
    assert all(getattr(failed.metrics, name) is None for name in FOUR)
    assert unanswered.metrics.context_recall == 0.5, "the pass went on to the next question"
    printed = capsys.readouterr().out
    assert "3 questions, 1 scored, 1 skipped, 1 failed" in printed
    assert "failed: q001" in printed


def test_a_judge_that_miscounts_twice_fails_only_that_question(wired, tmp_path):
    path = three_questions(tmp_path)
    one_verdict = SentenceVerdicts(verdicts=[SentenceVerdict(reason="passage 1", supported=True)])
    script = {
        SentenceVerdicts: [one_verdict, one_verdict, recall(Q002, True, False, False, True)],
        PassageVerdicts: [precision(False, True)],
    }
    wired(judge(script))

    exit_code = main([str(path)])

    assert exit_code == 1
    failed, unanswered, _ = load(path).questions
    assert failed.scoring_detail["error"]["type"] == "MetricError"
    assert unanswered.metrics.context_precision == 0.5


class SilentFailure(ScriptedChatModel):
    """A judge whose every call fails with an exception that carries no message at all."""

    def complete_structured(self, system, user, schema):
        raise ValueError()


def test_a_failure_with_an_empty_message_is_still_recorded(wired, tmp_path, capsys):
    path = results_file(tmp_path, row(Q002, [chunk(1, DEDUCTIBLE, 27)]))
    wired(SilentFailure({}))

    exit_code = main([str(path)])

    assert exit_code == 1
    assert load(path).questions[0].scoring_detail == {
        "error": {"type": "ValueError", "message": ""}
    }
    assert "q002  multi_section FAILED  ValueError" in capsys.readouterr().out


def test_the_next_pass_tries_a_failed_question_again(wired, tmp_path):
    path = three_questions(tmp_path)
    script = full_script()
    del script[AnswerClaims]
    wired(judge(script))
    main([str(path)])
    retry = ScriptedChatModel(first_question_only(full_script()))
    wired(retry)

    exit_code = main([str(path)])

    assert exit_code == 0
    assert len(retry.calls) == 5, "q001 alone; q002 and q003 already have their scores"
    first = load(path).questions[0]
    assert "error" not in first.scoring_detail
    assert first.metrics.faithfulness == 0.5


# --- files that cannot be scored ---------------------------------------------------


def test_a_file_whose_chunks_carry_no_text_is_refused_before_any_model_call(
    wired, tmp_path, capsys
):
    path = Path(shutil.copy(FIXTURES / "20260921T031500Z_hybrid.json", tmp_path))
    before = path.read_bytes()
    built = wired(None)

    exit_code = main([str(path)])

    assert exit_code == 2
    assert built == []
    assert path.read_bytes() == before
    error = capsys.readouterr().err
    assert "chunk text" in error
    assert "python -m evals.run_golden_set --config hybrid" in error


@pytest.mark.parametrize("name", PART_1_RUNS)
def test_the_runs_part_1_committed_are_refused(name, wired, tmp_path, capsys):
    path = Path(shutil.copy(COMMITTED_RESULTS / name, tmp_path))
    wired(None)

    exit_code = main([str(path)])

    assert exit_code == 2
    config_name = name.split("_", 1)[1].removesuffix(".json")
    assert f"python -m evals.run_golden_set --config {config_name}" in capsys.readouterr().err


def test_a_run_against_a_different_golden_set_is_refused(wired, tmp_path, capsys):
    path = results_file(
        tmp_path,
        row(Q001, [chunk(1, PREMIUM, 23)], answer=PREMIUM),
        golden_sha256="ab" * 32,
    )
    wired(None)

    exit_code = main([str(path)])

    assert exit_code == 2
    assert "golden set" in capsys.readouterr().err


def test_a_file_that_is_not_there_is_refused(wired, tmp_path, capsys):
    wired(None)

    exit_code = main([str(tmp_path / "missing.json")])

    assert exit_code == 2
    assert "missing.json" in capsys.readouterr().err


# --- explaining one question -------------------------------------------------------


def test_explain_prints_a_scored_questions_working_from_the_file(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    wired(judge())
    main([str(path)])
    before = path.read_bytes()
    capsys.readouterr()
    built = wired(None)

    exit_code = main([str(path), "--explain", "q001"])

    assert exit_code == 0
    assert built == [], "a scored question is explained from the file, with no model call"
    assert path.read_bytes() == before
    printed = capsys.readouterr().out
    # Every sentence of the reference answer, with its verdict and the judge's reason.
    for sentence in split_sentences(Q001.expected_answer):
        assert sentence in flat(printed)
    assert "NOT supported" in printed and "reason: in no passage" in printed
    # The passages in rank order, where each sits, and whether it is useful.
    assert re.search(r"1\. +useful +p\.23", printed)
    assert re.search(r"2\. +NOT useful +p\.23", printed)
    # The claims in the answer, each with its verdict.
    for claim in PREMIUM_CLAIMS:
        assert claim in flat(printed)
    # The questions written back from the answer, with their similarity to the one asked.
    assert f"1.00 {Q001.question}" in flat(printed)
    # And the four numbers.
    assert re.search(r"context_recall +0\.67\n", printed)
    assert re.search(r"context_precision +1\.00\n", printed)
    assert re.search(r"faithfulness +0\.50\n", printed)
    assert re.search(r"answer_relevance +1\.00\n", printed)


def test_explain_asks_the_judge_afresh_for_a_question_not_yet_scored(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    before = path.read_bytes()
    scripted = judge(first_question_only(full_script()))
    wired(scripted)

    exit_code = main([str(path), "--explain", "q001"])

    assert exit_code == 0
    assert len(scripted.calls) == 5
    assert path.read_bytes() == before, "--explain never writes"
    printed = capsys.readouterr().out
    assert "nothing is written" in printed
    assert re.search(r"faithfulness +0\.50\n", printed)
    assert "5,000 in, 500 out" in printed


def test_explain_says_why_a_question_with_no_answer_has_two_scores(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    wired(judge())
    main([str(path)])
    capsys.readouterr()

    main([str(path), "--explain", "q002"])

    printed = capsys.readouterr().out
    assert re.search(r"context_recall +0\.50\n", printed)
    assert re.search(r"faithfulness +—\n", printed)
    assert "no answer" in printed


def test_explain_on_an_unanswerable_question_calls_no_judge(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    idle = ScriptedChatModel({})
    wired(idle)

    exit_code = main([str(path), "--explain", "q003"])

    assert exit_code == 0
    assert idle.calls == []
    assert "does not answer this question" in flat(capsys.readouterr().out)


def test_explain_refuses_a_question_the_file_does_not_hold(wired, tmp_path, capsys):
    path = three_questions(tmp_path)
    wired(None)

    exit_code = main([str(path), "--explain", "q999"])

    assert exit_code == 2
    assert "q999" in capsys.readouterr().err


# --- the real judge ----------------------------------------------------------------


def test_build_judge_is_the_bedrock_chat_model_on_the_model_and_region_given():
    judge = build_judge(Settings(aws_region="us-east-1"), "us.anthropic.claude-sonnet-5")

    assert isinstance(judge, BedrockChatModel)
    assert (judge.model, judge.region) == ("us.anthropic.claude-sonnet-5", "us-east-1")
