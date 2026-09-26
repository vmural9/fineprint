"""Tests for the four RAGAS metrics and the sentence splitter behind context recall.

No model is called. `ScriptedChatModel` plays the judge and hands back the verdicts each test
wrote, which is how every formula can be checked against a number worked out by hand, and how a
judge that miscounts can be staged on purpose.

The passages are sentences from the 2026 handbook and the reference answer is the golden set's,
so every figure in these fixtures was copied from the handbook, not remembered.
"""

import hashlib
import json

import pytest

from evals import ragas_metrics
from evals.golden import load_golden_set
from evals.ragas_metrics import (
    ANSWER_RELEVANCE_SYSTEM,
    CONTEXT_PRECISION_SYSTEM,
    CONTEXT_RECALL_SYSTEM,
    FAITHFULNESS_CLAIMS_SYSTEM,
    FAITHFULNESS_VERDICTS_SYSTEM,
    NO_PASSAGES,
    PROMPTS,
    PROMPTS_SHA256,
    REPLY_SCHEMAS,
    AnswerClaims,
    ClaimVerdict,
    ClaimVerdicts,
    GeneratedQuestions,
    MetricError,
    MetricResult,
    PassageVerdict,
    PassageVerdicts,
    SentenceVerdict,
    SentenceVerdicts,
    answer_relevance,
    average_precision,
    context_precision,
    context_recall,
    faithfulness,
    prompts_sha256,
    split_sentences,
)
from tests.fakes import FakeEmbedder, ScriptedChatModel

GOLDEN = load_golden_set()
Q001 = next(item for item in GOLDEN if item.id == "q001")  # the Part B premium question

# Passages from the 2026 handbook, pages 23, 23, 42, 30 and 40.
PREMIUM = (
    "The standard Part B premium amount in 2026 is $202.90. Most people pay the standard "
    "Part B premium amount every month."
)
IRMAA = (
    "For 2026, if your modified adjusted gross income for 2024 was above $109,000 if you file "
    "individually or $218,000 if you’re married and file jointly, then you may pay an IRMAA. "
    "Visit Medicare.gov to learn more about IRMAA."
)
HEARING = "Note: Original Medicare doesn’t cover hearing aids or exams for fitting hearing aids."
DEDUCTIBLE = (
    "Under Original Medicare, if the Part B deductible ($283 in 2026) applies, you must pay all "
    "costs (up to the Medicare-approved amount) until you meet the yearly Part B deductible."
)
EQUIPMENT = (
    "More expensive equipment, like wheelchairs and hospital beds, become yours after 13 months "
    "of rental payments."
)

# A generated answer: two claims the premium passages support, and one that page 42 contradicts.
QUESTION = "How much is the Part B premium in 2026?"
ANSWER = (
    "The standard Part B premium in 2026 is $202.90 a month. You may pay more, an IRMAA, if your "
    "2024 income was above $109,000 and you file individually. Original Medicare covers hearing "
    "aids."
)
PREMIUM_CLAIM = "The standard Part B premium in 2026 is $202.90 a month."
IRMAA_CLAIM = (
    "You may pay an IRMAA if your 2024 income was above $109,000 and you file individually."
)
HEARING_CLAIM = "Original Medicare covers hearing aids."

# An abstention, the kind of answer that makes no claims at all.
ABSTENTION_QUESTION = "Does Medicare drug coverage pay for Eliquis?"
ABSTENTION = (
    "The handbook doesn't list the drugs a plan covers. Check the plan's formulary or visit "
    "Medicare.gov/plan-compare."
)


def sentence_verdicts(*supported: bool) -> SentenceVerdicts:
    """A context recall judge's reply: one verdict per reference sentence."""
    return SentenceVerdicts(
        verdicts=[
            SentenceVerdict(reason="passage 1" if ok else "in no passage", supported=ok)
            for ok in supported
        ]
    )


def passage_verdicts(*useful: bool) -> PassageVerdicts:
    """A context precision judge's reply: one verdict per passage, in rank order."""
    return PassageVerdicts(
        verdicts=[
            PassageVerdict(reason="gives the premium" if ok else "gives nothing used", useful=ok)
            for ok in useful
        ]
    )


def claim_list(*claims: str) -> AnswerClaims:
    """A faithfulness judge's first reply: the claims it found in the answer."""
    return AnswerClaims(claims=list(claims))


def claim_verdicts(*supported: bool) -> ClaimVerdicts:
    """A faithfulness judge's second reply: one verdict per claim."""
    return ClaimVerdicts(
        verdicts=[
            ClaimVerdict(reason="passage 1" if ok else "page 42 says the opposite", supported=ok)
            for ok in supported
        ]
    )


def generated(*questions: str, noncommittal: bool = False) -> GeneratedQuestions:
    """An answer relevance judge's reply: the questions it wrote from the answer."""
    return GeneratedQuestions(questions=list(questions), noncommittal=noncommittal)


def non_whitespace(text: str) -> str:
    """Every character of `text` except whitespace, in order."""
    return "".join(text.split())


class TableEmbedder:
    """Looks each text up in a table of fixed vectors, deliberately not of unit length."""

    dimension = 2

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


# --- splitting a reference answer into sentences ----------------------------------------


@pytest.mark.parametrize("item", GOLDEN, ids=lambda item: item.id)
def test_every_golden_expected_answer_splits_into_whole_sentences(item):
    sentences = split_sentences(item.expected_answer)

    assert sentences, "a reference answer has at least one sentence"
    assert all(sentence and sentence == sentence.strip() for sentence in sentences)
    assert all(sentence in item.expected_answer for sentence in sentences), "cut, never rewritten"
    assert non_whitespace("".join(sentences)) == non_whitespace(item.expected_answer), (
        "every character but the spaces between sentences is kept, in order"
    )


def test_golden_expected_answers_split_where_a_reader_would():
    by_id = {item.id: split_sentences(item.expected_answer) for item in GOLDEN}

    assert by_id["q001"][0] == (
        "The standard Part B premium in 2026 is $202.90 a month, and most people pay that "
        "standard amount."
    )
    assert len(by_id["q001"]) == 3
    assert by_id["q010"][0] == "No.", "a one-word reply is a sentence of its own"
    assert by_id["q006"][1] == (
        "(The shifted window only applies if the birthday falls on the first of the month, and "
        "hers does not.)"
    )
    # "U.S." ends the second sentence only where a closing quote shows the phrase is over.
    assert len(by_id["q021"]) == 3
    assert by_id["q021"][1].endswith("so Puerto Rico is not “outside the U.S.”")
    assert by_id["q021"][2].startswith("There are some limited exceptions")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Example: Mr. Smith’s Initial Enrollment Period ended December 2021. He waited until "
            "March 2024 (during the General Enrollment Period) to sign up for Part B.",
            [
                "Example: Mr. Smith’s Initial Enrollment Period ended December 2021.",
                "He waited until March 2024 (during the General Enrollment Period) to sign up for "
                "Part B.",
            ],
            id="a title before a name",
        ),
        pytest.param(
            "Medicare generally doesn’t cover health care while you’re traveling outside the U.S. "
            "(the “U.S.” includes the 50 states, the District of Columbia, Puerto Rico, the U.S. "
            "Virgin Islands, Guam, the Northern Mariana Islands, and American Samoa). There are "
            "some limited exceptions.",
            [
                "Medicare generally doesn’t cover health care while you’re traveling outside the "
                "U.S. (the “U.S.” includes the 50 states, the District of Columbia, Puerto Rico, "
                "the U.S. Virgin Islands, Guam, the Northern Mariana Islands, and American Samoa).",
                "There are some limited exceptions.",
            ],
            id="initials",
        ),
        pytest.param(
            "Some plans are offered by private companies, e.g. Medicare Advantage Plans. They must "
            "follow rules set by Medicare.",
            [
                "Some plans are offered by private companies, e.g. Medicare Advantage Plans.",
                "They must follow rules set by Medicare.",
            ],
            id="e.g. before a capital",
        ),
        pytest.param(
            "The Open Enrollment Period runs Oct. 15 to Dec. 7 each year. Coverage starts on "
            "January 1.",
            [
                "The Open Enrollment Period runs Oct. 15 to Dec. 7 each year.",
                "Coverage starts on January 1.",
            ],
            id="abbreviated dates",
        ),
        pytest.param(
            "Use the return envelope that came with your bill, and mail your Medicare payment "
            "coupon and payment to Medicare Premium Collection Center, PO Box 790355, St. Louis, "
            "MO 63179-0355. If you have questions about your premiums, call 1-800-MEDICARE or "
            "visit Medicare.gov/basics/costs/pay-premiums.",
            [
                "Use the return envelope that came with your bill, and mail your Medicare payment "
                "coupon and payment to Medicare Premium Collection Center, PO Box 790355, St. "
                "Louis, MO 63179-0355.",
                "If you have questions about your premiums, call 1-800-MEDICARE or visit "
                "Medicare.gov/basics/costs/pay-premiums.",
            ],
            id="an address and a web address",
        ),
        pytest.param(
            "At a glance: Original Medicare vs. Medicare Advantage Plan",
            ["At a glance: Original Medicare vs. Medicare Advantage Plan"],
            id="vs.",
        ),
    ],
)
def test_an_abbreviation_does_not_end_a_sentence(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "The standard Part B premium amount in 2026 is $202.90. Most people pay the standard "
            "Part B premium amount every month.",
            [
                "The standard Part B premium amount in 2026 is $202.90.",
                "Most people pay the standard Part B premium amount every month.",
            ],
            id="after a dollar amount",
        ),
        pytest.param(
            "Important! If you don’t sign up for Part B when you’re first eligible, you may have "
            "to pay a late enrollment penalty for as long as you have Part B.",
            [
                "Important!",
                "If you don’t sign up for Part B when you’re first eligible, you may have to pay a "
                "late enrollment penalty for as long as you have Part B.",
            ],
            id="after an exclamation",
        ),
        pytest.param(
            "Do I need to choose a primary care doctor? No. Do I have to get a referral to use a "
            "specialist?",
            [
                "Do I need to choose a primary care doctor?",
                "No.",
                "Do I have to get a referral to use a specialist?",
            ],
            id="a question answered in one word",
        ),
        pytest.param(
            "His Part B premium penalty is 20%, and he’ll have to pay this penalty in addition to "
            "his standard Part B premium for as long as he has Part B. (Even though Mr. Smith "
            "didn’t have Part B for 27 months, this included only 2 full 12-month periods.)",
            [
                "His Part B premium penalty is 20%, and he’ll have to pay this penalty in addition "
                "to his standard Part B premium for as long as he has Part B.",
                "(Even though Mr. Smith didn’t have Part B for 27 months, this included only 2 "
                "full 12-month periods.)",
            ],
            id="after a single letter, and a sentence in brackets",
        ),
        pytest.param("  Yes.   No.  \n", ["Yes.", "No."], id="spaces around sentences"),
    ],
)
def test_a_sentence_ends_where_a_reader_would_end_it(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_text_with_nothing_but_whitespace_has_no_sentences(text):
    assert split_sentences(text) == []


# --- context recall ---------------------------------------------------------------------


def test_recall_is_the_share_of_reference_sentences_the_passages_support():
    judge = ScriptedChatModel({SentenceVerdicts: [sentence_verdicts(True, True, False)]})

    result = context_recall(Q001.question, Q001.expected_answer, [PREMIUM, IRMAA, HEARING], judge)

    assert result.value == pytest.approx(2 / 3)
    rows = result.detail["sentences"]
    assert [row["sentence"] for row in rows] == split_sentences(Q001.expected_answer)
    assert [row["supported"] for row in rows] == [True, True, False]
    assert rows[2]["reason"] == "in no passage"
    assert (result.input_tokens, result.output_tokens) == (900, 120)


def test_recall_shows_the_judge_the_question_and_the_numbered_sentences_and_passages():
    judge = ScriptedChatModel({SentenceVerdicts: [sentence_verdicts(True, True, True)]})

    context_recall(Q001.question, Q001.expected_answer, [PREMIUM, IRMAA], judge)

    (call,) = judge.calls
    assert call.system == CONTEXT_RECALL_SYSTEM
    assert call.schema is SentenceVerdicts
    assert Q001.question in call.user
    first, second, third = split_sentences(Q001.expected_answer)
    assert f"1. {first}\n2. {second}\n3. {third}" in call.user
    assert f'<passage number="1">{PREMIUM}</passage>' in call.user
    assert f'<passage number="2">{IRMAA}</passage>' in call.user


def test_recall_asks_again_when_the_judge_miscounts_and_uses_the_second_reply():
    judge = ScriptedChatModel(
        {SentenceVerdicts: [sentence_verdicts(True, True), sentence_verdicts(True, False, False)]}
    )

    result = context_recall(Q001.question, Q001.expected_answer, [PREMIUM], judge)

    assert result.value == pytest.approx(1 / 3)
    first, second = judge.calls
    assert second.user.startswith(first.user), "the same request, with a note added"
    assert "gave 2 verdicts, but exactly 3" in second.user
    assert (result.input_tokens, result.output_tokens) == (1800, 240), "both attempts count"


def test_recall_gives_up_when_the_judge_miscounts_twice():
    judge = ScriptedChatModel(
        {SentenceVerdicts: [sentence_verdicts(True, True), sentence_verdicts(*[True] * 4)]}
    )

    with pytest.raises(MetricError, match="2 and then 4 verdicts where 3"):
        context_recall(Q001.question, Q001.expected_answer, [PREMIUM], judge)
    assert len(judge.calls) == 2, "one retry, no more"


def test_recall_tells_a_judge_with_no_passages_that_there_are_none():
    judge = ScriptedChatModel({SentenceVerdicts: [sentence_verdicts(False, False, False)]})

    result = context_recall(Q001.question, Q001.expected_answer, [], judge)

    assert NO_PASSAGES in judge.calls[0].user
    assert result.value == 0.0


def test_recall_needs_a_reference_answer_with_something_in_it():
    with pytest.raises(ValueError, match="no sentences"):
        context_recall(Q001.question, "   ", [PREMIUM], ScriptedChatModel({}))


# --- context precision ------------------------------------------------------------------


def test_precision_worked_example_useful_passages_at_ranks_1_and_3_of_5():
    judge = ScriptedChatModel(
        {PassageVerdicts: [passage_verdicts(True, False, True, False, False)]}
    )

    result = context_precision(
        Q001.question, Q001.expected_answer, [PREMIUM, HEARING, IRMAA, DEDUCTIBLE, EQUIPMENT], judge
    )

    # precision@1 = 1/1 and precision@3 = 2/3, averaged over the two ranks that are useful.
    assert result.value == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert round(result.value, 4) == 0.8333
    rows = result.detail["passages"]
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["useful"] for row in rows] == [True, False, True, False, False]
    assert rows[1]["reason"] == "gives nothing used"
    assert (result.input_tokens, result.output_tokens) == (900, 120)


@pytest.mark.parametrize(
    ("useful", "expected"),
    [
        ([True, False, True, False, False], (1 / 1 + 2 / 3) / 2),
        ([False, False, False, True, True], (1 / 4 + 2 / 5) / 2),
        ([True, True, False, False, False], 1.0),
        ([False, False, True], 1 / 3),
        ([False, False, False], 0.0),
        ([], 0.0),
    ],
)
def test_average_precision_rewards_useful_passages_ranked_first(useful, expected):
    assert average_precision(useful) == pytest.approx(expected)


def test_precision_shows_the_judge_the_whole_reference_answer_and_the_ranked_passages():
    judge = ScriptedChatModel({PassageVerdicts: [passage_verdicts(True, False)]})

    context_precision(Q001.question, Q001.expected_answer, [PREMIUM, HEARING], judge)

    (call,) = judge.calls
    assert call.system == CONTEXT_PRECISION_SYSTEM
    assert call.schema is PassageVerdicts
    assert Q001.question in call.user
    assert Q001.expected_answer in call.user
    assert call.user.index(f'<passage number="1">{PREMIUM}</passage>') < call.user.index(
        f'<passage number="2">{HEARING}</passage>'
    )


def test_precision_with_no_passages_is_zero_and_asks_nobody():
    judge = ScriptedChatModel({})

    result = context_precision(Q001.question, Q001.expected_answer, [], judge)

    assert result == MetricResult(
        value=0.0, detail={"passages": []}, input_tokens=0, output_tokens=0
    )
    assert judge.calls == []


# --- faithfulness -----------------------------------------------------------------------


def test_faithfulness_is_the_share_of_the_answers_claims_the_passages_support():
    judge = ScriptedChatModel(
        {
            AnswerClaims: [claim_list(PREMIUM_CLAIM, IRMAA_CLAIM, HEARING_CLAIM)],
            ClaimVerdicts: [claim_verdicts(True, True, False)],
        }
    )

    result = faithfulness(QUESTION, ANSWER, [PREMIUM, IRMAA], judge)

    assert result.value == pytest.approx(2 / 3)
    assert result.detail["no_claims"] is False
    rows = result.detail["claims"]
    assert [row["claim"] for row in rows] == [PREMIUM_CLAIM, IRMAA_CLAIM, HEARING_CLAIM]
    assert [row["supported"] for row in rows] == [True, True, False]
    assert rows[2]["reason"] == "page 42 says the opposite"
    assert (result.input_tokens, result.output_tokens) == (1800, 240), "two calls, both counted"


def test_faithfulness_takes_claims_from_the_answer_then_checks_them_against_the_passages():
    judge = ScriptedChatModel(
        {
            AnswerClaims: [claim_list(PREMIUM_CLAIM, IRMAA_CLAIM)],
            ClaimVerdicts: [claim_verdicts(True, True)],
        }
    )

    faithfulness(QUESTION, ANSWER, [PREMIUM, IRMAA], judge)

    claims_call, verdicts_call = judge.calls
    assert claims_call.system == FAITHFULNESS_CLAIMS_SYSTEM
    assert claims_call.schema is AnswerClaims
    assert QUESTION in claims_call.user and ANSWER in claims_call.user
    assert PREMIUM not in claims_call.user, "the claims come from the answer alone"
    assert verdicts_call.system == FAITHFULNESS_VERDICTS_SYSTEM
    assert verdicts_call.schema is ClaimVerdicts
    assert f"1. {PREMIUM_CLAIM}\n2. {IRMAA_CLAIM}" in verdicts_call.user
    assert f'<passage number="1">{PREMIUM}</passage>' in verdicts_call.user
    assert ANSWER not in verdicts_call.user, "each claim is checked on its own"


def test_an_answer_that_makes_no_claims_has_no_faithfulness_score():
    judge = ScriptedChatModel({AnswerClaims: [claim_list()]})

    result = faithfulness(ABSTENTION_QUESTION, ABSTENTION, [PREMIUM], judge)

    assert result.value is None
    assert result.detail == {"no_claims": True, "claims": []}
    assert len(judge.calls) == 1, "with nothing to check, the judge is not asked to check it"
    assert (result.input_tokens, result.output_tokens) == (900, 120)


# --- answer relevance -------------------------------------------------------------------


def test_relevance_is_one_when_every_generated_question_is_the_one_asked():
    judge = ScriptedChatModel({GeneratedQuestions: [generated(QUESTION, QUESTION, QUESTION)]})

    result = answer_relevance(QUESTION, ANSWER, judge, FakeEmbedder())

    assert result.value == pytest.approx(1.0)
    assert result.detail["noncommittal"] is False
    assert [row["question"] for row in result.detail["questions"]] == [QUESTION] * 3
    assert (result.input_tokens, result.output_tokens) == (900, 120)


def test_relevance_is_the_mean_cosine_similarity_to_the_question_asked():
    questions = [
        "What is the standard Part B premium in 2026?",
        "How much do most people pay for Part B each month?",
        "When do you pay an IRMAA?",
    ]
    judge = ScriptedChatModel({GeneratedQuestions: [generated(*questions)]})
    embedder = FakeEmbedder()

    result = answer_relevance(QUESTION, ANSWER, judge, embedder)

    # FakeEmbedder's vectors have unit length, so each cosine is a plain dot product.
    asked = embedder.embed_query(QUESTION)
    expected = [
        sum(a * b for a, b in zip(embedder.embed_query(text), asked, strict=True))
        for text in questions
    ]
    assert [row["similarity"] for row in result.detail["questions"]] == pytest.approx(expected)
    assert result.value == pytest.approx(sum(expected) / 3)


def test_relevance_divides_out_vector_lengths_rather_than_assuming_unit_vectors():
    embedder = TableEmbedder(
        {
            QUESTION: [3.0, 4.0],  # length 5
            "same direction": [6.0, 8.0],  # length 10, cosine 1.0
            "at right angles": [4.0, -3.0],  # cosine 0.0
            "in between": [3.0, 0.0],  # cosine 9 / (5 × 3) = 0.6
        }
    )
    judge = ScriptedChatModel(
        {GeneratedQuestions: [generated("same direction", "at right angles", "in between")]}
    )

    result = answer_relevance(QUESTION, ANSWER, judge, embedder)

    similarities = [row["similarity"] for row in result.detail["questions"]]
    assert similarities == pytest.approx([1.0, 0.0, 0.6])
    # A bare dot product would have given (50 + 0 + 9) / 3.
    assert result.value == pytest.approx((1.0 + 0.0 + 0.6) / 3)


def test_a_noncommittal_answer_scores_zero_however_close_its_questions_come():
    judge = ScriptedChatModel(
        {
            GeneratedQuestions: [
                generated(*[ABSTENTION_QUESTION] * 3, noncommittal=True),
            ]
        }
    )

    result = answer_relevance(ABSTENTION_QUESTION, ABSTENTION, judge, FakeEmbedder())

    assert result.value == 0.0
    assert result.detail["noncommittal"] is True
    similarities = [row["similarity"] for row in result.detail["questions"]]
    assert similarities == pytest.approx([1.0] * 3), "measured and recorded, then overruled"


def test_relevance_shows_the_judge_the_answer_but_never_the_question():
    judge = ScriptedChatModel({GeneratedQuestions: [generated("One?", "Two?")]})

    answer_relevance(QUESTION, ANSWER, judge, FakeEmbedder(), n_questions=2)

    (call,) = judge.calls
    assert call.system == ANSWER_RELEVANCE_SYSTEM
    assert call.schema is GeneratedQuestions
    assert ANSWER in call.user
    assert QUESTION not in call.user, "a judge that saw the question could simply copy it"
    assert "exactly 2 questions" in call.user


def test_relevance_needs_at_least_one_question():
    with pytest.raises(ValueError, match="n_questions"):
        answer_relevance(QUESTION, ANSWER, ScriptedChatModel({}), FakeEmbedder(), n_questions=0)


# --- rules every metric follows ---------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "script"),
    [
        pytest.param(
            lambda judge: context_precision(
                Q001.question, Q001.expected_answer, [PREMIUM, IRMAA], judge
            ),
            {PassageVerdicts: [passage_verdicts(True), passage_verdicts(True, True, True)]},
            id="context_precision",
        ),
        pytest.param(
            lambda judge: faithfulness(QUESTION, ANSWER, [PREMIUM], judge),
            {
                AnswerClaims: [claim_list(PREMIUM_CLAIM, IRMAA_CLAIM)],
                ClaimVerdicts: [claim_verdicts(True), claim_verdicts(True, True, True)],
            },
            id="faithfulness",
        ),
        pytest.param(
            lambda judge: answer_relevance(QUESTION, ANSWER, judge, FakeEmbedder()),
            {GeneratedQuestions: [generated("One?", "Two?"), generated("A?", "B?", "C?", "D?")]},
            id="answer_relevance",
        ),
    ],
)
def test_every_metric_gives_up_when_the_judge_miscounts_twice(score, script):
    with pytest.raises(MetricError, match="where"):
        score(ScriptedChatModel(script))


def test_every_detail_can_be_written_into_a_results_file_as_json():
    judge = ScriptedChatModel(
        {
            SentenceVerdicts: [sentence_verdicts(True, False, True)],
            PassageVerdicts: [passage_verdicts(True, False)],
            AnswerClaims: [claim_list(PREMIUM_CLAIM)],
            ClaimVerdicts: [claim_verdicts(True)],
            GeneratedQuestions: [generated(QUESTION, QUESTION, QUESTION)],
        }
    )

    results = [
        context_recall(Q001.question, Q001.expected_answer, [PREMIUM, IRMAA], judge),
        context_precision(Q001.question, Q001.expected_answer, [PREMIUM, HEARING], judge),
        faithfulness(QUESTION, ANSWER, [PREMIUM], judge),
        answer_relevance(QUESTION, ANSWER, judge, FakeEmbedder()),
    ]

    for result in results:
        assert json.loads(json.dumps(result.detail)) == result.detail


SYSTEM_PROMPTS = {
    "context_recall": CONTEXT_RECALL_SYSTEM,
    "context_precision": CONTEXT_PRECISION_SYSTEM,
    "faithfulness_claims": FAITHFULNESS_CLAIMS_SYSTEM,
    "faithfulness_verdicts": FAITHFULNESS_VERDICTS_SYSTEM,
    "answer_relevance": ANSWER_RELEVANCE_SYSTEM,
}


@pytest.mark.parametrize("prompt", SYSTEM_PROMPTS.values(), ids=SYSTEM_PROMPTS.keys())
def test_every_prompt_keeps_the_judge_to_what_it_is_given_and_to_the_schema(prompt):
    lowered = prompt.lower()
    assert "use only" in lowered
    assert "anything you know about medicare" in lowered
    assert "answer in the structured form you are asked for" in lowered


# --- the prompts fingerprint ------------------------------------------------------------


def test_prompts_sha256_is_the_sha256_of_the_prompts_joined_together():
    assert hashlib.sha256("".join(PROMPTS).encode("utf-8")).hexdigest() == PROMPTS_SHA256
    assert prompts_sha256(PROMPTS) == PROMPTS_SHA256


def test_every_piece_of_prompt_text_in_the_module_is_in_the_fingerprint():
    texts = {
        name: value
        for name, value in vars(ragas_metrics).items()
        if name.isupper() and isinstance(value, str) and name != "PROMPTS_SHA256"
    }

    assert len(texts) == len(PROMPTS)
    assert set(texts.values()) == set(PROMPTS)


def test_the_reply_schemas_are_in_the_fingerprint_as_the_judge_receives_them():
    # The field descriptions are instructions too: they reach the judge with the schema.
    schemas = [SentenceVerdicts, PassageVerdicts, AnswerClaims, ClaimVerdicts, GeneratedQuestions]

    assert json.loads(REPLY_SCHEMAS) == [schema.model_json_schema() for schema in schemas]
    assert REPLY_SCHEMAS in PROMPTS


@pytest.mark.parametrize("index", range(len(PROMPTS)))
def test_changing_any_prompt_by_one_character_changes_the_fingerprint(index):
    changed = list(PROMPTS)
    changed[index] += " "

    assert prompts_sha256(changed) != PROMPTS_SHA256


# --- the scripted judge itself ----------------------------------------------------------


def test_the_scripted_judge_answers_each_schema_from_its_own_queue_in_order():
    first, second = claim_list(PREMIUM_CLAIM), claim_list(IRMAA_CLAIM)
    verdicts = claim_verdicts(True)
    script = {AnswerClaims: [first, second], ClaimVerdicts: [verdicts]}
    judge = ScriptedChatModel(script, input_tokens=7, output_tokens=3)

    one = judge.complete_structured(system="s", user="u1", schema=AnswerClaims)
    two = judge.complete_structured(system="s", user="u2", schema=ClaimVerdicts)
    three = judge.complete_structured(system="s", user="u3", schema=AnswerClaims)

    assert (one.parsed, two.parsed, three.parsed) == (first, verdicts, second)
    assert (one.input_tokens, one.output_tokens, one.model) == (7, 3, "scripted-chat-model")
    assert [call.user for call in judge.calls] == ["u1", "u2", "u3"]
    assert [call.schema for call in judge.calls] == [AnswerClaims, ClaimVerdicts, AnswerClaims]
    assert len(script[AnswerClaims]) == 2, "the test's own lists are left as they were"
    with pytest.raises(AssertionError, match="AnswerClaims"):
        judge.complete_structured(system="s", user="u4", schema=AnswerClaims)
    with pytest.raises(AssertionError, match="SentenceVerdicts"):
        judge.complete_structured(system="s", user="u5", schema=SentenceVerdicts)


def test_the_scripted_judge_refuses_a_reply_filed_under_the_wrong_schema():
    with pytest.raises(TypeError, match="ClaimVerdicts"):
        ScriptedChatModel({AnswerClaims: [claim_verdicts(True)]})
