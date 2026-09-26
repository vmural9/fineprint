"""The four RAGAS metrics, implemented here from the RAGAS definitions, with a model as judge.

Part 1 measured retrieval by pages: was an expected page among the passages? These four metrics
read the text itself, which takes a reader, so a language model, the judge, does the reading and
plain Python turns its verdicts into numbers. The definitions come from RAGAS (the paper "RAGAS:
Automated Evaluation of Retrieval Augmented Generation", Es et al., 2023, and the project's
documentation); every prompt and every formula is written out below.

- **Context recall**: did retrieval find what the answer needs? The reference answer is split
  into sentences, the judge says for each one whether the retrieved passages support it, and the
  score is the supported share.
- **Context precision**: did retrieval rank the useful passages first? The judge marks each
  passage useful or not for reaching the reference answer, and `average_precision` rewards the
  useful ones for sitting near the top.
- **Faithfulness**: does the generated answer stick to its passages? The judge lists the
  answer's factual claims, then checks each one against the passages; the score is the
  supported share.
- **Answer relevance**: does the answer address the question that was asked? The judge, shown
  only the answer, writes questions it would answer completely, and the score is how close those
  come to the real question: the mean cosine similarity of their embeddings. An answer that
  avoids the question scores 0.

Every judge call goes through `ChatModel.complete_structured`, so a verdict arrives as a validated
Pydantic object rather than as free text to be parsed. Every verdict also carries the 1-based
number of the sentence, passage or claim it is about, so it can be matched to the right one even
when the judge lists them out of order — the failure a live run once produced, where the
precision judge marked passage 1 useful for a figure that was actually in passage 2. When the
judge is asked for one verdict per numbered item and returns the wrong count, or numbers that are
not `1..n` once each, it is asked once more; a second miss raises `MetricError`, because a
verdict list that still cannot be matched to the items it is about cannot be scored, and a guessed
match would be a made-up score.

A judged score is an estimate with its own error, not a measurement like page hit. So each
`MetricResult` keeps the judge's verdicts and reasons, and `PROMPTS_SHA256` records which prompts
produced it. Which questions a metric applies to is the caller's decision: a question the
handbook cannot answer has no facts to recall and no claims to check.
"""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

from fineprint.providers.base import ChatModel, Embedder, LLMResult

# What the judge is told, metric by metric. Each `_SYSTEM` prompt holds the instructions and each
# `_USER` template lays out the material to be judged; the reply schemas further down give the
# shape its answer must take. `PROMPTS_SHA256` covers all of it.

CONTEXT_RECALL_SYSTEM = """\
You check whether passages found by a search of the official U.S. government handbook \
"Medicare & You 2026" contain the facts of a reference answer.

You are given a question, a reference answer to it split into numbered sentences, and the \
passages the search found, numbered in the order it ranked them. For each sentence, decide \
whether the passages support it.

Rules:

1. Use only the passages. Do not use anything you know about Medicare, and do not assume a \
passage says more than it does.

2. A sentence is supported when every fact in it (each amount, date, time limit, condition and \
rule) is stated in the passages or follows directly from them. A sentence the passages support \
only in part is not supported.

3. A figure must match the passages exactly, or be worked out directly from figures they give. \
An amount, percentage, date or number of days that differs from the passages, or that no \
passage gives, leaves its sentence unsupported.

4. Read each sentence with the question in mind. A short reply such as "Yes." or "No to both." \
stands for the answer it gives to the question, and a sentence about the answer itself, such as \
"A correct answer must give the deductible", stands for the facts it names. Judge those facts.

5. Give each verdict a one-line reason: the number of the passage that supports the sentence, \
or what is missing.

6. Give each verdict the number of the sentence it is about, and list your verdicts in that \
order, sentence 1 first.

Answer in the structured form you are asked for, with exactly one verdict for each numbered \
sentence, each carrying that sentence's number, listed in the same order.\
"""

CONTEXT_RECALL_USER = """\
Question: {question}

Reference answer, in {count} numbered sentences:
{sentences}

Passages, in the order the search ranked them:
{passages}\
"""

CONTEXT_PRECISION_SYSTEM = """\
You judge whether each passage found by a search of the official U.S. government handbook \
"Medicare & You 2026" is useful for answering a question.

You are given a question, a reference answer that is known to be correct, and the passages the \
search found, numbered in the order it ranked them. For each passage, decide whether it is \
useful for reaching the reference answer.

Rules:

1. Use only the question, the reference answer and the passages. Do not use anything you know \
about Medicare.

2. A passage is useful when it gives at least one fact, figure, condition or rule that the \
reference answer states or relies on. A passage on the same topic that gives none of them is \
not useful.

3. Judge each passage on its own, as if it were the only one. A passage that repeats a useful \
fact from another passage is still useful.

4. Give each verdict a one-line reason: the fact from the reference answer that the passage \
gives, or why it gives none.

5. Give each verdict the number of the passage it is about, and list your verdicts in that \
order, passage 1 first.

Answer in the structured form you are asked for, with exactly one verdict for each numbered \
passage, each carrying that passage's number, listed in the same order.\
"""

CONTEXT_PRECISION_USER = """\
Question: {question}

Reference answer:
{expected_answer}

Passages, {count} in all, in the order the search ranked them:
{passages}\
"""

FAITHFULNESS_CLAIMS_SYSTEM = """\
You break an answer about Medicare into the factual claims it makes, so that each claim can be \
checked on its own against the official U.S. government handbook "Medicare & You 2026".

You are given a question and the answer that was written for it. List every factual claim the \
answer makes about Medicare: what it covers, what things cost, who qualifies, deadlines, and how \
its rules work.

Rules:

1. Use only the answer. The question is there to make the answer's meaning clear: do not take \
from it anything the answer does not say, and do not add anything you know about Medicare.

2. One fact per claim. Split a sentence that states two facts into two claims.

3. Write each claim as a complete sentence that makes sense on its own: name what "it", "this" \
or "she" refers to, and keep the conditions the fact depends on, such as the year or "if you \
have Part A". When the answer says the handbook states something, the claim is the thing stated.

4. Copy every amount, percentage, date and time limit exactly as the answer gives it, even one \
you believe is wrong. Do not correct, round or check anything.

5. Leave out whatever is not a claim about Medicare: a statement that some information is not \
available or not in the handbook, a suggestion of where to get more help, and advice. An answer \
that only says the information is not available makes no claims, and its list is empty.

Answer in the structured form you are asked for.\
"""

FAITHFULNESS_CLAIMS_USER = """\
Question: {question}

Answer:
{answer}\
"""

FAITHFULNESS_VERDICTS_SYSTEM = """\
You check claims made in an answer about Medicare against passages from the official U.S. \
government handbook "Medicare & You 2026".

You are given numbered passages and numbered claims. For each claim, decide whether the \
passages support it.

Rules:

1. Use only the passages. Do not use anything you know about Medicare: a claim the passages do \
not support is unsupported even if you believe it is true.

2. A claim is supported when the passages state it or it follows directly from what they state. \
A claim that goes beyond the passages, or that they support only in part, is not supported.

3. A figure must match the passages exactly, or be worked out directly from figures they give. \
An amount, percentage, date, year or time limit that differs from the passages leaves its claim \
unsupported.

4. Give each verdict a one-line reason: the number of the passage that supports the claim, or \
what is missing or different.

5. Give each verdict the number of the claim it is about, and list your verdicts in that order, \
claim 1 first.

Answer in the structured form you are asked for, with exactly one verdict for each numbered \
claim, each carrying that claim's number, listed in the same order.\
"""

FAITHFULNESS_VERDICTS_USER = """\
Passages:
{passages}

Claims, {count} in all:
{claims}\
"""

ANSWER_RELEVANCE_SYSTEM = """\
You work out which questions an answer about Medicare responds to.

You are given an answer that a question-answering service wrote from the official U.S. \
government handbook "Medicare & You 2026". You are not shown the question it was answering.

Rules:

1. Write the number of questions you are asked for. Each must be a question this answer would \
answer completely, asked the way a person with Medicare or a family member might ask it. Word \
them differently from one another.

2. Use only the answer. Ask for what it gives and no more, and do not add anything you know \
about Medicare.

3. Set noncommittal to true when the answer as a whole avoids answering: when it is evasive or \
vague, or says that the information is not available, as in "the handbook does not say" or \
"I don't know". An answer that commits to an answer is not noncommittal, even if it also \
mentions something it cannot tell.

Answer in the structured form you are asked for.\
"""

ANSWER_RELEVANCE_USER = """\
Answer:
{answer}

Write exactly {count} questions.\
"""

# Added to a request whose reply had the wrong number of items, when it is sent a second time.
RETRY_NOTE = """\
Your previous reply gave {got} {items}, but exactly {expected} were asked for. Reply again with \
exactly {expected} {items}.\
"""

# Added to a request whose reply's verdicts were not numbered 1..expected, each once, when it is
# sent a second time. A reply that only lists them out of that order is not sent back for this:
# `_ask_for_list` repairs it by sorting, since the numbers alone are enough to put it back in order.
INDEX_RETRY_NOTE = """\
Your previous reply {problem}. Reply again with exactly {expected} {items}, numbered 1..{expected} \
in whatever order you list them, each number used once.\
"""

# Shown in place of the passages when retrieval returned none, so the judge is told so.
NO_PASSAGES = "No passages were retrieved."

# What the judge must reply with, one schema per prompt. In each verdict the index comes first,
# so its number is unambiguous, then the reason before the yes or no, so the judge explains
# before it decides — the order the RAGAS prompts ask for.


def _list_from_json_string(value: Any) -> Any:
    """Turn a list the judge wrote out as a JSON string back into a list; pass anything else on.

    The judge sometimes sends a list-valued field as text holding JSON, '[{"reason": ...}]',
    instead of as a list. Rejecting that would make the chat model send the reply back to be
    redone, which costs a second call, and a second such reply would fail the question. A string
    that is not JSON is passed on unchanged, for validation to reject as usual.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# Lets a list-valued reply field also arrive as that list written out as a JSON string. It runs
# before validation and leaves the JSON schema, and so what the judge is shown, unchanged.
_OR_JSON_STRING = BeforeValidator(_list_from_json_string)


class SentenceVerdict(BaseModel):
    """The judge's call on one sentence of the reference answer."""

    index: int = Field(
        description="The 1-based number of the sentence, as it is numbered in the prompt"
    )
    reason: str = Field(
        description="One line: the passage that supports the sentence, or what is missing"
    )
    supported: bool = Field(description="True when the passages support everything it states")


class SentenceVerdicts(BaseModel):
    """What the context recall judge returns."""

    verdicts: Annotated[list[SentenceVerdict], _OR_JSON_STRING] = Field(
        description="One verdict for each numbered sentence, in the same order"
    )


class PassageVerdict(BaseModel):
    """The judge's call on one retrieved passage."""

    index: int = Field(
        description="The 1-based number of the passage, as it is numbered in the prompt"
    )
    reason: str = Field(
        description="One line: the fact from the reference answer it gives, or why it gives none"
    )
    useful: bool = Field(description="True when it gives a fact the reference answer uses")


class PassageVerdicts(BaseModel):
    """What the context precision judge returns."""

    verdicts: Annotated[list[PassageVerdict], _OR_JSON_STRING] = Field(
        description="One verdict for each numbered passage, in the same order"
    )


class AnswerClaims(BaseModel):
    """What the faithfulness judge returns first: the claims it finds in the answer."""

    claims: Annotated[list[str], _OR_JSON_STRING] = Field(
        description="Every factual claim the answer makes, one fact each; empty when it makes none"
    )


class ClaimVerdict(BaseModel):
    """The judge's call on one claim."""

    index: int = Field(
        description="The 1-based number of the claim, as it is numbered in the prompt"
    )
    reason: str = Field(
        description="One line: the passage that supports the claim, or what is missing or different"
    )
    supported: bool = Field(description="True when the passages support the claim")


class ClaimVerdicts(BaseModel):
    """What the faithfulness judge returns second."""

    verdicts: Annotated[list[ClaimVerdict], _OR_JSON_STRING] = Field(
        description="One verdict for each numbered claim, in the same order"
    )


class GeneratedQuestions(BaseModel):
    """What the answer relevance judge returns."""

    questions: Annotated[list[str], _OR_JSON_STRING] = Field(
        description="Questions this answer would answer completely"
    )
    noncommittal: bool = Field(
        description="True when the answer avoids answering, or says the information is unavailable"
    )


# The reply schemas as the judge receives them, field names and descriptions included. They are
# part of what the judge is told, so they are part of the fingerprint.
REPLY_SCHEMAS = json.dumps(
    [
        schema.model_json_schema()
        for schema in (
            SentenceVerdicts,
            PassageVerdicts,
            AnswerClaims,
            ClaimVerdicts,
            GeneratedQuestions,
        )
    ]
)

# Every piece of text the judge is given, apart from the material being judged.
PROMPTS: tuple[str, ...] = (
    CONTEXT_RECALL_SYSTEM,
    CONTEXT_RECALL_USER,
    CONTEXT_PRECISION_SYSTEM,
    CONTEXT_PRECISION_USER,
    FAITHFULNESS_CLAIMS_SYSTEM,
    FAITHFULNESS_CLAIMS_USER,
    FAITHFULNESS_VERDICTS_SYSTEM,
    FAITHFULNESS_VERDICTS_USER,
    ANSWER_RELEVANCE_SYSTEM,
    ANSWER_RELEVANCE_USER,
    RETRY_NOTE,
    INDEX_RETRY_NOTE,
    NO_PASSAGES,
    REPLY_SCHEMAS,
)


def prompts_sha256(prompts: Sequence[str]) -> str:
    """The sha256 of the prompts joined end to end, as 64 hex digits."""
    return hashlib.sha256("".join(prompts).encode("utf-8")).hexdigest()


# Stored with every score, so a number can be tied to the exact prompt text that produced it and
# a score from before a prompt changed can be told apart from one after.
PROMPTS_SHA256 = prompts_sha256(PROMPTS)


@dataclass(frozen=True)
class MetricResult:
    """One metric's score for one question, with the judge's working and what it cost.

    `detail` holds the sentences, claims, questions and verdicts the score was computed from, in
    plain JSON types, so a score can be explained and stored in a results file.
    """

    value: float | None  # None means not measurable for this question (see each metric)
    detail: dict[str, Any]  # the judge's verdicts and what they are about
    input_tokens: int
    output_tokens: int


class MetricError(RuntimeError):
    """The judge twice replied with a list that does not line up with what it was asked about."""


# Where a sentence can end: a full stop, question mark or exclamation mark (or a run of them),
# then whitespace and a capital letter. Requiring the whitespace and the capital is what keeps
# "$202.90", "Medicare.gov/plan-compare" and "Oct. 15" whole.
_SENTENCE_END = re.compile(
    r"""
    (?P<stop>[.!?]+)          # the sentence's own punctuation: . ! ? or a run such as ?!
    (?P<closers>[”’"')\]]*)   # quotes and brackets that close inside the sentence
    \s+                       # the space between two sentences
    (?=[“‘"'(\[]*[A-Z])       # a capital next, perhaps after an opening quote or bracket
    """,
    re.VERBOSE,
)

# Words that end in a full stop without ending the sentence, and that a capital often follows:
# "Mr. Smith", "St. Louis" and "Original Medicare vs. Medicare Advantage Plan" are all in the
# handbook. Initials such as "U.S." and "e.g." are recognised by their shape instead.
_ABBREVIATIONS = frozenset({"dr", "jr", "mr", "mrs", "ms", "sr", "st", "vs"})

# Single letters joined by full stops, as they stand before the last one: "U.S", "e.g", "D.C".
_INITIALS = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, keeping every character except the space between them.

    A sentence ends at ".", "!" or "?" followed by a space and a capital letter, so a figure
    ("$202.90"), a web address ("Medicare.gov/plan-compare") or a date ("Oct. 15") never ends
    one. Neither does the full stop of an abbreviation ("Mr. Smith", "the U.S. Virgin Islands",
    "e.g. Medicare Advantage Plans"), unless a closing quote or bracket follows it: in 'is not
    “outside the U.S.” There are', the quote shows that the sentence is over.

    Each sentence is a slice of `text` with the space around it trimmed, and none is empty. The
    rules are fixed, so the same reference answer always gives the same sentences, and a context
    recall score can be traced back to them.
    """
    sentences: list[str] = []
    start = 0
    for end in _SENTENCE_END.finditer(text):
        after_abbreviation = (
            end["stop"] == "."
            and not end["closers"]
            and _ends_in_abbreviation(text[start : end.start()])
        )
        if after_abbreviation:
            continue
        sentences.append(text[start : end.end()].strip())
        start = end.end()
    sentences.append(text[start:].strip())
    return [sentence for sentence in sentences if sentence]


def _ends_in_abbreviation(text: str) -> bool:
    """True when the last word of `text`, whose full stop comes next, is an abbreviation."""
    words = text.split()
    if not words:
        return False
    word = words[-1].lstrip("“‘\"'([")
    return word.lower() in _ABBREVIATIONS or _INITIALS.fullmatch(word) is not None


def numbered_lines(items: Sequence[str]) -> str:
    """`1. first`, `2. second`, one item to a line: how sentences and claims are shown."""
    lines = (" ".join(item.split()) for item in items)
    return "\n".join(f"{number}. {line}" for number, line in enumerate(lines, start=1))


def numbered_passages(contexts: Sequence[str]) -> str:
    """Each passage in a tag carrying its rank, so a reason can say which passage it relies on."""
    if not contexts:
        return NO_PASSAGES
    return "\n\n".join(
        f'<passage number="{number}">{text}</passage>'
        for number, text in enumerate(contexts, start=1)
    )


def average_precision(useful: Sequence[bool]) -> float:
    """Context precision's formula: precision@k averaged over the ranks k that are useful.

    precision@k is the share of useful passages among the first k. Averaging it over the useful
    ranks rewards putting them first: useful passages at ranks 1 and 3 of 5 score
    (1/1 + 2/3) / 2 = 0.8333, and the same two at ranks 4 and 5 score (1/4 + 2/5) / 2 = 0.325.
    With no useful passage the score is 0.0.
    """
    total = 0.0
    found = 0
    for k, is_useful in enumerate(useful, start=1):
        if is_useful:
            found += 1
            total += found / k
    return total / found if found else 0.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """The cosine of the angle between two vectors: 1.0 for the same direction, 0.0 at right angles.

    Titan's vectors have unit length, which makes this a plain dot product for them, but the
    lengths are divided out anyway, so an embedder that does not normalise is scored correctly.
    A zero vector has no direction and scores 0.0.
    """
    lengths = math.sqrt(math.sumprod(a, a)) * math.sqrt(math.sumprod(b, b))
    return math.sumprod(a, b) / lengths if lengths else 0.0


def _index_problem(entries: Sequence[Any], expected: int) -> str | None:
    """None when `entries`' 1-based `index` fields are exactly `1..expected`, each used once, in
    whatever order; otherwise a one-line description of the numbers they gave instead.

    Some other order than `1..expected` is not a problem: `_ask_for_list` repairs it by sorting
    on `index` once this returns `None`. A repeat, a gap or a number outside `1..expected` is a
    different kind of miscount, one that cannot be matched back to the sentence, passage or claim
    the entry was supposed to be a verdict on — the failure that let a passage's verdict be
    attributed to the wrong passage — and gets the same retry a wrong count of entries does.
    """
    indexes = [entry.index for entry in entries]
    if sorted(indexes) == list(range(1, expected + 1)):
        return None
    return f"gave the numbers {indexes} instead of 1..{expected}, each used exactly once"


def _ask_for_list[T: BaseModel](
    judge: ChatModel,
    system: str,
    user: str,
    schema: type[T],
    *,
    items: str,
    expected: int,
    indexed: bool = False,
) -> LLMResult[T]:
    """Ask the judge for a reply whose list field `items` holds exactly `expected` entries.

    A schema can say "a list of verdicts" but not "exactly five", so the count is checked here.
    When `indexed` is set, each entry also carries a 1-based `index` naming which numbered
    sentence, passage or claim it is a verdict on; `_index_problem` checks that those numbers are
    `1..expected`, each once, and a reply that only gives them in some other order is re-sorted
    into place before it is returned.

    On a wrong count, or — when `indexed` — on numbers that are not a plain reordering of
    `1..expected`, the judge is asked again and told what went wrong; a second bad reply raises
    `MetricError`, because a reply that still cannot be matched to the items it is about cannot be
    scored, and a guessed match would be a made-up score. The tokens in the result cover both
    attempts, since both were paid for.
    """
    counts: list[int] = []
    index_problems: list[str] = []
    input_tokens = output_tokens = 0
    latency_ms = 0.0
    request = user
    for _ in range(2):
        result = judge.complete_structured(system=system, user=request, schema=schema)
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        latency_ms += result.latency_ms
        entries = getattr(result.parsed, items)
        counts.append(len(entries))
        if counts[-1] == expected:
            problem = _index_problem(entries, expected) if indexed else None
            if problem is None:
                if indexed:
                    entries.sort(key=lambda entry: entry.index)
                return replace(
                    result,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                )
            index_problems.append(problem)
            note = INDEX_RETRY_NOTE.format(problem=problem, items=items, expected=expected)
        else:
            note = RETRY_NOTE.format(got=counts[-1], items=items, expected=expected)
        request = f"{user}\n\n{note}"
    if counts[0] != expected or counts[1] != expected:
        raise MetricError(
            f"the judge returned {counts[0]} and then {counts[1]} {items} where {expected} were "
            f"asked for ({schema.__name__})"
        )
    raise MetricError(
        f"the judge's {items} for {schema.__name__} were misnumbered twice: "
        f"{index_problems[0]}; then {index_problems[1]}"
    )


def context_recall(
    question: str, expected_answer: str, contexts: list[str], judge: ChatModel
) -> MetricResult:
    """The share of the reference answer's sentences that the retrieved passages support.

    1.0 means the passages hold every fact the reference answer states; 0.5 means half of its
    sentences have no support in them, so no answer written from these passages could be
    complete.

    Each verdict carries the number of the sentence it is about, so a reply that lists them out
    of order is still matched to the right sentence instead of silently scoring the wrong one.
    `detail["sentences"]` lists each sentence with the judge's `index`, `supported` and `reason`.
    """
    sentences = split_sentences(expected_answer)
    if not sentences:
        raise ValueError("the expected answer has no sentences to look for")
    result = _ask_for_list(
        judge,
        system=CONTEXT_RECALL_SYSTEM,
        user=CONTEXT_RECALL_USER.format(
            question=question,
            count=len(sentences),
            sentences=numbered_lines(sentences),
            passages=numbered_passages(contexts),
        ),
        schema=SentenceVerdicts,
        items="verdicts",
        expected=len(sentences),
        indexed=True,
    )
    verdicts = result.parsed.verdicts
    return MetricResult(
        value=sum(verdict.supported for verdict in verdicts) / len(sentences),
        detail={
            "sentences": [
                {
                    "index": verdict.index,
                    "sentence": sentence,
                    "supported": verdict.supported,
                    "reason": verdict.reason,
                }
                for sentence, verdict in zip(sentences, verdicts, strict=True)
            ]
        },
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def context_precision(
    question: str, expected_answer: str, contexts: list[str], judge: ChatModel
) -> MetricResult:
    """How well the passages that lead to the reference answer were ranked.

    The judge marks each passage useful or not, and `average_precision` turns the marks, in rank
    order, into a score: 1.0 when every useful passage comes before every useless one, lower as
    useful passages sink down the list, and 0.0 when none is useful. With no passages there is
    nothing to judge, so the score is 0.0 and the judge is not asked.

    Each verdict carries the number of the passage it is about, so a reply that lists them out of
    order is still matched to the right passage — this is what stops a judge that says "useful,
    for $202.90" from having that verdict misattributed to whichever passage came first instead
    of the one the figure is actually in. `detail["passages"]` lists each passage's `rank` and
    the judge's `index`, `useful` and `reason`.
    """
    if not contexts:
        return MetricResult(value=0.0, detail={"passages": []}, input_tokens=0, output_tokens=0)
    result = _ask_for_list(
        judge,
        system=CONTEXT_PRECISION_SYSTEM,
        user=CONTEXT_PRECISION_USER.format(
            question=question,
            expected_answer=expected_answer,
            count=len(contexts),
            passages=numbered_passages(contexts),
        ),
        schema=PassageVerdicts,
        items="verdicts",
        expected=len(contexts),
        indexed=True,
    )
    verdicts = result.parsed.verdicts
    return MetricResult(
        value=average_precision([verdict.useful for verdict in verdicts]),
        detail={
            "passages": [
                {
                    "rank": rank,
                    "index": verdict.index,
                    "useful": verdict.useful,
                    "reason": verdict.reason,
                }
                for rank, verdict in enumerate(verdicts, start=1)
            ]
        },
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def faithfulness(question: str, answer: str, contexts: list[str], judge: ChatModel) -> MetricResult:
    """The share of the generated answer's factual claims that its passages support.

    Two judge calls: the first lists the answer's claims, the second checks each one against the
    passages. An answer that makes no claims, such as one saying the handbook does not give this,
    has nothing to check: its value is None and `detail["no_claims"]` is true. Whether it was
    right to say so is what abstention accuracy measures.

    Each verdict carries the number of the claim it is about, so a reply that lists them out of
    order is still matched to the right claim. `detail["claims"]` lists each claim with the
    judge's `index`, `supported` and `reason`.
    """
    listed = judge.complete_structured(
        system=FAITHFULNESS_CLAIMS_SYSTEM,
        user=FAITHFULNESS_CLAIMS_USER.format(question=question, answer=answer),
        schema=AnswerClaims,
    )
    claims = listed.parsed.claims
    if not claims:
        return MetricResult(
            value=None,
            detail={"no_claims": True, "claims": []},
            input_tokens=listed.input_tokens,
            output_tokens=listed.output_tokens,
        )
    checked = _ask_for_list(
        judge,
        system=FAITHFULNESS_VERDICTS_SYSTEM,
        user=FAITHFULNESS_VERDICTS_USER.format(
            passages=numbered_passages(contexts),
            count=len(claims),
            claims=numbered_lines(claims),
        ),
        schema=ClaimVerdicts,
        items="verdicts",
        expected=len(claims),
        indexed=True,
    )
    verdicts = checked.parsed.verdicts
    return MetricResult(
        value=sum(verdict.supported for verdict in verdicts) / len(claims),
        detail={
            "no_claims": False,
            "claims": [
                {
                    "index": verdict.index,
                    "claim": claim,
                    "supported": verdict.supported,
                    "reason": verdict.reason,
                }
                for claim, verdict in zip(claims, verdicts, strict=True)
            ],
        },
        input_tokens=listed.input_tokens + checked.input_tokens,
        output_tokens=listed.output_tokens + checked.output_tokens,
    )


def answer_relevance(
    question: str, answer: str, judge: ChatModel, embedder: Embedder, n_questions: int = 3
) -> MetricResult:
    """How closely the questions the answer responds to match the question that was asked.

    The judge, shown only the answer, writes `n_questions` questions it would answer completely.
    An answer that drifts from the question, or answers only part of it, leads to questions
    unlike the real one. The value is the mean cosine similarity between the embedding of each
    generated question and that of the real one. An answer the judge calls noncommittal, one
    that is evasive or says the information is not available, scores 0.0 however close its
    questions come.

    `detail["questions"]` lists each generated question with its `similarity`, and
    `detail["noncommittal"]` records the judge's call.
    """
    if n_questions < 1:
        raise ValueError(f"n_questions must be at least 1, got {n_questions}")
    result = _ask_for_list(
        judge,
        system=ANSWER_RELEVANCE_SYSTEM,
        user=ANSWER_RELEVANCE_USER.format(answer=answer, count=n_questions),
        schema=GeneratedQuestions,
        items="questions",
        expected=n_questions,
    )
    generated = result.parsed.questions
    vectors = embedder.embed_documents(generated)
    asked = embedder.embed_query(question)
    similarities = [cosine_similarity(vector, asked) for vector in vectors]
    noncommittal = result.parsed.noncommittal
    return MetricResult(
        value=0.0 if noncommittal else fmean(similarities),
        detail={
            "noncommittal": noncommittal,
            "questions": [
                {"question": text, "similarity": similarity}
                for text, similarity in zip(generated, similarities, strict=True)
            ],
        },
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
