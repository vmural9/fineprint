"""The results file: everything one eval run saw, written to `evals/results/<run_id>.json`.

A published number has to be traceable to the run that produced it, so a results file records
more than the scores. It records the commit the code was on and whether the tree was dirty, the
whole configuration, the corpus and golden-set hashes, and, per question, the chunks that came
back with their ranks, the metrics, the full answer, and the human verdict. The scoreboard
recomputes every aggregate from those per-question rows and stores no summary of its own.

`run_id` is `<UTC timestamp>_<config name>`, for example `20260921T031500Z_hybrid`, which sorts
by time and names the configuration in the file listing. A configuration name may contain `+`
(`hybrid+rerank`), which is a valid character in both a file name and this join.

Every field part 2 adds defaults to `None`, so the three results files part 1 committed —
written before any of them existed — still load unchanged; a later scoring pass is what fills
them in, over a file `run_golden_set` already wrote (decision D4).
"""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.metrics import Aggregates, QuestionMetrics, aggregate

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
# How long to wait for git before deciding the provenance is not worth blocking a run for.
GIT_TIMEOUT_SECONDS = 10


class ResultsError(ValueError):
    """Raised when a results file cannot be read."""


@dataclass(frozen=True, slots=True)
class GitState:
    """Which commit the run was made from, and whether anything was uncommitted."""

    commit: str | None
    dirty: bool


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The knobs that were set for this run. Changing any of them makes a different row."""

    name: str  # hybrid, vector-only, lexical-only, hybrid+rerank, sections, sections+rerank
    mode: str  # hybrid, vector, lexical
    chunk_set: str
    top_k: int
    candidates: int
    rrf_k: int
    embedding_model: str
    llm_model: str | None  # None on a --retrieval-only run: no answer model was called
    reranker: str | None = None  # the Bedrock re-ranker model id, or None: this run had none
    rerank_candidates: int | None = None  # candidates it read; None when this run had no re-ranker


@dataclass(frozen=True, slots=True)
class Corpus:
    """Which handbook the run searched, pinned by the hash of the PDF that was ingested."""

    edition: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GoldenSetInfo:
    """Which golden set the run used, pinned by the hash of the file."""

    sha256: str
    count: int


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk the retriever returned: where it sits in the handbook and where it ranked.

    From part 2 on (decision D5) this also carries the chunk's own text, its ordinal and its
    section heading, because the scoring pass needs the text to judge and reads it only from
    this file — never from the database. `ordinal` survives a re-ingest where `chunk_id` does
    not. `fused_rank` is the chunk's place after fusion and before any re-ranking (`None` outside
    hybrid mode, and equal to `rank` when nothing re-ranked the list); `rerank_score` is `None`
    when no re-ranker ran. A file written before part 2 has all five fields `None`.
    """

    chunk_id: int
    page_start: int
    page_end: int
    rank: int
    score: float | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    ordinal: int | None = None
    section: str | None = None
    text: str | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class Review:
    """What a person decided about one generated answer, once they have looked at it."""

    verdict: str | None = None  # "pass", "fail", or None while nobody has looked
    note: str = ""
    reviewer: str | None = None
    reviewed_at: str | None = None


@dataclass(slots=True)
class QuestionResult:
    """One golden-set question's row: what came back, what it scored, what a person thought."""

    id: str
    question: str
    type: str
    category: str
    difficulty: str
    answerable: bool
    expected_pages: list[int]
    retrieved: list[RetrievedChunk]
    metrics: QuestionMetrics
    answer: dict[str, Any] | None = None  # the whole AnswerResponse, or None if not answered
    review: Review = field(default_factory=Review)
    error: str | None = None  # set when this question failed; the run carried on without it
    # The judge's verdicts behind this question's four RAGAS metrics — the sentences, claims and
    # per-passage calls a scoring pass made — kept for `--explain` and for anyone auditing a
    # score. `None` until this question has been scored.
    scoring_detail: dict[str, Any] | None = None

    @property
    def answered(self) -> bool:
        """True when an answer was generated for this question."""
        return self.answer is not None

    @property
    def verdict(self) -> str | None:
        """The human verdict, lifted out of the review so aggregation can read it directly."""
        return self.review.verdict


@dataclass
class Scoring:
    """Provenance for a scoring pass: which judge, which prompts, and what it cost.

    Set once a scoring pass has scored every question it could; `None` on a file that has not
    been through that pass yet, including every file part 1 committed.
    """

    judge_model: str
    embedding_model: str
    prompts_sha256: str  # sha256 of the prompt texts in evals/ragas_metrics.py
    scored_at: str  # UTC, same format as created_at
    judge_input_tokens: int
    judge_output_tokens: int


@dataclass(slots=True)
class RunResult:
    """One run of one configuration over the golden set."""

    run_id: str
    created_at: str
    git: GitState
    config: RunConfig
    corpus: Corpus
    golden_set: GoldenSetInfo
    questions: list[QuestionResult]
    scoring: Scoring | None = None

    @property
    def aggregates(self) -> Aggregates:
        """The run's numbers, averaged from its per-question rows every time they are asked for."""
        return aggregate(self.questions)

    @property
    def errors(self) -> list[QuestionResult]:
        """The questions that failed. A failure is recorded, not fatal."""
        return [question for question in self.questions if question.error]


def now_utc() -> str:
    """The moment something happened, spelled the way a results file spells it."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(config_name: str, when: datetime | None = None) -> str:
    """Build `<UTC timestamp>_<config name>`, the name of one run and of its file."""
    moment = when or datetime.now(UTC)
    return f"{moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT)}_{config_name}"


def results_path(run_id: str, directory: Path = RESULTS_DIR) -> Path:
    """Where the results file for a run belongs."""
    return directory / f"{run_id}.json"


def save(result: RunResult, path: Path) -> Path:
    """Write the run to `path`, creating the results directory when it is missing.

    The file is written beside its destination and then moved into place, so an interrupted
    save — the review tool saves after every verdict — cannot leave a half-written file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def load(path: Path) -> RunResult:
    """Read one results file.

    Raises:
        ResultsError: if the file is not JSON, or is missing a field the scoreboard needs.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultsError(f"{path}: cannot be read: {error}") from error
    if not isinstance(record, dict):
        raise ResultsError(f"{path}: expected a JSON object")

    try:
        scoring = record.get("scoring")
        return RunResult(
            run_id=_field(record, "run_id"),
            created_at=_field(record, "created_at"),
            git=GitState(**_field(record, "git")),
            config=RunConfig(**_field(record, "config")),
            corpus=Corpus(**_field(record, "corpus")),
            golden_set=GoldenSetInfo(**_field(record, "golden_set")),
            questions=[_question(row) for row in _field(record, "questions")],
            scoring=Scoring(**scoring) if scoring is not None else None,
        )
    except (KeyError, TypeError) as error:
        raise ResultsError(f"{path}: not a results file: {error}") from error


def load_all(directory: Path = RESULTS_DIR) -> list[RunResult]:
    """Read every results file in the directory, oldest run first."""
    if not directory.is_dir():
        return []
    results = [load(path) for path in sorted(directory.glob("*.json"))]
    return sorted(results, key=lambda result: (result.created_at, result.run_id))


def latest_per_config(directory: Path = RESULTS_DIR) -> dict[str, RunResult]:
    """The most recent run of each configuration, keyed by configuration name.

    This is what the scoreboard shows: one row per configuration, the last time it was measured.
    The name inside the file decides, not the file name, so renaming a file changes nothing.
    """
    latest: dict[str, RunResult] = {}
    for result in load_all(directory):  # oldest first, so the last one to land wins
        latest[result.config.name] = result
    return latest


def sha256_of(path: Path) -> str:
    """The sha256 of a file's bytes, the way the download script pins the handbook."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo_root: Path = REPO_ROOT) -> GitState:
    """The commit the run was made from, and whether the tree had uncommitted changes.

    Provenance is worth recording but never worth failing a run for: outside a repository, or
    with no git on the machine, this says so with `commit=None` instead of raising.
    """
    commit = _git(repo_root, "rev-parse", "HEAD")
    if commit is None:
        return GitState(commit=None, dirty=False)
    return GitState(commit=commit, dirty=bool(_git(repo_root, "status", "--porcelain")))


def _git(repo_root: Path, *arguments: str) -> str | None:
    """Run one git command in the repository and return its output, or None if git could not."""
    try:
        finished = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - needs git to be missing
        return None
    return finished.stdout.strip() if finished.returncode == 0 else None


def _field(record: dict[str, Any], name: str) -> Any:
    """Read a required top-level field, naming it when it is absent."""
    if name not in record:
        raise KeyError(f"{name!r} is missing")
    return record[name]


def _question(row: Any) -> QuestionResult:
    """Rebuild one question's row from its JSON object."""
    if not isinstance(row, dict):
        raise TypeError(f"expected a question object, got {type(row).__name__}")
    return QuestionResult(
        id=_field(row, "id"),
        question=_field(row, "question"),
        type=_field(row, "type"),
        category=_field(row, "category"),
        difficulty=_field(row, "difficulty"),
        answerable=_field(row, "answerable"),
        expected_pages=list(_field(row, "expected_pages")),
        retrieved=[RetrievedChunk(**chunk) for chunk in _field(row, "retrieved")],
        metrics=QuestionMetrics(**_field(row, "metrics")),
        answer=row.get("answer"),
        review=Review(**row.get("review", {})),
        error=row.get("error"),
        scoring_detail=row.get("scoring_detail"),
    )
