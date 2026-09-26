# Golden set

`golden_set.jsonl` holds the evaluation questions for the Medicare Q&A service: one JSON
object per line, written in the voice of a beneficiary or their adult child, with the pages
of the handbook a correct answer needs. Every figure in it comes from the pinned corpus
(Medicare & You 2026, January 2026 printing) and not from anyone's memory of Medicare.

## Schema

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | `q001`, `q002`, … Unique, never reused or renumbered. |
| `question` | string | How a beneficiary or their adult child would really ask it. |
| `expected_answer` | string | What a correct answer must say, with exact amounts and conditions as printed, in 2–4 sentences a human grader can mark pass or fail against. |
| `expected_pages` | list of int | Every page a correct answer needs, and only those. Empty exactly when `type` is `unanswerable`. Page numbers are printed page numbers, which equal 1-based PDF page indices. |
| `category` | string | `costs`, `coverage`, `enrollment`, `original_vs_advantage`, `part_d`, `medigap` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `type` | string | `lookup`, `table`, `multi_section`, `unanswerable` |
| `evidence` | list of `{page, quote}` | Verbatim spans supporting the expected answer. At least one per expected page; quotes on other pages are allowed. |
| `alt_pages` | object, optional | Maps one expected page (a string key, because JSON keys are strings) to the list of other pages that state the same fact. Omit the field when the item has no alternates. |
| `needs_review` | bool | True when a page or the answer could not be fully verified. |

### How `alt_pages` is scored

`alt_pages` exists so that retrieving an equally good page is not punished. A flat list
could not do that job: `page_hit@5` credited the alternates while `page_recall@5` could
not, so one retrieval could score 1.0 on the first metric and 0.0 on the second. The map
ties each alternate to the expected page it stands in for:

```json
"expected_pages": [27, 30], "alt_pages": {"30": [23]}
```

- An expected page **counts as found** when a retrieved chunk covers it *or* covers one of
  the pages listed under it.
- `page_hit@5` is 1 when **at least one** expected page was found.
- `page_recall@5` is the **share** of expected pages found.

Every key must be one of the item's own `expected_pages`, every value must be a non-empty
list of page numbers, and no alternate may itself be an expected page. `verify_golden_set`
checks that alternate page numbers exist in the PDF.

## Authoring rules

- Every amount, date, rule and page number comes from the extracted text of the pinned PDF.
  When the text and your memory of Medicare disagree, the text wins and the disagreement is
  worth writing down: the September 2025 printing of this same edition prints 2025 amounts.
- `expected_pages` lists only the pages a correct answer needs. Then search the whole
  corpus for other places that state the same fact and record them under the expected page
  they stand in for; an eval that punishes retrieving an equally good page is a bad eval. A
  page that only mentions the topic, or that restates nothing specific, is not an alternate.
- Quotes are short (under about 200 characters) and copied from the extracted text, not from
  the rendered PDF. For a `table` question, quote the row the way pypdf extracts it, since
  that is what the retriever will see.
- An `unanswerable` question must be confirmed unanswerable by searching the whole corpus.
  Its expected answer says the handbook does not give this and names the right resource. Its
  `evidence` quotes come from the page where the handbook sends the reader elsewhere — the
  page that says the figure or the list is not here and names Medicare.gov, SSA, or the plan.
  That page is **not** an expected page: `expected_pages` stays empty, because retrieving it
  is not what the item measures. Retrieval metrics skip unanswerable items; what they
  measure is abstention.
- Avoid trivia about the document, questions that turn on facts the question does not give,
  yes/no questions with no substance, and two questions that test the same fact.
- Set `needs_review: true` on anything you could not fully verify, and say why. An honest
  flag is better than a confident guess.

## Commands

```bash
uv run pytest tests/test_golden_set.py     # schema, validator, and the evidence quotes
uv run python -m evals.verify_golden_set   # re-extract the cited pages and check every quote
```

`verify_golden_set` prints one line per item and a summary. It exits 1 when a quote or a
page number does not check out, and 2 when the handbook PDF is missing, naming
`scripts/download_handbook.py`. Pass `--pdf PATH` to check against a different copy.

## Configurations

`evals/configs.py` registers six named configurations, and `run_golden_set --config` names one of
them. Each is a fixed bundle of which chunk set to search, which retrieval mode to search it in,
and whether a re-ranker re-orders the fused list before the top-5 cut:

| Configuration | Chunk set | Mode | Re-ranker |
|---|---|---|---|
| `hybrid` | `fixed-220w` | hybrid | — |
| `vector-only` | `fixed-220w` | vector | — |
| `lexical-only` | `fixed-220w` | lexical | — |
| `hybrid+rerank` | `fixed-220w` | hybrid | Cohere Rerank 3.5 |
| `sections` | `sections` | hybrid | — |
| `sections+rerank` | `sections` | hybrid | Cohere Rerank 3.5 |

`fixed-220w` cuts the handbook into fixed-size, roughly 220-word chunks; `sections` cuts at the
handbook's own headings instead, so a table or a list stays whole in one chunk. `hybrid` fuses
lexical and vector search by reciprocal rank fusion; `vector-only` and `lexical-only` run one
retriever alone, which is how the scoreboard shows what fusion adds. The three names on the left
are part 1's, kept exactly so their committed results files still key to the same rows.

`settings_for(config, base)` turns a configuration into the `Settings` a run makes its Postgres
and Bedrock calls with: it sets the chunk set, and turns the re-ranker on with its model for the
two `+rerank` configurations — the other four (`hybrid`, `vector-only`, `lexical-only` and
`sections`) run with the re-ranker off. Everything else `base` already had — the database, the
region, how many fused candidates a re-ranker reads — is left alone.

## Running an eval

Run, review, regenerate — in that order. They need an ingested corpus, and the runs call AWS
Bedrock: every mode that searches by meaning embeds each question, a configured re-ranker sends
one more request per question, and any run without `--retrieval-only` also sends the retrieved
chunks to the answer model. `lexical-only --retrieval-only` is the one command that calls no
model at all.

```bash
uv run python -m evals.run_golden_set --config hybrid                     # search, answer, score
uv run python -m evals.run_golden_set --config hybrid+rerank
uv run python -m evals.run_golden_set --config sections
uv run python -m evals.run_golden_set --config vector-only --retrieval-only
uv run python -m evals.run_golden_set --config lexical-only --retrieval-only   # no model at all
uv run python -m evals.review evals/results/<run_id>.json                 # your pass or fail
uv run python -m evals.scoreboard                                         # regenerate the tables
```

- **`run_golden_set`** writes `evals/results/<run_id>.json`, where `run_id` is
  `<UTC timestamp>_<config name>` (a name may itself contain a `+`, as in `hybrid+rerank`). The
  file records the commit and whether the tree was dirty, the whole configuration — including,
  from part 2 on, the re-ranker model and how many candidates it read, when one ran — the corpus
  and golden-set hashes, and, per question, every chunk that came back with its text, its
  ordinal, its section heading and its ranks, the metrics, the full answer, and the manual
  verdict. One question failing is recorded on that question; the run carries on. Only files
  directly under `evals/results/` are runs; a subdirectory such as `evals/results/candidates/`
  (the candidate-rank diagnostic's own output, below) never is.
- **`review`** shows each answered question with its expected answer, the generated answer, the
  pages it cited and whether each quote was verified, and reads `p`, `f`, `s` or `q` plus an
  optional note. It saves after every verdict. `--only-unreviewed` picks up where you left off.
- **`scoreboard`** regenerates `evals/scoreboard.md` and the block between
  `<!-- scoreboard:start -->` and `<!-- scoreboard:end -->` in the repository `README.md` from
  the latest results file of each configuration. Once a results file carries a scoring pass, its
  four RAGAS columns fill in on the headline table, and `evals/scoreboard.md` gains their own
  breakdown by question type (`page_recall@5`, `context_recall` and `faithfulness`, alongside
  `page_hit@5`); a row with a re-ranker also gets a "Re-ranker movement" table, showing how many
  of its answerable questions the re-ranker lifted into the top 5, left there already, or could
  not find an expected page for at all; and the provenance table names that row's re-ranker and
  judge model. `--check` writes nothing and exits non-zero when either file is out of date, which
  is what stops a hand edit. Neither file is ever edited by hand: if a number is wrong, fix the
  results or the generator and run it again.

The first six are defined in `metrics.py`, one short function each; the last four are defined in
`ragas_metrics.py`, one function each:

| Metric | What it asks |
|--------|--------------|
| `page_hit@5` | Did any of the 5 retrieved chunks cover a page the answer needs? |
| `page_recall@5` | What share of the pages the answer needs did they cover? |
| `mrr` | How far down the list was the first chunk that covered one? |
| `cited_page_hit` | Did the answer's own citations land on a page the answer needs? |
| `abstention_accuracy` | Was "the handbook does not say this" right, on every answered question? |
| `manual_pass` | Of the answers a person read, what share were right? |
| `context_recall` | What share of the expected answer's sentences do the retrieved passages support? |
| `context_precision` | Of the retrieved passages, how many actually help reach the expected answer, weighted by rank? |
| `faithfulness` | What share of the generated answer's own claims are supported by the retrieved passages? |
| `answer_relevance` | How closely do questions written back from the answer match the one actually asked? |

Retrieval metrics are computed over the answerable questions only: an `unanswerable` item has
no expected pages, and what it measures is abstention. A metric that does not apply to a
question is left empty rather than scored zero, so it never drags an average down.

The last four are judged by a language model rather than computed from page numbers. A results
file carries a field for each, per question, and a `scoring` block naming the judge model, the
embedding model `answer_relevance` compares with, a hash of the prompts that judged it, and when
it ran — all `null` until a scoring pass has filled them in, the same way `manual_pass` prints
`—` before anyone has reviewed an answer. Each question also carries a `scoring_detail` object
for the judge's own verdicts behind its four scores — which sentences and claims it found, and
why — empty until then.

## Scoring a run

`python -m evals.score` is the second pass over a results file. After `run_golden_set` has written
a run, and before `scoreboard` regenerates the tables, it has a judge model score every question on
the four RAGAS metrics above and writes the scores into the same file:

```bash
uv run python -m evals.score evals/results/<run_id>.json                  # score what has no score yet
uv run python -m evals.score evals/results/<run_id>.json --explain q001   # one question's working
uv run python -m evals.score evals/results/<run_id>.json --rescore        # score every question again
```

The judge is Claude Sonnet 5 (the `JUDGE_MODEL` setting; `--judge-model ID` picks another for one
pass), a different model from the one that writes the answers, so no answer is scored by its own
author. Answer relevance embeds with the same Titan model retrieval searches with. The pass calls
Bedrock: an answered question takes five judge calls — one each for recall, precision and
relevance, two for faithfulness — and four embeddings, and more calls when a reply has to be asked
for again.

What a question is scored on depends on what the run recorded for it:

| The question | Context recall and precision | Faithfulness and answer relevance |
|---|---|---|
| is `unanswerable` | — | — |
| has no answer (a `--retrieval-only` run) | scored | — |
| was answered | scored | scored |

An unanswerable question's reference answer says the handbook does not give this: there are no
facts for retrieval to find and no claims to check, and abstention accuracy is its measure. An
answer that makes no factual claims at all, such as "the handbook does not say", has no
faithfulness score either. A question whose retrieval failed during the run is skipped.

- **Passages come from the file.** The judge reads each retrieved chunk's text as the results file
  recorded it, never from the database, whose chunks can be re-ingested after a run. A file from
  before results files carried chunk text, such as the three part 1 runs, is refused, and the
  message names the `run_golden_set` command that makes a scorable one.
- **Saved after every question.** An interrupted pass loses nothing: run the same command again
  and it carries on, leaving the questions that already have scores alone. `--rescore` clears
  every score in the file and scores every question again.
- **The working is kept.** Each question's `scoring_detail` holds the judge's verdicts: every
  sentence of the reference answer and whether the passages support it, every passage and whether
  it is useful, every claim in the answer and whether it is supported, and the questions written
  back from the answer with their similarity to the one asked, plus the judge tokens the question
  took. The file's `scoring` block names the judge model, the embedding model and the sha256 of the
  prompts in `ragas_metrics.py`, with the token totals behind the file's scores. Scores from a
  different judge, embedding model or prompt text are never added to a scored file: the pass asks
  for `--rescore` instead.
- **A judge failure stays with its question.** A reply that will not validate, or a count of
  verdicts still wrong after one retry, is recorded as an `error` in that question's
  `scoring_detail`; its four scores stay empty and the pass goes on. The next pass tries it again.
- **`--explain QID`** prints one question's working in plain text: each sentence of the expected
  answer with the judge's verdict and reason, each passage with its verdict, each claim in the
  answer with its verdict, the questions written back with their similarities, and the four
  numbers. A scored question is explained from the file with no model call; one not yet scored is
  judged afresh, and `--rescore` forces that. It never writes to the file.

Exit codes: `0` when every question was scored or skipped; `1` when the judge failed on at least
one question, which the summary names; `2` when the file cannot be scored as it stands, in which
case nothing is called and nothing is written.

## Candidate-rank diagnostic

A re-ranker can only reorder chunks retrieval already put in front of it. Before crediting the
re-ranker experiment with a page it recovered, it is worth knowing whether that page was ever
reachable at all. `python -m evals.candidates` answers that, independently of whether a
re-ranker is even configured:

```bash
uv run python -m evals.candidates                                    # Settings().chunk_set, hybrid
uv run python -m evals.candidates --chunk-set fixed-220w --mode hybrid
uv run python -m evals.candidates --chunk-set sections --mode hybrid
```

For every answerable question, it calls `Retriever.candidates()` — the full ordered pool exactly
as fusion left it, before the top-5 cut and before any re-ranking — and finds the 1-based pool
rank of the first chunk whose page range covers an expected page (or one of the alternate pages
recorded for it in `alt_pages`; see "How `alt_pages` is scored" above). The pool depends on only
`--chunk-set` and `--mode`, since re-ranking and the top-k cut both happen after `candidates()`
returns, which is why this command takes those two directly rather than an evaluation
configuration name.

Each question's rank is put in one of four buckets:

| Bucket | Meaning |
|--------|---------|
| `1-5` | Already where `fineprint search` would return it today; a re-ranker changes nothing here. |
| `6-20` | In the pool, past the top 5 — exactly what raising `RERANK_CANDIDATES` and turning on the re-ranker could promote. |
| `21+` | Deeper still in the pool, open-ended — the exact rank is in `first_hit_rank`. |
| `absent` | Not retrieved at all, at any rank the pool reached. No re-ranker can fix this one; the chunking or the retriever has to change instead. |

The command prints one row per question type (`lookup`, `table`, `multi_section`) plus an `all`
row, each with its four bucket counts, then one line per question outside the top 5, naming its
rank and its expected pages. `--mode lexical` calls no model; `hybrid` and `vector` each embed
every question once, the same one Bedrock call per question as any other retrieval-only run.

It writes to its own subdirectory, apart from the run results above, since its files carry no
`config` field and `evals.results.load_all` would refuse them as a run:
`evals/results/candidates/<UTC timestamp>_candidates-<chunk_set>-<mode>.json`:

| Field | Meaning |
|-------|---------|
| `run_id` | `<UTC timestamp>_candidates-<chunk_set>-<mode>`. |
| `created_at` | When the run finished, UTC. |
| `git` | `{commit, dirty}` — the commit the run was made from. |
| `chunk_set` | Which chunk set the pool was drawn from. |
| `mode` | `hybrid`, `vector`, or `lexical`. |
| `pool_size` | `RETRIEVAL_CANDIDATES`, the cap each retriever's own query ran with. |
| `corpus` | `{edition, sha256}` — which handbook was searched. |
| `golden_set` | `{sha256, count}` — which golden set was used. |
| `questions` | One row per answerable question: `{id, type, expected_pages, first_hit_rank, bucket}`. `first_hit_rank` is `null` exactly when `bucket` is `"absent"`. |

Unanswerable questions have no expected page to look for, so they are skipped outright, the same
way the retrieval metrics skip them.
