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
