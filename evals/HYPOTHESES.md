# Part 2 hypotheses

Written on 2026-09-26, before any part 2 run, from part 1's results in `evals/results/` and
`evals/scoreboard.md`. The part 2 findings in the README judge each one held, partly held or not
held, with numbers the scoreboard generator produced. Nothing here is edited after the runs.

## What part 1 measured

Hybrid retrieval over the `fixed-220w` chunk set scored 85.3% on page hit@5 (29 of 34 answerable
questions). It missed q019, q021, q034 and q035, all `lookup` questions, and q031, a `table`
question whose answer is on pages 11 and 12. `lookup` was the weakest type at 73.3%; `table` was
88.9% and `multi_section` 100%. Vector-only retrieval beat the hybrid by one question, q034, whose
right page the hybrid had at fused rank 6, one below the cut.

## Hypotheses

- **H1 — the `sections` chunker.** Cutting at headings and keeping tables whole raises
  `page_recall@5` on `table` and `multi_section` questions and leaves `lookup` within one question
  of the baseline, because those two types need a whole table or a whole passage that a 220-word
  window cuts in two. Prediction for q031: found.
- **H2 — the re-ranker.** Re-ranking the fused candidates fixes lookups whose expected page is
  already in the pool at ranks 6 to 20, and does not move questions whose page is absent from the
  pool; the candidate diagnostic (`evals/candidates.py`) says how many of each there are.
  Prediction for q034: found at rank 5 or better.
- **H3 — the two together.** `sections+rerank` is the best row on `page_hit@5` and on context
  recall, and its gain is additive to within one question of the two effects measured separately.
- **H4 — faithfulness.** Faithfulness is above 0.90 for every row, because the answer prompt
  forbids answering outside the passages and every citation is verified; where it is lower, the
  failing claims are figures copied with a different year or unit.
- **H5 — context precision.** Context precision is higher for `sections` than for `fixed-220w`
  at the same k, because overlapping fixed windows put near-duplicate passages into the top 5.

# Part 3 hypotheses

Written before any part 3 run, from the part 2 scoreboard and the golden set. Part 3 re-runs
`sections` and `sections+rerank` with tracing on and adds the `agent` configuration.

- **H1 — where the time goes.** On `sections+rerank`, model generation is more than 80% of a
  question's time at the median, and the retriever step (embedding, SQL and re-rank together) is
  under 2 seconds at p95. Judged from the scoreboard's "Time by step" table.
- **H2 — which questions call tools.** The agent calls `lookup_cost_figure` on every question
  whose expected answer states a 2026 amount the figure table holds (q001, q002, q011, q013, q014,
  q023, q035, q038), calls `check_enrollment_window` on every question that gives a birthday and
  asks for dates worked out from it (q006, q008), and calls no tool on the six unanswerable
  questions (q003, q012, q026, q033, q037, q040). The lists were computed from the golden set's
  expected answers and the figure table. Judged from the scoreboard's "Tool calls" table.
- **H3 — what the tools cost.** The `agent` row's cost per query is at least 1.5 times
  `sections+rerank`'s and its p95 latency at least 1.5 times, because every tool question takes two
  or more model turns. Its page hit@5 equals `sections+rerank`'s (same retrieval).
- **H4 — faithfulness.** The `agent` row's faithfulness is at or above `sections+rerank`'s, because
  the figures it states come from a table checked against the handbook, and each tool result is
  scored as context exactly as the model received it.

Each is judged held, partly held or not held in the README's findings, with the numbers.
