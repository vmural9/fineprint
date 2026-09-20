# Plan: the whole series

One product, shipped in parts. Each part is a working increment, a repo tag, a README section, and
a 7 to 10 minute screen-recorded walkthrough video. The videos are a series about one product, so
continuity matters: the same golden set and the same scoreboard table appear at the end of every
part with new columns filled in.

This file is the plan. The build spec for a part lives in `docs/parts/NN-name.md` and is written
when we reach that part, not before. The rules that apply to every part are in `CLAUDE.md`.

## Done for every part

Every part carries these six items in addition to its own definition of done. Part 5 is optional;
if it is built, it carries them too.

- [ ] Its build spec is written **first**, in `docs/parts/NN-name.md`, detailed enough that a
      session with no other context can build from it.
- [ ] Its README section is written, and every number in it comes from the generated scoreboard
      block.
- [ ] `evals/scoreboard.md` and the README block are regenerated from committed results files.
- [ ] The repo is tagged `part-N`.
- [ ] Its video script is written in `docs/scripts/NN-name.md`, **after** the part is built.
- [ ] The owner has recorded the 7 to 10 minute walkthrough video from the script, and the README
      links to it.

## Part 1 — RAG service

PDF ingested, chunked, embedded into Postgres with pgvector. Hybrid retrieval (lexical search via
Postgres full-text search plus vector similarity, fused with reciprocal rank fusion; the brief
calls the lexical half "BM25", which decision D3 corrects). FastAPI service with `/ask` and
`/search` endpoints. LLM answer generation with a Pydantic-structured response (answer, cited
chunks with page numbers, confidence). A first version of the golden set (about 40 real questions a
Medicare beneficiary would ask, each with expected answer and expected source pages). A CLI that
runs the golden set and prints the scoreboard, even if the metrics in part 1 are just
exact-page-hit rate and a manual pass/fail column.

Build spec: [parts/01-rag-service.md](parts/01-rag-service.md). It carries thirteen ordered tasks
(Task 0 to Task 12) with acceptance criteria, the data model, the golden-set schema, the eval
design, and what the video must show. It also carries the eight build decisions, which were settled
on 2026-09-21 and recorded with their outcomes in that spec's decisions table; no task is waiting on
one. Two of them show up in this file: D1 and D2 put both model calls on AWS Bedrock, and D3
concerns the phrase "BM25 via Postgres full-text search" above. Both are in "Findings from setup
that affect the plan".

### Definition of done

The authoritative list is the spec's own "Definition of done". In summary:

- [ ] From a clean clone, `uv sync`, `docker compose up -d --wait`, the handbook download,
      `init-db`, `ingest`, `serve` and the three eval commands all work by following the README
      alone.
- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` pass; the
      integration tests pass with the database up.
- [ ] The golden set holds about 40 verified questions and the owner has approved it.
- [ ] `evals/scoreboard.md` and the README block are generated, `scoreboard --check` passes, and
      every number in the README comes from that block.
- [ ] The three retrieval configurations — hybrid, vector-only, lexical-only — have been run on the
      real corpus and their results files are committed.
- [ ] Plus everything under "Done for every part".

## Part 2 — Retrieval quality

RAGAS metrics (context recall, context precision, faithfulness, answer relevance) on the golden
set. Experiments: naive fixed-size chunking vs structure-aware chunking (the handbook has sections,
cross-references by page, and cost tables) vs adding a cross-encoder re-ranker. Scoreboard shows
each configuration side by side.

### Definition of done

- [ ] All four RAGAS metrics — context recall, context precision, faithfulness, answer relevance —
      are computed over the **full** golden set for **each** configuration, by a script in
      `evals/`, with the results files committed.
- [ ] At least the three configurations the brief names are on the scoreboard side by side: naive
      fixed-size chunking, structure-aware chunking, and structure-aware chunking plus a
      cross-encoder re-ranker. Configurations added beyond those three are welcome but do not
      replace them.
- [ ] Each experiment has a written finding that says what changed, by how much, and for which
      question types. "Structure-aware chunking raised page recall@5" is not a finding;
      "structure-aware chunking raised page recall@5 from X to Y, entirely on `table` and
      `multi_section` questions, and left `lookup` unchanged" is.
- [ ] The structure-aware chunker is registered by chunk-set name next to the fixed-size chunker,
      so both remain runnable and comparable, and both chunk sets stay in the database.
- [ ] Plus everything under "Done for every part".

## Part 3 — Observability and agent

Langfuse tracing on every LLM and tool call with cost and latency. A tool-using agent layer on top
of the retriever with two tools: a 2026 cost-figure lookup (premiums, deductibles, caps) and an
enrollment-window checker given a date. Provider retries, timeouts, and fallback. Cost and p95
latency per query added to the scoreboard.

### Definition of done

- [ ] Every LLM call and every tool call is traced, and each trace carries cost and latency. A
      trace from an eval run is shown end to end in the video.
- [ ] Exactly the two tools the brief names exist: a 2026 cost-figure lookup (premiums,
      deductibles, caps) and an enrollment-window checker given a date. Both have deterministic
      logic — no LLM inside the tool — and unit tests covering their boundaries, including dates
      on the edge of an enrollment window.
- [ ] Retries, timeouts and fallback are demonstrated by a fault-injection test that forces the
      provider to fail, time out, and then recover, and asserts what the service did in each case.
- [ ] Cost per query and p95 latency are filled into the scoreboard by a script from trace data,
      not typed in.
- [ ] Plus everything under "Done for every part".

## Part 4 — Eval gate

LLM-as-judge rubric calibrated against about 20 hand-labelled cases. Adversarial slice in the
golden set (prompt injection, out-of-scope questions, requests for personal medical advice,
prompt-leak attempts). Deterministic assertions on tool calls. promptfoo running in GitHub Actions
with pass/fail thresholds. Demo: a PR that degrades answers gets blocked; swapping the corpus from
the 2025 edition to the 2026 edition shows exactly which answers changed.

**A note, not a deliverable: CI will need AWS credentials.** Decisions D1 and D2 put both model
calls on AWS Bedrock, so even a retrieval-only eval run makes a Bedrock call to embed each
question. The GitHub Actions job therefore needs an AWS role that it assumes through OIDC (OpenID
Connect: GitHub signs a short-lived token for the workflow run and AWS trusts it, so no long-lived
access key is stored in the repository). The AWS account is shared with other projects, so that
role must be created through a CloudFormation stack named `fineprint-ci`, per the AWS rule in
`CLAUDE.md`. The trust policy, the permissions, and which Bedrock models the role may call belong
in the part 4 spec.

### Definition of done

- [ ] The judge is calibrated against about 20 hand-labelled cases and its agreement with the hand
      labels is reported as a number on the scoreboard detail. The manual pass/fail labels recorded
      during part 1's review step are the hand labels; they are not re-created.
- [ ] The adversarial slice exists in the golden set and covers all four attack types: prompt
      injection, out-of-scope questions, requests for personal medical advice, and prompt-leak
      attempts.
- [ ] Tool calls are asserted deterministically — which tool, which arguments — not judged.
- [ ] promptfoo runs in GitHub Actions with pass/fail thresholds, and the thresholds are committed
      values derived from measured baselines.
- [ ] A pull request that degrades answers is opened and shown blocked by the gate, with the
      failing run linked from the README section.
- [ ] The edition swap is run and produces a committed report listing exactly which answers changed
      between the 2025 and 2026 editions.
- [ ] Plus everything under "Done for every part".

## Part 5 — Optional

Online evaluation on sampled traffic with alerts on score drops.

### Definition of done

- [ ] A sample of live traffic is scored online by the part 4 judge, and the sampling rate and
      volume are recorded.
- [ ] An alert fires on a score drop, demonstrated by inducing a regression rather than by
      describing the mechanism.
- [ ] Plus everything under "Done for every part".

## Continuity across parts

The series is one product, not five demos. Three things stay fixed so that each part can be
compared with the one before it.

- **One golden set.** `evals/golden_set.jsonl` grows — part 2 does not change the questions, part 4
  adds an adversarial slice — but ids are never reused and never renumbered, and expected pages
  never change unless the corpus does. Every results file records the golden set's sha256 and its
  question count, so a comparison across parts can prove the two runs used the same questions.
- **One scoreboard, with fixed headline columns.** The column list is fixed for the whole series,
  so the same table closes every video with more cells filled in. The columns and which part fills
  each one are in "Scoreboard format" in [parts/01-rag-service.md](parts/01-rag-service.md). A cell
  with no measurement renders as `—`. `evals/scoreboard.md` is generated and never hand-edited.
- **One product.** Each part adds a layer to the same service rather than forking a new one. Part 1
  leaves the seams: a chunker registry and a `chunk_set` column for part 2, `LLMResult` fields for
  part 3's cost and latency, a single `answer_question()` entry point for part 3's tracing and
  agent, a `documents.edition` column for part 4's corpus swap.

## Findings from setup that affect the plan

Established on 2026-09-21, from the PDF, the package index, and live API calls rather than from
memory. They change what later parts must do.

**Both model calls go to AWS Bedrock, so one provider covers both.** The setup session assumed that
one provider could not cover both kinds of call, because the Anthropic API has no embeddings
endpoint. On AWS Bedrock it can, and that is what was decided. Answers come from Claude Opus 5
(model ID `us.anthropic.claude-opus-5`, region `us-west-2`, through the `anthropic` SDK's
`AnthropicBedrock` client) and embeddings from Amazon Titan Text Embeddings v2
(`amazon.titan-embed-text-v2:0`, 1024 numbers per passage, through boto3) — one provider, one set
of credentials, no Anthropic API key. Three consequences for the plan. Chunk vectors are 1024 wide,
which fixes the database column. Nothing in the project runs on a local embedding model any more,
so **a local embedder becomes a candidate experiment for part 2**, measured on the same golden set
as the chunking and re-ranker experiments. And no eval run is free or offline, because embedding a
question is a Bedrock call — which is the problem part 4's CI gate has to solve, noted above. The
endpoint and model-ID details behind this, all checked with live calls, are in the part 1 spec
under "Provider facts verified on 2026-09-21".

**The edition rolled over to 2027 before we started.** medicare.gov serves the handbook at one
unversioned URL, and since September 2026 that URL has served "Medicare & You 2027". CMS keeps no
public archive of prior-year handbooks, so the download script pins the 2026 bytes by sha256 and
fetches them from the Internet Archive, with a primary and a fallback snapshot. The 2025 edition is
pinned the same way behind `--edition 2025`, ready for part 4. Consequence for the plan: the corpus
is reproducible, but it now depends on a third-party archive, so part 4's CI job must cache the
file and retry rather than re-download it blindly. Decision D8 settled the edition on 2026-09-21:
the corpus stays on the 2026 edition, as briefed.

**The 2026 edition has two printings that disagree about money.** The September 2025 printing
prints 2025 dollar amounts throughout — a $185 Part B premium, a $1,676 Part A deductible — under a
cover that says 2026. The January 2026 printing prints the final 2026 amounts: $202.90 and $1,736.
Eighty-four of 128 pages differ between them. This is why the hash pin exists, and it strengthens
part 4: the corpus-swap demo is no longer an artificial exercise in re-ingestion but a
demonstration of a real failure mode, where a document that looks like the right one gives
last year's answers. Part 4 should say so when it reports which answers changed.

**Postgres full-text ranking is not BM25.** The brief describes the lexical half of hybrid
retrieval as "BM25 via Postgres full-text search". Built-in Postgres ranking (`ts_rank_cd`) is a
coverage-density score, not BM25; real BM25 needs an extension that the `pgvector/pgvector:pg16`
image does not ship. Decision D3 settled this on 2026-09-21: part 1 uses built-in full-text search
and calls it "lexical (Postgres FTS)" everywhere rather than claiming BM25. Consequence for the
plan: **true BM25 becomes a candidate experiment for part 2**, alongside the chunking and re-ranker
experiments, measured on the same golden set and the same scoreboard. If it is run, it needs a
database image that ships a BM25 extension, and that cost belongs in the part 2 spec.

**RAGAS pulls LangChain in as a hard dependency — a note for the part 2 spec.** Checked against
`https://pypi.org/pypi/ragas/json` on 2026-09-21: `ragas` 0.4.3 lists `langchain`, `langchain-core`,
`langchain-community` and `langchain_openai` in `requires_dist` with **no extras marker**, so they
install unconditionally, along with `openai`, `instructor`, `datasets`, `tiktoken`, `networkx`,
`scikit-network` and `pillow`. `CLAUDE.md` bans LangChain from application code unless a part calls
for it, and part 2 does not: it calls for RAGAS. So the part 2 spec must (a) put `ragas` in a
dev/eval dependency group rather than in the project's runtime dependencies, (b) forbid any import
of a LangChain package from `src/fineprint/`, so the ban holds where it matters, and (c) decide
which models compute the RAGAS metrics and how they are wired. RAGAS needs a judge model and an
embedding model of its own, and its hard dependency on `langchain_openai` and `openai` means its
defaults are OpenAI-shaped, while this project's models are both on AWS Bedrock (D1 and D2). Either
RAGAS is pointed at Bedrock or part 2 accepts a second provider for evaluation only. That needs an
answer before part 2 starts, not during it.
