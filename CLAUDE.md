# CLAUDE.md — standing brief

Read this before touching anything in this repo. It is the owner's brief, condensed. Where a task
needs more detail than this file gives, the spec for the current part in `docs/parts/` is the
authority.

## The product

`fineprint` answers questions about Medicare from the official U.S. government handbook
"Medicare & You 2026" (CMS publication 10050, public domain), and cites the page each answer came
from. The chatbot is not the point. The point is production RAG engineering the way senior GenAI
roles describe it: hybrid retrieval, retrieval quality measured against a golden set, observability
on every call, LLM-as-judge evals, and an eval gate in CI. Every part ends with measured numbers,
and those numbers go into the scoreboard whether they flatter the system or not.

## The series

One product, five parts. `docs/PLAN.md` holds the full plan and a definition of done per part.

- **Part 1 — RAG service.** Ingest, chunk, embed into Postgres with pgvector; hybrid retrieval;
  FastAPI `/ask` and `/search`; structured cited answers; golden set v1; the first scoreboard row.
  Spec: `docs/parts/01-rag-service.md`.
- **Part 2 — Retrieval quality.** RAGAS metrics; chunking and re-ranker experiments side by side.
- **Part 3 — Observability and agent.** Langfuse tracing; a two-tool agent; retries, timeouts,
  fallback; cost and p95 latency.
- **Part 4 — Eval gate.** Calibrated LLM judge; adversarial slice; promptfoo in GitHub Actions.
- **Part 5 — Optional.** Online evaluation on sampled traffic with alerts on score drops.

**Workflow per part.** Write the part's spec in `docs/parts/NN-name.md` when we reach that part and
not before. Build it. Add its README section. Regenerate the scoreboard. Tag `part-N`. Then write
the video script in `docs/scripts/NN-name.md` — after the part is built, never before.

**Issue tracking.** Work is tracked in GitHub issues at https://github.com/vmural9/fineprint: one
parent issue per part and one child issue per task of the current part. Reference the issue in
commits (`Refs #N`) and close it with `Closes #N`.

**AWS.** Use the `merlion-brands` profile, and create resources only through a stack whose name
starts with `fineprint-`. The account is shared with other projects. Part 1 needs AWS: both model
calls go to Bedrock. Part 4's CI eval gate will need a role that GitHub Actions assumes through
OIDC, created through a CloudFormation stack named `fineprint-ci`.

## Current status (2026-09-21)

Setup is done: the uv project, the docker compose Postgres, `scripts/download_handbook.py`, and the
first 15 golden-set questions.

**The eight build decisions are settled.** They were taken on 2026-09-21 and their outcomes are
recorded in the decisions table in `docs/parts/01-rag-service.md`. Nothing is blocked on an open
decision. The one with consequences everywhere: both model calls — answers and embeddings — go to
AWS Bedrock, so read "Provider facts you must not get wrong" below before writing any provider code.

**Part 1 is under way**, starting with Task 1 (settings, database plumbing, schema). No eval has
been run yet, and there are no results and no scoreboard file.

## Stack and conventions

- Python 3.12, uv for dependency management, ruff for lint and format, pytest for tests.
- FastAPI, Postgres 16 with pgvector, docker compose for local Postgres.
- LLM calls go through a thin provider abstraction so the model can be switched by config. Default
  to one provider now; do not spend effort on multi-provider until part 3. Embeddings go through
  the same abstraction, and since both calls go to AWS Bedrock, one provider really does cover both.
- Conventional commit messages. Tag the end of each part as `part-N`.

## Hard rules

These are not style preferences. Breaking one invalidates what the project claims to demonstrate.

1. **No number goes into `README.md` or the scoreboard unless a script in `evals/` produced it.**
   No placeholder metrics anywhere — not a "TBD", not a rounded guess, not an illustrative table
   with numbers in it. A measurement that does not exist renders as `—`, which means absent, never
   a stand-in value.
2. **`evals/scoreboard.md` is generated and never hand-edited.** The same holds for the block
   between `<!-- scoreboard:start -->` and `<!-- scoreboard:end -->` in `README.md`. If a number
   there is wrong, fix the results or the generator and re-run it.
3. **Secrets only through environment variables and `.env`.** `.env` is never committed;
   `.env.example` is committed with no real values. No secret has a default in code.
4. **Keep the code plain and readable.** Hiring managers read this repo. Prefer clarity over
   cleverness, and prefer the standard library or a well-known package over a framework that hides
   what is happening. No LangChain and no LlamaIndex in `src/fineprint/` unless a part's spec
   calls for one by name.
5. **One LLM provider behind a thin abstraction until part 3.** Retries, fallback, and any second
   provider are part 3 work, not part 1 work.
6. **Golden-set pages come from the PDF and are never guessed.** Verify every expected page and
   every evidence quote against the extracted text. Anything unverified carries
   `needs_review: true`.
7. **Conventional commits**, one logical change per commit, with the tests in the same commit as
   the code they cover.
8. **Nothing written here may imply that something exists when it does not.** No links to tags,
   videos, endpoints, or results that have not been produced.

## Provider facts you must not get wrong

Decided and checked with live calls on 2026-09-21. `docs/parts/01-rag-service.md` carries the
detail, under the decisions table and "Provider facts verified on 2026-09-21". Do not replace any
of these from memory; re-check with a live call first.

- **Everything goes to AWS Bedrock in `us-west-2`**, under the `merlion-brands` profile, through
  the standard AWS credential chain. **There is no Anthropic API key in this project**, and
  `ANTHROPIC_API_KEY` is not a setting. Locally, `AWS_PROFILE` in `.env` is what supplies
  credentials.
- **Answers:** Claude Opus 5, model ID `us.anthropic.claude-opus-5`, called with the `anthropic`
  SDK's `AnthropicBedrock` client (install `anthropic[bedrock]`). The `us.` prefix marks a regional
  inference profile — an ID that routes the request across a group of regions. The bare
  `anthropic.claude-opus-5` is refused with "on-demand throughput isn't supported".
- **Use `AnthropicBedrock`, the standard runtime endpoint, not `AnthropicBedrockMantle`.** In
  `us-west-2` the Mantle endpoint serves Claude Haiku 4.5 but returns 404 for Opus 5 and Sonnet 5.
- **The API's built-in structured output does not work on this path.**
  `client.messages.parse(..., output_format=...)` fails with 400 `output_config.format: Extra inputs
  are not permitted`. Validated structures come from tool use plus Pydantic validation instead; the
  part 1 spec's Task 7 says exactly how.
- **Embeddings:** `amazon.titan-embed-text-v2:0` through boto3's `bedrock-runtime` `invoke_model`,
  1024 dimensions, normalized. So `chunks.embedding` is `vector(1024)`, and changing the embedding
  model means a fresh database and a full re-ingest.
- **Nothing in the eval path is free or offline.** `--retrieval-only` skips the answer model but
  still embeds each question through Bedrock. Tests stay offline with fakes and a stubbed boto3
  client; the one live test carries the `live` pytest marker and skips without credentials.

## Corpus facts you must not get wrong

`docs/parts/01-rag-service.md` has the detail under "Corpus facts". The short version:

- The corpus is **Medicare & You 2026, the January 2026 printing**, 128 pages. CMS revises the
  handbook mid-edition: the September 2025 printing of the same 2026 edition prints 2025 dollar
  amounts throughout. Expected answers match the pinned printing and no other.
- medicare.gov serves the handbook at one unversioned URL, which has served the **2027** edition
  since September 2026, and CMS keeps no per-year archive. So `scripts/download_handbook.py`
  fetches pinned bytes from the Internet Archive and verifies sha256
  `d7a341bc3d2d3dab59af746a0875752761e6c2f2dc107d95e9078a23933513d6`. **Never relax the hash
  check** and never point the script at a live URL.
- **The printed page number equals the 1-based PDF page index** for pages 2 to 126. Page 1 is the
  cover, 127 is blank, 128 is the back cover. There is no translation layer anywhere in the code.
- Extraction uses **pypdf**, not pdfplumber: pdfplumber interleaves the handbook's two-column pages
  line by line and destroys the comparison table and the index.
- The PDF is never committed; `data/` is gitignored. Fetch it with the script.

## Repo map

```
CLAUDE.md                     this file
README.md                     public overview, series index, generated scoreboard block
docs/PLAN.md                  the whole series, with a definition of done per part
docs/parts/NN-name.md         build spec for one part, written when we reach it
docs/scripts/NN-name.md       video script, written after the part is built
evals/                        golden set, eval runners, scoreboard generator
scripts/download_handbook.py  pinned, hash-verified handbook download
src/fineprint/                application code — empty; part 1 fills it
tests/
data/                         the handbook PDF lands here; gitignored
docker-compose.yml  pyproject.toml  .env.example
```

## Commands that work today

```bash
uv sync                                      # install dependencies
uv run python scripts/download_handbook.py   # fetch and verify the 2026 handbook into data/raw/
docker compose up -d --wait                  # Postgres 16 with pgvector on localhost:5432
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run python -m evals.verify_golden_set     # check golden-set evidence quotes against the PDF
```

`uv run python scripts/download_handbook.py --edition 2025` fetches the 2025 handbook, which part 4
needs. The service commands (`fineprint init-db`, `ingest`, `search`, `ask`, `serve`) do not
exist yet — part 1 creates them.
