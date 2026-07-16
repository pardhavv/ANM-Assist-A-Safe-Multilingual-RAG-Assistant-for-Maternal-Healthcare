# \# ANM Assist — Multilingual RAG Clinical Decision Support Assistant

# 

# A production-shaped (not production-scale) RAG system for frontline maternal-health

# workers, built to demonstrate: hybrid retrieval, reranking, a multi-signal confidence

# engine, layered emergency/out-of-scope escalation, medical query normalization,

# multilingual support, and cost/latency-aware evaluation. Built against ARTPARK's

# GenAI/AI-ML Engineer JD, which covers exactly these areas.

# 

# > ⚠️ \*\*Demo project, not a validated clinical tool.\*\* The pipeline, safety logic,

# > and evaluation methodology are real and tested (see below), but this has not

# > undergone clinical validation and should not be used to inform actual patient

# > care.

# 

# See \[`ANM\_Assist\_Technical\_Case\_Study.md`](./ANM\_Assist\_Technical\_Case\_Study.md)

# for a detailed writeup of the architecture, a real safety bug that was found and

# fixed during development (escalation accuracy 56.7% → 100% on the eval set), and

# an honestly-reported negative result from the embedding fine-tuning experiment.

# 

# \## Knowledge base sources

# 

# This repository does \*\*not\*\* include the source PDFs in `knowledge\_base/` (see

# `.gitignore`) — some are large, and one is a copyrighted journal article not

# freely redistributable. To run this project, download the following yourself

# and place them in `knowledge\_base/`:

# 

# \- WHO guideline documents (search by ISBN on who.int): `9789240045989`,

# &#x20; `9789241549356`, `9789241549912`, `9789241550215`

# \- UNFPA Midwifery report ("UNFPA-Midwifery\_15\_Web")

# \- ICMR-DHR Standard Treatment Workflows (Obstetrics \& Gynecology)

# \- PMSMA Operational Framework (Ministry of Health \& Family Welfare, India)

# \- LaQshya Guideline (Ministry of Health \& Family Welfare, India)

# \- "Antenatal management of normal pregnancy" (search title on who.int or a

# &#x20; national health portal)

# \- Wright et al., "FIGO good clinical practice paper: management of the second

# &#x20; stage of labor," \*International Journal of Gynecology \& Obstetrics\*, 2020 —

# &#x20; access via your institution or publisher (Wiley); not redistributed here due

# &#x20; to journal copyright.

# 

# Three short original `.md` files (danger signs, ANC schedule, newborn/postnatal

# care) \*\*are\*\* included directly in `knowledge\_base/` as illustrative starter

# content, so the pipeline runs out of the box even before you add the PDFs above.

# 

# After placing documents in `knowledge\_base/`, run:

# ```bash

# python -m app.retriever.ingest

# ```

# 

# \## Scope note

# 

# The original spec this was built from (Parts 1–4) additionally described Docker,

# CI, a Streamlit dashboard, RAGAS/DeepEval integration, conversation memory with

# summarization, and a self-reflection LLM pass. Those are real, addable pieces —

# they're just not in this cut, because "one script, one run" and "full production

# infra" pull in opposite directions, and the eval/interview value is concentrated

# in the pipeline below, not the deployment scaffolding. Ask if you want any of

# them added next.

# 

# \## A note on the LLM model

# 

# This project calls Google's Gemini API. \*\*Google has been deprecating Gemini

# models aggressively through 2026\*\* — `gemini-2.0-flash` shut down June 1, 2026,

# and `gemini-2.5-flash` began returning 404s for many API keys in July 2026,

# ahead of its official shutdown date. `GENERATION\_MODEL` / `MODEL\_NAME` is fully

# configurable via environment variable (see `.env.example`) for exactly this

# reason — check

# \[Google's current model list](https://ai.google.dev/gemini-api/docs/models) and

# set it to whatever's current (e.g. `gemini-3.1-flash-lite` at time of writing)

# if you hit a `404 model no longer available` error.

# 

# \## Quickstart

# 

# ```bash

# pip install -r requirements.txt

# 

# cp .env.example .env

# \# edit .env: set GEMINI\_API\_KEY, and GENERATION\_MODEL if the default is deprecated

# export $(cat .env | xargs)   # macOS/Linux; on Windows, set each var manually

# 

# \# 1. Build the hybrid index (dense + BM25) from knowledge\_base/\*.md and \*.pdf

# python -m app.retriever.ingest

# 

# \# 2. Ask a question from the CLI (this is the "just run it" entry point)

# python run\_query.py "BP high during pregnancy, what should I do?"

# 

# \# 3. Run the eval harness (30 Q\&A pairs: escalation accuracy, LLM-judge, cost/latency)

# python -m eval.run\_eval

# 

# \# 4. (optional) Run the API server

# uvicorn app.api.main:app --reload --port 8000

# \# then POST {"question": "..."} to http://localhost:8000/query

# \# or open http://localhost:8000/docs

# ```

# 

# To use your own PDFs: drop them into `knowledge\_base/`, re-run step 1. The

# ingestion pipeline (`app/retriever/ingest.py`) reads `.md` and `.pdf` alike.

# 

# \*\*Windows users:\*\* use `set VARNAME=value` in Command Prompt (not `export`,

# which is bash-only), or `$env:VARNAME="value"` in PowerShell. Also make sure

# `EMBEDDING\_MODEL` is set consistently before running `ingest.py` — mixing

# embedding models between index-build time and query time causes a hard

# dimension-mismatch crash (e.g. `bge-m3` produces 1024-dim vectors, the default

# `paraphrase-multilingual-MiniLM-L12-v2` produces 384-dim; the two are not

# interchangeable without rebuilding the index).

# 

# \## Architecture

# 

# ```

# User question (any of EN/HI/KN/TE)

# &#x20;       │

# &#x20;       ▼

# Language detection → translate to English (app/multilingual/translator.py)

# &#x20;       │

# &#x20;       ▼

# Query normalization: acronym expansion + informal-term mapping

# (app/query\_processing/normalizer.py) — deterministic, zero-latency, runs

# before any LLM call

# &#x20;       │

# &#x20;       ▼

# Emergency detection (keyword + regex scan) AND

# Out-of-scope detection (drug dosing / brand questions — regex scan)

# (app/query\_processing/emergency\_detector.py) — both deterministic, zero-latency

# &#x20;       │

# &#x20;       ▼

# Hybrid retrieval (app/retriever/hybrid\_retriever.py)

# &#x20;  ├─ Dense search: multilingual sentence-transformer embeddings

# &#x20;  ├─ BM25 sparse search

# &#x20;  └─ Reciprocal Rank Fusion of the two rank lists

# &#x20;       │

# &#x20;       ▼

# Cross-encoder reranking, top-8 → top-4 (app/retriever/reranker.py)

# &#x20;       │

# &#x20;       ▼

# Confidence engine (app/rag/confidence\_engine.py): blends dense score,

# rerank score, and source agreement into one number

# &#x20;       │

# &#x20;       ├─ low confidence           ──┐

# &#x20;       ├─ emergency detected        ─┼─→ ESCALATE, NO LLM call (saves cost + latency)

# &#x20;       ├─ out-of-scope detected    ──┘

# &#x20;       │

# &#x20;       ▼ (only if none of the above fired)

# Prompt builder (app/rag/prompt\_builder.py) → Gemini generation

# &#x20;       │

# &#x20;       ├─ model outputs INSUFFICIENT\_CONTEXT → escalate

# &#x20;       │

# &#x20;       ▼

# Translate answer back to user's language

# &#x20;       │

# &#x20;       ▼

# SQLite logging (app/logging/query\_logger.py): every field from the spec

# (latency, tokens, cost, confidence, escalation reason, sources)

# ```

# 

# \*\*Note on the three hard-escalation gates:\*\* emergency and out-of-scope

# detection independently force escalation regardless of retrieval confidence —

# this wasn't the original design. During development, testing surfaced that a

# detected emergency was being fed into the confidence score as a \*soft\* signal

# only, meaning the system could still generate a direct answer to a flagged

# emergency if retrieval happened to look confident. This was found via the eval

# harness (escalation accuracy of 56.7%, with every failure in the

# under-escalate direction) and fixed by making emergency/out-of-scope detection

# hard, independent triggers. See the case study for the full story.

# 

# \## Why these specific design choices

# 

# \*\*Hybrid retrieval (dense + BM25) fused with RRF, not a weighted score blend.\*\*

# Dense embeddings catch semantic paraphrase ("baby not moving" ≈ "decreased fetal

# movement"); BM25 catches exact rare-term matches (specific thresholds like

# "140/90", drug names) that embeddings can blur. RRF combines their \*rankings\*

# rather than their raw scores, which sidesteps the problem of two incompatible

# scoring scales (bounded cosine similarity vs. unbounded, corpus-dependent BM25)

# needing manual normalization.

# 

# \*\*Retrieve-then-rerank, not rerank-everything.\*\* The cross-encoder reranker is

# far more accurate than bi-encoder cosine similarity because it scores the

# query and chunk jointly rather than independently — but it's too slow to run

# over an entire corpus. Two-stage retrieval (cheap hybrid search narrows to 8

# candidates, expensive reranker picks the best 4) is the standard way to get

# both accuracy and speed.

# 

# \*\*Confidence is a blend of three signals, not one score.\*\* Retrieval similarity

# alone can be high even when the corpus doesn't actually cover the topic — a

# "closest match among irrelevant options" problem. Adding reranker confidence

# and source agreement (do the top chunks come from the same document, i.e. is

# there one coherent answer or fragments stitched together) catches cases

# similarity-only scoring misses.

# 

# \*\*Three independent hard-escalation gates, not one.\*\* Low retrieval confidence,

# a detected emergency keyword/pattern, and a detected out-of-scope category

# (drug dosing, brand recommendations) are each independently sufficient to

# escalate — none depends on the others being correctly calibrated. A fourth,

# softer gate (the model self-reporting `INSUFFICIENT\_CONTEXT` after actually

# reading the retrieved passages) catches cases where retrieval looked

# confident but the content genuinely doesn't answer the question. See the case

# study for why this ended up as four gates instead of the original two.

# 

# \*\*Retrieval is translation-free; the answer path isn't.\*\* The embedding model

# lets Hindi/Kannada/Telugu queries match English source chunks directly — no

# translation round-trip needed for retrieval, which is both faster and avoids

# losing meaning in an early translation step. The final answer still goes

# through an explicit translation call, since free-text medical phrasing

# benefits from a dedicated pass rather than a single multilingual generation

# prompt. (Note: this path is implemented but has not yet been systematically

# evaluated the way the English pipeline has — see "What I'd add next.")

# 

# \*\*Normalization before retrieval, not inside the LLM prompt.\*\* Acronym

# expansion and informal-term mapping (`BP` → `Blood Pressure`, `fits` →

# `seizure`) are deterministic dictionary lookups — free and instant. Doing this

# before retrieval means the embedding/BM25 search sees the medically precise

# term, improving recall, without spending an LLM call on something a dictionary

# handles reliably.

# 

# \*\*Emergency/out-of-scope detection are keyword and regex scans, not LLM

# calls.\*\* This is the one stage where latency directly trades against safety

# margin — it must never be the bottleneck. A fast, deterministic pass trades

# some recall for a hard latency guarantee; the confidence engine downstream

# still catches danger-sign language these lists miss, so this isn't the sole

# safety net. (This is also their main limitation: fixed keyword/regex lists have

# real, systematic blind spots for phrasing not anticipated in advance — this was

# directly observed and partially addressed during development; see the case

# study.)

# 

# \## Evaluation

# 

# `eval/eval\_questions.json` — 30 hand-written Q\&A pairs spanning high-risk danger

# signs (must escalate), routine informational questions (answerable from the

# docs), and deliberately out-of-scope questions (drug dosing, pediatric vaccine

# brands — must escalate rather than hallucinate).

# 

# `eval/run\_eval.py` reports:

# \- \*\*Escalation accuracy\*\* against the gold labels — currently \*\*1.0 (30/30)\*\*

# \- \*\*LLM-as-judge accuracy\*\* on non-escalated answers (graded against a gold

# &#x20; topic, not exact wording) — currently \*\*1.0\*\*

# \- \*\*Latency\*\* (avg + p90) and \*\*cost-per-query in ₹\*\* — directly comparable to

# &#x20; ARTPARK's own stated benchmark (\~₹1.20/query, 8–10s response time); this

# &#x20; system runs at roughly ₹0.006/query and \~1.7s average latency once models

# &#x20; are warm (see case study for the cold-start vs. warm-model distinction)

# 

# This wasn't the starting result — escalation accuracy began at 56.7%, with

# every miss under-escalating a genuine emergency. The full diagnosis and fix

# process is documented in

# \[`ANM\_Assist\_Technical\_Case\_Study.md`](./ANM\_Assist\_Technical\_Case\_Study.md).

# 

# \## Embedding fine-tuning: a documented negative result

# 

# `app/training/finetune\_embeddings.py` fine-tunes the retrieval embedding model

# on 40 hand-authored (query, passage) pairs using `MultipleNegativesRankingLoss`;

# `app/training/evaluate\_retrieval.py` measures Recall@1/Recall@3/MRR against an

# 18-pair held-out set with different phrasing than training.

# 

# \*\*This was actually run, and the honest result is that it doesn't help:\*\* with

# only 40 training pairs, fine-tuning produced no measurable improvement and, at

# higher epoch counts, mildly degraded retrieval quality versus the untuned base

# model. `USE\_FINETUNED\_EMBEDDER=false` (the default) is the empirically correct

# setting given this data, not just the cautious one. See the case study for the

# full numbers and root-cause analysis — the limiting factor is training data

# volume, not the method.

# 

# ```bash

# python -m app.training.finetune\_embeddings     # writes to models/finetuned-embedder/

# python -m app.training.evaluate\_retrieval       # prints base vs. fine-tuned Recall@1/3, MRR

# ```

# 

# \## Project structure

# 

# ```

# ANM-Assist/

# ├── app/

# │   ├── config/settings.py         # all env-driven config, acronym/term dicts

# │   ├── llm/gemini\_client.py       # single point of contact with Gemini API

# │   ├── query\_processing/

# │   │   ├── normalizer.py          # acronym expansion, informal-term mapping

# │   │   └── emergency\_detector.py  # emergency + out-of-scope detection (keyword/regex)

# │   ├── retriever/

# │   │   ├── ingest.py              # PDF/MD loading, chunking, index building

# │   │   ├── hybrid\_retriever.py    # dense + BM25 + RRF

# │   │   └── reranker.py            # cross-encoder reranking

# │   ├── rag/

# │   │   ├── prompt\_builder.py      # structured system prompt + context assembly

# │   │   ├── confidence\_engine.py   # multi-signal confidence scoring

# │   │   └── pipeline.py            # orchestrates the full flow, incl. escalation gates

# │   ├── multilingual/translator.py # detect + translate via Gemini

# │   ├── logging/query\_logger.py    # SQLite logging + summary stats

# │   ├── training/

# │   │   ├── finetune\_embeddings.py # contrastive fine-tuning of the embedder

# │   │   └── evaluate\_retrieval.py  # base vs fine-tuned Recall@k/MRR comparison

# │   └── api/main.py                # FastAPI: /query, /health, /metrics

# ├── training\_data/

# │   ├── embedding\_train\_pairs.json # 40 (query, passage) pairs for fine-tuning

# │   └── embedding\_eval\_pairs.json  # 18 held-out pairs, different phrasing

# ├── models/finetuned-embedder/     # generated by finetune\_embeddings.py (gitignored)

# ├── knowledge\_base/                # .md files included; add your own PDFs (see above)

# ├── vector\_store/                  # generated: dense + BM25 indices (gitignored)

# ├── eval/

# │   ├── eval\_questions.json

# │   └── run\_eval.py

# ├── ANM\_Assist\_Technical\_Case\_Study.md  # detailed writeup: architecture, bug fix, evals

# ├── run\_query.py                   # one-command CLI entry point

# ├── requirements.txt

# ├── .env.example

# └── .gitignore

# ```

# 

# \## What I'd add next (in priority order)

# 

# 1\. A larger, more adversarial eval set — the current 30 questions are hand-picked

# &#x20;  and demonstrate the escalation-logic fix well, but aren't a substitute for

# &#x20;  real-world phrasing variety or clinical validation

# 2\. Systematic evaluation of the Hindi/Kannada/Telugu path specifically — it's

# &#x20;  implemented and the design is sound, but has not yet been tested with the

# &#x20;  same rigor as the English pipeline

# 3\. More training data for the embedding fine-tuning experiment (several

# &#x20;  hundred pairs at minimum) — 40 pairs was enough to demonstrate the method

# &#x20;  but not enough to produce a measurable improvement (see above)

# 4\. Conversation memory (a summarization buffer) so multi-turn context like

# &#x20;  "she is 28 weeks pregnant" → later "what should I do?" resolves correctly

# 5\. A proper eval framework integration (RAGAS/DeepEval) for faithfulness and

# &#x20;  context-precision metrics beyond the custom escalation/judge scoring here

# 6\. Self-reflection pass: a second, cheaper LLM call that checks the drafted

# &#x20;  answer against the retrieved context before it's returned

# 7\. Docker + docker-compose for reproducible deployment

# 8\. Swap SQLite for Postgres and add the Streamlit dashboard once there's

# &#x20;  enough query volume to make the aggregations interesting

