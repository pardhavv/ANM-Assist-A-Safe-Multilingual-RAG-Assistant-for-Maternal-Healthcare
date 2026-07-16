ANM Assist — Multilingual RAG Clinical Decision Support Assistant

A production-shaped (not production-scale) RAG system for frontline maternal-health
workers, built to demonstrate: hybrid retrieval, reranking, a multi-signal confidence
engine, layered emergency/out-of-scope escalation, medical query normalization,
multilingual support, and cost/latency-aware evaluation. Built against ARTPARK's
GenAI/AI-ML Engineer JD, which covers exactly these areas.

> ⚠️ **Demo project, not a validated clinical tool.** The pipeline, safety logic,
> and evaluation methodology are real and tested (see below), but this has not
> undergone clinical validation and should not be used to inform actual patient
> care.

📄 See [`ANM_Assist_Technical_Case_Study.md`](./ANM_Assist_Technical_Case_Study.md)
for a detailed writeup of the architecture, a real safety bug that was found and
fixed during development (escalation accuracy 56.7% → 100% on the eval set), and
an honestly-reported negative result from the embedding fine-tuning experiment.

---

## Table of contents

- [Knowledge base sources](#knowledge-base-sources)
- [Scope note](#scope-note)
- [A note on the LLM model](#a-note-on-the-llm-model)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Why these specific design choices](#why-these-specific-design-choices)
- [Evaluation](#evaluation)
- [Embedding fine-tuning: a documented negative result](#embedding-fine-tuning-a-documented-negative-result)
- [Project structure](#project-structure)
- [What I'd add next](#what-id-add-next-in-priority-order)

---

## Knowledge base sources

This repository does **not** include the source PDFs in `knowledge_base/` (see
`.gitignore`) — some are large, and one is a copyrighted journal article not
freely redistributable. To run this project, download the following yourself
and place them in `knowledge_base/`:

| Document | Source |
|---|---|
| WHO guideline documents | Search ISBNs `9789240045989`, `9789241549356`, `9789241549912`, `9789241550215` on who.int |
| UNFPA Midwifery report | `UNFPA-Midwifery_15_Web` |
| ICMR-DHR Standard Treatment Workflows (Obstetrics & Gynecology) | ICMR / Dept. of Health Research, India |
| PMSMA Operational Framework | Ministry of Health & Family Welfare, India |
| LaQshya Guideline | Ministry of Health & Family Welfare, India |
| "Antenatal management of normal pregnancy" | who.int or relevant national health portal |
| Wright et al., "FIGO good clinical practice paper: management of the second stage of labor" | *Int. J. Gynecology & Obstetrics*, 2020 — access via institution/Wiley; **not redistributed here** (journal copyright) |

Three short original `.md` files (danger signs, ANC schedule, newborn/postnatal
care) **are** included directly in `knowledge_base/` as illustrative starter
content, so the pipeline runs out of the box even before you add the PDFs above.

After placing documents in `knowledge_base/`, run:
```bash
python -m app.retriever.ingest
```

---

## Scope note

The original spec this was built from (Parts 1–4) additionally described Docker,
CI, a Streamlit dashboard, RAGAS/DeepEval integration, conversation memory with
summarization, and a self-reflection LLM pass. Those are real, addable pieces —
they're just not in this cut, because "one script, one run" and "full production
infra" pull in opposite directions, and the eval/interview value is concentrated
in the pipeline below, not the deployment scaffolding. Ask if you want any of
them added next.

---

## A note on the LLM model

This project calls Google's Gemini API. **Google has been deprecating Gemini
models aggressively through 2026** — `gemini-2.0-flash` shut down June 1, 2026,
and `gemini-2.5-flash` began returning 404s for many API keys in July 2026,
ahead of its official shutdown date. `GENERATION_MODEL` / `MODEL_NAME` is fully
configurable via environment variable (see `.env.example`) for exactly this
reason — check [Google's current model list](https://ai.google.dev/gemini-api/docs/models)
and set it to whatever's current (e.g. `gemini-3.1-flash-lite` at time of
writing) if you hit a `404 model no longer available` error.

---

## Quickstart

```bash
pip install -r requirements.txt

cp .env.example .env
# edit .env: set GEMINI_API_KEY, and GENERATION_MODEL if the default is deprecated
export $(cat .env | xargs)   # macOS/Linux; on Windows, set each var manually

# 1. Build the hybrid index (dense + BM25) from knowledge_base/*.md and *.pdf
python -m app.retriever.ingest

# 2. Ask a question from the CLI (this is the "just run it" entry point)
python run_query.py "BP high during pregnancy, what should I do?"

# 3. Run the eval harness (30 Q&A pairs: escalation accuracy, LLM-judge, cost/latency)
python -m eval.run_eval

# 4. (optional) Run the API server
uvicorn app.api.main:app --reload --port 8000
# then POST {"question": "..."} to http://localhost:8000/query
# or open http://localhost:8000/docs
```

To use your own PDFs: drop them into `knowledge_base/`, re-run step 1. The
ingestion pipeline (`app/retriever/ingest.py`) reads `.md` and `.pdf` alike.

> 🪟 **Windows users:** use `set VARNAME=value` in Command Prompt (not
> `export`, which is bash-only), or `$env:VARNAME="value"` in PowerShell. Also
> make sure `EMBEDDING_MODEL` is set *consistently* before running
> `ingest.py` — mixing embedding models between index-build time and query
> time causes a hard dimension-mismatch crash (e.g. `bge-m3` produces
> 1024-dim vectors, the default `paraphrase-multilingual-MiniLM-L12-v2`
> produces 384-dim; the two are not interchangeable without rebuilding the
> index).

---

## Architecture

```
User question (any of EN/HI/KN/TE)
        │
        ▼
Language detection → translate to English (app/multilingual/translator.py)
        │
        ▼
Query normalization: acronym expansion + informal-term mapping
(app/query_processing/normalizer.py) — deterministic, zero-latency, runs
before any LLM call
        │
        ▼
Emergency detection (keyword + regex scan) AND
Out-of-scope detection (drug dosing / brand questions — regex scan)
(app/query_processing/emergency_detector.py) — both deterministic, zero-latency
        │
        ▼
Hybrid retrieval (app/retriever/hybrid_retriever.py)
   ├─ Dense search: multilingual sentence-transformer embeddings
   ├─ BM25 sparse search
   └─ Reciprocal Rank Fusion of the two rank lists
        │
        ▼
Cross-encoder reranking, top-8 → top-4 (app/retriever/reranker.py)
        │
        ▼
Confidence engine (app/rag/confidence_engine.py): blends dense score,
rerank score, and source agreement into one number
        │
        ├─ low confidence           ──┐
        ├─ emergency detected        ─┼─→ ESCALATE, NO LLM call (saves cost + latency)
        ├─ out-of-scope detected    ──┘
        │
        ▼ (only if none of the above fired)
Prompt builder (app/rag/prompt_builder.py) → Gemini generation
        │
        ├─ model outputs INSUFFICIENT_CONTEXT → escalate
        │
        ▼
Translate answer back to user's language
        │
        ▼
SQLite logging (app/logging/query_logger.py): every field from the spec
(latency, tokens, cost, confidence, escalation reason, sources)
```

> 💡 **Note on the three hard-escalation gates:** emergency and out-of-scope
> detection independently force escalation regardless of retrieval
> confidence — this wasn't the original design. During development, testing
> surfaced that a detected emergency was being fed into the confidence score
> as a *soft* signal only, meaning the system could still generate a direct
> answer to a flagged emergency if retrieval happened to look confident. This
> was found via the eval harness (escalation accuracy of 56.7%, with every
> failure in the under-escalate direction) and fixed by making
> emergency/out-of-scope detection hard, independent triggers. See the case
> study for the full story.

---

## Why these specific design choices

**Hybrid retrieval (dense + BM25) fused with RRF, not a weighted score blend.**
Dense embeddings catch semantic paraphrase ("baby not moving" ≈ "decreased fetal
movement"); BM25 catches exact rare-term matches (specific thresholds like
"140/90", drug names) that embeddings can blur. RRF combines their *rankings*
rather than their raw scores, which sidesteps the problem of two incompatible
scoring scales (bounded cosine similarity vs. unbounded, corpus-dependent BM25)
needing manual normalization.

**Retrieve-then-rerank, not rerank-everything.** The cross-encoder reranker is
far more accurate than bi-encoder cosine similarity because it scores the
query and chunk jointly rather than independently — but it's too slow to run
over an entire corpus. Two-stage retrieval (cheap hybrid search narrows to 8
candidates, expensive reranker picks the best 4) is the standard way to get
both accuracy and speed.

**Confidence is a blend of three signals, not one score.** Retrieval similarity
alone can be high even when the corpus doesn't actually cover the topic — a
"closest match among irrelevant options" problem. Adding reranker confidence
and source agreement (do the top chunks come from the same document, i.e. is
there one coherent answer or fragments stitched together) catches cases
similarity-only scoring misses.

**Three independent hard-escalation gates, not one.** Low retrieval confidence,
a detected emergency keyword/pattern, and a detected out-of-scope category
(drug dosing, brand recommendations) are each independently sufficient to
escalate — none depends on the others being correctly calibrated. A fourth,
softer gate (the model self-reporting `INSUFFICIENT_CONTEXT` after actually
reading the retrieved passages) catches cases where retrieval looked
confident but the content genuinely doesn't answer the question. See the case
study for why this ended up as four gates instead of the original two.

**Retrieval is translation-free; the answer path isn't.** The embedding model
lets Hindi/Kannada/Telugu queries match English source chunks directly — no
translation round-trip needed for retrieval, which is both faster and avoids
losing meaning in an early translation step. The final answer still goes
through an explicit translation call, since free-text medical phrasing
benefits from a dedicated pass rather than a single multilingual generation
prompt. *(Note: this path is implemented but has not yet been systematically
evaluated the way the English pipeline has — see "What I'd add next.")*

**Normalization before retrieval, not inside the LLM prompt.** Acronym
expansion and informal-term mapping (`BP` → `Blood Pressure`, `fits` →
`seizure`) are deterministic dictionary lookups — free and instant. Doing this
before retrieval means the embedding/BM25 search sees the medically precise
term, improving recall, without spending an LLM call on something a dictionary
handles reliably.

**Emergency/out-of-scope detection are keyword and regex scans, not LLM
calls.** This is the one stage where latency directly trades against safety
margin — it must never be the bottleneck. A fast, deterministic pass trades
some recall for a hard latency guarantee; the confidence engine downstream
still catches danger-sign language these lists miss, so this isn't the sole
safety net. *(This is also their main limitation: fixed keyword/regex lists
have real, systematic blind spots for phrasing not anticipated in advance —
this was directly observed and partially addressed during development; see
the case study.)*

---

## Evaluation

`eval/eval_questions.json` — 30 hand-written Q&A pairs spanning high-risk danger
signs (must escalate), routine informational questions (answerable from the
docs), and deliberately out-of-scope questions (drug dosing, pediatric vaccine
brands — must escalate rather than hallucinate).

`eval/run_eval.py` reports:

| Metric | Result |
|---|---|
| Escalation accuracy | **1.0** (30/30) |
| LLM-as-judge accuracy (non-escalated answers) | **1.0** |
| Avg latency (warm) | ~1.7s |
| Avg cost/query | ~₹0.006 |

This wasn't the starting result — escalation accuracy began at **0.567**, with
every miss under-escalating a genuine emergency. The full diagnosis and fix
process is documented in
[`ANM_Assist_Technical_Case_Study.md`](./ANM_Assist_Technical_Case_Study.md).

For reference, this beats ARTPARK's own stated benchmark (~₹1.20/query, 8–10s
response time) on cost by roughly 200×; the latency comparison is fairer once
you account for cold-start vs. warm-model timing (see case study).

---

## Embedding fine-tuning: a documented negative result

`app/training/finetune_embeddings.py` fine-tunes the retrieval embedding model
on 40 hand-authored (query, passage) pairs using `MultipleNegativesRankingLoss`;
`app/training/evaluate_retrieval.py` measures Recall@1/Recall@3/MRR against an
18-pair held-out set with different phrasing than training.

**This was actually run, and the honest result is that it doesn't help:** with
only 40 training pairs, fine-tuning produced no measurable improvement and, at
higher epoch counts, mildly degraded retrieval quality versus the untuned base
model. `USE_FINETUNED_EMBEDDER=false` (the default) is the empirically correct
setting given this data, not just the cautious one. See the case study for the
full numbers and root-cause analysis — the limiting factor is training data
volume, not the method.

```bash
python -m app.training.finetune_embeddings     # writes to models/finetuned-embedder/
python -m app.training.evaluate_retrieval       # prints base vs. fine-tuned Recall@1/3, MRR
```

---

## Project structure

```
ANM-Assist/
├── app/
│   ├── config/settings.py         # all env-driven config, acronym/term dicts
│   ├── llm/gemini_client.py       # single point of contact with Gemini API
│   ├── query_processing/
│   │   ├── normalizer.py          # acronym expansion, informal-term mapping
│   │   └── emergency_detector.py  # emergency + out-of-scope detection (keyword/regex)
│   ├── retriever/
│   │   ├── ingest.py              # PDF/MD loading, chunking, index building
│   │   ├── hybrid_retriever.py    # dense + BM25 + RRF
│   │   └── reranker.py            # cross-encoder reranking
│   ├── rag/
│   │   ├── prompt_builder.py      # structured system prompt + context assembly
│   │   ├── confidence_engine.py   # multi-signal confidence scoring
│   │   └── pipeline.py            # orchestrates the full flow, incl. escalation gates
│   ├── multilingual/translator.py # detect + translate via Gemini
│   ├── logging/query_logger.py    # SQLite logging + summary stats
│   ├── training/
│   │   ├── finetune_embeddings.py # contrastive fine-tuning of the embedder
│   │   └── evaluate_retrieval.py  # base vs fine-tuned Recall@k/MRR comparison
│   └── api/main.py                # FastAPI: /query, /health, /metrics
├── training_data/
│   ├── embedding_train_pairs.json # 40 (query, passage) pairs for fine-tuning
│   └── embedding_eval_pairs.json  # 18 held-out pairs, different phrasing
├── models/finetuned-embedder/     # generated by finetune_embeddings.py (gitignored)
├── knowledge_base/                # .md files included; add your own PDFs (see above)
├── vector_store/                  # generated: dense + BM25 indices (gitignored)
├── eval/
│   ├── eval_questions.json
│   └── run_eval.py
├── ANM_Assist_Technical_Case_Study.md  # detailed writeup: architecture, bug fix, evals
├── run_query.py                   # one-command CLI entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## What I'd add next (in priority order)

1. A larger, more adversarial eval set — the current 30 questions are hand-picked
   and demonstrate the escalation-logic fix well, but aren't a substitute for
   real-world phrasing variety or clinical validation
2. Systematic evaluation of the Hindi/Kannada/Telugu path specifically — it's
   implemented and the design is sound, but has not yet been tested with the
   same rigor as the English pipeline
3. More training data for the embedding fine-tuning experiment (several
   hundred pairs at minimum) — 40 pairs was enough to demonstrate the method
   but not enough to produce a measurable improvement (see above)
4. Conversation memory (a summarization buffer) so multi-turn context like
   "she is 28 weeks pregnant" → later "what should I do?" resolves correctly
5. A proper eval framework integration (RAGAS/DeepEval) for faithfulness and
   context-precision metrics beyond the custom escalation/judge scoring here
6. Self-reflection pass: a second, cheaper LLM call that checks the drafted
   answer against the retrieved context before it's returned
7. Docker + docker-compose for reproducible deployment
8. Swap SQLite for Postgres and add the Streamlit dashboard once there's
   enough query volume to make the aggregations interesting
