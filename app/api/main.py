"""
FastAPI backend.

Run:
    uvicorn app.api.main:app --reload --port 8000

Then POST to http://localhost:8000/query with {"question": "..."}
or open http://localhost:8000/docs for interactive OpenAPI docs.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dataclasses import asdict

from app.rag.pipeline import run_pipeline
from app.logging.query_logger import get_summary_stats, init_db

app = FastAPI(
    title="ANM Assist API",
    description="Multilingual RAG assistant for frontline maternal-health workers.",
    version="0.1.0",
)


@app.on_event("startup")
def _startup():
    init_db()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The health worker's question, in any supported language.")
    skip_translation: bool = Field(False, description="Set true if the question is already in English and you want to skip the translation round-trip (faster/cheaper).")


class QueryResponse(BaseModel):
    original_query: str
    normalized_query: str
    detected_language: str
    is_emergency: bool
    answer: str
    escalated: bool
    escalation_reason: str | None
    confidence_score: float
    confidence_signals: dict
    sources: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    cost_inr: float


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = run_pipeline(req.question, skip_translation=req.skip_translation)
        return asdict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return get_summary_stats()
