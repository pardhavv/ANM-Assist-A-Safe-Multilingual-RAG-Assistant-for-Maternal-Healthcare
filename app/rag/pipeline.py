"""
The orchestrator: wires every stage together in the order specified in the
architecture doc, minus the stages we explicitly scoped out (self-reflection
as a second LLM pass, conversation memory). Each stage is independently
testable/importable — this module just sequences them and logs the result.

Pipeline:
    normalize -> detect emergency -> detect language -> translate to English
    -> hybrid retrieve -> rerank -> confidence check -> escalate OR generate
    -> translate back -> log
"""
import time
from dataclasses import dataclass, field

from app.query_processing.normalizer import normalize_query
from app.query_processing.emergency_detector import detect_emergency
from app.retriever.hybrid_retriever import hybrid_retrieve
from app.retriever.reranker import rerank
from app.rag.prompt_builder import build_prompt, SYSTEM_INSTRUCTION
from app.rag.confidence_engine import compute_confidence
from app.multilingual.translator import detect_language, translate_to_english, translate_from_english
from app.llm.gemini_client import generate
from app.logging.query_logger import log_query
from app.config.settings import CONFIDENCE_ESCALATION_THRESHOLD, USE_RERANKER

ESCALATION_MESSAGE = (
    "I don't have enough verified information to answer this confidently. "
    "This has been logged and flagged for a qualified Medical Officer to follow up. "
    "Please do not delay care for this patient while waiting on this."
)


@dataclass
class PipelineResult:
    original_query: str
    normalized_query: str
    detected_language: str
    is_emergency: bool
    answer: str
    escalated: bool
    escalation_reason: str
    confidence_score: float
    confidence_signals: dict
    sources: list
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_inr: float


def run_pipeline(raw_query: str, skip_translation: bool = False) -> PipelineResult:
    t_start = time.time()

    # 1. Language detection + translation to English (retrieval itself is
    #    translation-free thanks to the multilingual embedding model, but the
    #    LLM prompt and downstream normalization dictionaries are English-only).
    if skip_translation:
        lang, english_query = "en", raw_query
    else:
        lang = detect_language(raw_query)
        english_query = translate_to_english(raw_query, lang)

    # 2. Deterministic normalization (acronyms, informal terms)
    normalized = normalize_query(english_query)

    # 3. Emergency detection (fast keyword scan, before retrieval)
    emergency_result = detect_emergency(normalized)
    is_emergency = emergency_result["is_emergency"]

    # 4. Hybrid retrieval (dense + BM25 + RRF)
    t_retrieval_start = time.time()
    candidates, top_dense_score = hybrid_retrieve(normalized)

    # 5. Rerank (optional)
    if USE_RERANKER and candidates:
        final_chunks = rerank(normalized, candidates)
    else:
        final_chunks = candidates[:4]
    top_rerank_score = final_chunks[0].get("rerank_score", 0.0) if final_chunks else 0.0
    retrieval_latency_ms = (time.time() - t_retrieval_start) * 1000

    # 6. Confidence engine
    confidence = compute_confidence(
        top_dense_score=top_dense_score,
        top_rerank_score=top_rerank_score,
        context_chunks=final_chunks,
        is_emergency=is_emergency,
        threshold=CONFIDENCE_ESCALATION_THRESHOLD,
    )

    generation_latency_ms = 0.0
    input_tokens = output_tokens = 0
    cost_usd = cost_inr = 0.0

    # 7a. Escalate on low confidence (no LLM call — saves cost + latency)
    if confidence.should_escalate:
        answer_en = ESCALATION_MESSAGE
        escalated = True
        escalation_reason = f"low_confidence (score={confidence.score} < threshold={confidence.signals['effective_threshold']})"
    else:
        # 7b. Generate grounded answer
        prompt = build_prompt(normalized, final_chunks, is_emergency)
        llm_result = generate(prompt, system_instruction=SYSTEM_INSTRUCTION, max_output_tokens=500)
        generation_latency_ms = llm_result.latency_ms
        input_tokens, output_tokens = llm_result.input_tokens, llm_result.output_tokens
        cost_usd, cost_inr = llm_result.cost_usd, llm_result.cost_inr

        # 8. Model self-report gate: even with decent confidence, the model
        #    itself may find the context doesn't actually cover the question.
        if llm_result.text.strip().upper().startswith("INSUFFICIENT_CONTEXT"):
            answer_en = ESCALATION_MESSAGE
            escalated = True
            escalation_reason = "model_flagged_insufficient_context"
        else:
            answer_en = llm_result.text
            escalated = False
            escalation_reason = None

    # 9. Translate back to user's language
    final_answer = answer_en if skip_translation else translate_from_english(answer_en, lang)

    total_latency_ms = (time.time() - t_start) * 1000
    sources = [c["doc_title"] for c in final_chunks] if not escalated else []

    result = PipelineResult(
        original_query=raw_query,
        normalized_query=normalized,
        detected_language=lang,
        is_emergency=is_emergency,
        answer=final_answer,
        escalated=escalated,
        escalation_reason=escalation_reason,
        confidence_score=confidence.score,
        confidence_signals=confidence.signals,
        sources=sources,
        retrieval_latency_ms=round(retrieval_latency_ms, 1),
        generation_latency_ms=round(generation_latency_ms, 1),
        total_latency_ms=round(total_latency_ms, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 6),
        cost_inr=round(cost_inr, 4),
    )

    log_query(
        original_query=result.original_query,
        normalized_query=result.normalized_query,
        detected_language=result.detected_language,
        is_emergency=int(result.is_emergency),
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        cost_inr=result.cost_inr,
        confidence_score=result.confidence_score,
        confidence_signals=result.confidence_signals,
        escalated=int(result.escalated),
        escalation_reason=result.escalation_reason,
        sources=result.sources,
        answer=result.answer,
    )

    return result
