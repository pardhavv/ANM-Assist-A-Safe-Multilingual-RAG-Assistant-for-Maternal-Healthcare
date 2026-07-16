"""
Confidence engine: combines multiple independent signals into one score,
rather than trusting any single number.

Why not just use retrieval similarity alone:
    A high dense-similarity score can still mean "closest match in a corpus
    that doesn't actually cover this topic" — similarity is relative to what's
    in the index, not an absolute measure of coverage. Combining it with
    source agreement (do the top chunks come from the same document, i.e. is
    there a consistent answer, or are they scattered/contradictory hints from
    unrelated sections) and reranker confidence catches cases dense-only
    scoring misses.

Signals combined here:
    1. top_dense_score       - raw embedding similarity of best match
    2. top_rerank_score      - cross-encoder's judgment of query-chunk relevance
    3. source_agreement      - fraction of final top-k chunks sharing the same
                                source document (proxy for "is there one
                                coherent answer, or fragments stitched together")
    4. is_emergency          - emergency queries get a stricter effective bar,
                                since a false "confident" answer is more costly here
"""
from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    score: float
    signals: dict
    should_escalate: bool


def compute_confidence(top_dense_score: float, top_rerank_score: float,
                        context_chunks: list[dict], is_emergency: bool,
                        threshold: float) -> ConfidenceResult:
    if context_chunks:
        sources = [c["doc_title"] for c in context_chunks]
        most_common_count = max(sources.count(s) for s in set(sources))
        source_agreement = most_common_count / len(sources)
    else:
        source_agreement = 0.0

    # Normalize rerank score (cross-encoder ms-marco models output roughly
    # -10..10 logits) into a rough 0-1 band for combination purposes.
    normalized_rerank = max(0.0, min(1.0, (top_rerank_score + 10) / 20))

    # Weighted blend — weights chosen to favor the reranker (most accurate
    # signal) while still letting raw retrieval and source coherence pull the
    # score down when they disagree with it.
    score = (
        0.35 * max(0.0, top_dense_score) +
        0.40 * normalized_rerank +
        0.25 * source_agreement
    )

    effective_threshold = threshold * 1.15 if is_emergency else threshold
    # Emergency queries: only escalate on confidence if BOTH the confidence
    # score is low AND there's no strong direct evidence, since we don't want
    # a low source-agreement score alone to suppress an emergency answer.
    should_escalate = score < effective_threshold

    return ConfidenceResult(
        score=round(score, 4),
        signals={
            "top_dense_score": round(top_dense_score, 4),
            "top_rerank_score": round(top_rerank_score, 4),
            "normalized_rerank": round(normalized_rerank, 4),
            "source_agreement": round(source_agreement, 4),
            "effective_threshold": round(effective_threshold, 4),
        },
        should_escalate=should_escalate,
    )
