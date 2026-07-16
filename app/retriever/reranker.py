"""
Cross-encoder reranker: takes the fused hybrid-retrieval candidates and
re-scores each (query, chunk) pair jointly, which is more accurate than
bi-encoder cosine similarity (which scores query and chunk independently and
can't model fine-grained interaction between them) but too slow to run over
the whole corpus — hence the two-stage retrieve-then-rerank pattern: cheap
retrieval narrows thousands of chunks to ~8, then the expensive-but-accurate
reranker picks the best 4 from those 8.

Set USE_RERANKER=false in settings to skip this stage (e.g. on CPU-only
machines where the extra model load isn't worth the latency).
"""
from app.config.settings import RERANKER_MODEL, FINAL_TOP_K

_reranker = None


def _load():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = FINAL_TOP_K):
    """
    candidates: list of dicts with a 'text' key (as returned by hybrid_retrieve).
    Returns the same dicts, top_k of them, sorted by rerank_score desc, with a
    'rerank_score' key added.
    """
    if not candidates:
        return []
    reranker = _load()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: -c["rerank_score"])
    return ranked[:top_k]
