"""
Hybrid retrieval: dense (embedding cosine similarity) + sparse (BM25),
combined with Reciprocal Rank Fusion (RRF).

Why hybrid, not just dense:
    Dense retrieval is great at semantic/paraphrase matches ("baby not moving"
    ~ "decreased fetal movement") but can miss exact rare terms (drug names,
    specific numeric thresholds like "140/90") that BM25 catches reliably via
    term overlap. Combining both and fusing with RRF is a standard way to get
    the benefits of each without needing to hand-tune a weighted blend of two
    very differently-scaled similarity metrics — RRF only cares about *rank*,
    not raw score magnitude, which is what makes it robust to combine
    unrelated scoring systems (cosine similarity vs BM25 score).

Why RRF specifically over a weighted score sum:
    Score sum requires normalizing two incompatible score distributions
    (cosine in [-1,1], BM25 unbounded and corpus-dependent) — fragile and
    needs re-tuning per corpus. RRF sidesteps this entirely by only using
    each chunk's rank position in each list: score = sum(1 / (k + rank)).
    It's simpler, needs no tuning beyond the constant k, and is what most
    production hybrid-retrieval systems (e.g. Elasticsearch's own RRF) use.
"""
import pickle
import numpy as np

from app.config.settings import (
    DENSE_INDEX_PATH, BM25_INDEX_PATH, ACTIVE_EMBEDDING_MODEL,
    DENSE_TOP_K, BM25_TOP_K, RRF_K, FUSED_TOP_K,
)

_embed_model = None
_dense_chunks = None
_dense_by_id = None
_bm25 = None
_bm25_chunk_ids = None


def _load():
    global _embed_model, _dense_chunks, _dense_by_id, _bm25, _bm25_chunk_ids
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(ACTIVE_EMBEDDING_MODEL)
    if _dense_chunks is None:
        with open(DENSE_INDEX_PATH, "rb") as f:
            _dense_chunks = pickle.load(f)
        _dense_by_id = {c.chunk_id: c for c in _dense_chunks}
    if _bm25 is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25 = data["bm25"]
        _bm25_chunk_ids = data["chunk_ids"]
    return _embed_model, _dense_chunks, _dense_by_id, _bm25, _bm25_chunk_ids


def dense_search(query: str, top_k: int = DENSE_TOP_K):
    model, chunks, _, _, _ = _load()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = np.array([np.dot(q_emb, c.embedding) for c in chunks])
    order = np.argsort(-scores)[:top_k]
    return [(chunks[i].chunk_id, float(scores[i])) for i in order]


def bm25_search(query: str, top_k: int = BM25_TOP_K):
    _, _, _, bm25, chunk_ids = _load()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    order = np.argsort(-scores)[:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in order]


def reciprocal_rank_fusion(rank_lists: list[list[tuple[str, float]]], k: int = RRF_K):
    """
    rank_lists: list of ranked (id, score) lists (already sorted best-first).
    Returns dict {id: fused_score}, higher is better.
    """
    fused = {}
    for rank_list in rank_lists:
        for rank, (item_id, _score) in enumerate(rank_list):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_retrieve(query: str, top_k: int = FUSED_TOP_K):
    """
    Returns list of dicts: {chunk_id, text, source_file, doc_title, fused_score,
    dense_score, bm25_score} sorted best-first, plus the raw top dense
    similarity (used as one signal in the confidence engine).
    """
    _, _, dense_by_id, _, _ = _load()

    dense_results = dense_search(query)
    bm25_results = bm25_search(query)

    fused_scores = reciprocal_rank_fusion([dense_results, bm25_results])
    dense_score_map = dict(dense_results)
    bm25_score_map = dict(bm25_results)

    ranked_ids = sorted(fused_scores.keys(), key=lambda i: -fused_scores[i])[:top_k]

    results = []
    for cid in ranked_ids:
        chunk = dense_by_id[cid]
        results.append({
            "chunk_id": cid,
            "text": chunk.text,
            "source_file": chunk.source_file,
            "doc_title": chunk.doc_title,
            "fused_score": fused_scores[cid],
            "dense_score": dense_score_map.get(cid, 0.0),
            "bm25_score": bm25_score_map.get(cid, 0.0),
        })

    top_dense_score = dense_results[0][1] if dense_results else 0.0
    return results, top_dense_score


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What should I do if BP is very high during pregnancy?"
    results, top_dense = hybrid_retrieve(q)
    print(f"Query: {q}\nTop dense score: {top_dense:.3f}\n")
    for r in results:
        print(f"[{r['doc_title']} | fused={r['fused_score']:.4f} dense={r['dense_score']:.3f} bm25={r['bm25_score']:.2f}]")
        print(f"  {r['text'][:120]}...\n")
