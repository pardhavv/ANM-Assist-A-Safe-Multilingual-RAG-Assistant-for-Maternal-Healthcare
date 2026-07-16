"""
Measures whether fine-tuning actually improved retrieval, on a held-out set
of (query, gold_passage) pairs that were NOT used in training — different
phrasing, same underlying facts. This distinguishes genuine generalization
from memorizing the training queries verbatim.

Metrics:
    - Recall@1, Recall@3: does the gold passage appear in the top-1 / top-3
      most similar chunks retrieved from the full knowledge base?
    - MRR (Mean Reciprocal Rank): rewards ranking the gold passage higher,
      not just getting it into the top-k at all.

This compares the BASE embedding model against the FINE-TUNED one on
identical queries, over identical candidate passages, so any difference is
attributable to the fine-tuning itself.

Usage:
    python -m app.training.finetune_embeddings   # train first
    python -m app.training.evaluate_retrieval
"""
import json
import os

from sentence_transformers import SentenceTransformer
import numpy as np

from app.config.settings import EMBEDDING_MODEL, BASE_DIR
from app.retriever.ingest import load_documents, chunk_text
from app.config.settings import KNOWLEDGE_BASE_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

EVAL_PAIRS_PATH = os.path.join(BASE_DIR, "training_data", "embedding_eval_pairs.json")
FINETUNED_MODEL_DIR = os.path.join(BASE_DIR, "models", "finetuned-embedder")


def build_candidate_passages():
    """Rebuilds the same chunk set used at query time, so retrieval here
    matches what the live pipeline would actually see."""
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = []
    for source, text in docs:
        chunks.extend(chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS))
    return chunks


import re


def _normalize_for_matching(text: str) -> str:
    """Strip markdown bold markers, leading list numbers/bullets, and collapse
    whitespace, so gold passages written as plain prose still match chunks
    that retain the source markdown formatting (**bold**, '1. ', '- ', etc.)."""
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"^\s*[\d]+\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_gold_chunk_index(gold_passage: str, chunks: list[str]) -> int:
    """The gold passage in the eval set is a short excerpt; find which chunk
    contains it (substring match on normalized text) since chunk boundaries
    don't necessarily align 1:1 with the excerpt, and markdown formatting
    (bold, numbering) may differ between how the excerpt was written and how
    it appears in the source chunk."""
    normalized_gold = _normalize_for_matching(gold_passage)
    for i, chunk in enumerate(chunks):
        if normalized_gold in _normalize_for_matching(chunk):
            return i
    return -1


def evaluate_model(model_name_or_path: str, chunks: list[str], eval_pairs: list[dict]):
    model = SentenceTransformer(model_name_or_path)
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0
    skipped = 0

    for pair in eval_pairs:
        gold_idx = find_gold_chunk_index(pair["gold_passage"], chunks)
        if gold_idx == -1:
            skipped += 1
            continue

        q_emb = model.encode([pair["query"]], normalize_embeddings=True)[0]
        scores = np.dot(chunk_embeddings, q_emb)
        ranking = np.argsort(-scores)  # best-first chunk indices

        rank_of_gold = int(np.where(ranking == gold_idx)[0][0]) + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank_of_gold)
        if rank_of_gold == 1:
            recall_at_1 += 1
        if rank_of_gold <= 3:
            recall_at_3 += 1

    n = len(eval_pairs) - skipped
    return {
        "n_evaluated": n,
        "n_skipped_no_gold_match": skipped,
        "recall_at_1": round(recall_at_1 / n, 3) if n else None,
        "recall_at_3": round(recall_at_3 / n, 3) if n else None,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else None,
    }


def run():
    with open(EVAL_PAIRS_PATH, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)

    chunks = build_candidate_passages()
    print(f"Evaluating over {len(chunks)} candidate chunks, {len(eval_pairs)} held-out queries.\n")

    print(f"--- BASE MODEL: {EMBEDDING_MODEL} ---")
    base_results = evaluate_model(EMBEDDING_MODEL, chunks, eval_pairs)
    print(json.dumps(base_results, indent=2))

    if not os.path.exists(FINETUNED_MODEL_DIR):
        print(f"\nNo fine-tuned model found at {FINETUNED_MODEL_DIR}. "
              f"Run `python -m app.training.finetune_embeddings` first to compare.")
        return

    print(f"\n--- FINE-TUNED MODEL: {FINETUNED_MODEL_DIR} ---")
    finetuned_results = evaluate_model(FINETUNED_MODEL_DIR, chunks, eval_pairs)
    print(json.dumps(finetuned_results, indent=2))

    print("\n--- DELTA (fine-tuned minus base) ---")
    for metric in ["recall_at_1", "recall_at_3", "mrr"]:
        b, ft = base_results.get(metric), finetuned_results.get(metric)
        if b is not None and ft is not None:
            print(f"{metric}: {b} -> {ft}  (Δ {round(ft - b, 3):+})")


if __name__ == "__main__":
    run()
