"""
Document ingestion: loads .md and .pdf files from KNOWLEDGE_BASE_DIR, chunks
them, and builds two indices:
    - dense index (sentence-transformer embeddings, pickled)
    - BM25 index (rank_bm25, pickled)

Run:
    python -m app.retriever.ingest
"""
import os
import glob
import pickle
import hashlib

import numpy as np

from app.config.settings import (
    KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, DENSE_INDEX_PATH, BM25_INDEX_PATH,
    CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, ACTIVE_EMBEDDING_MODEL,
)
from app.retriever.chunk_model import Chunk


def _read_pdf(path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _read_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(kb_dir: str = KNOWLEDGE_BASE_DIR):
    """Loads every .md and .pdf file in kb_dir. Returns list of (filename, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*"))):
        if path.lower().endswith(".md"):
            docs.append((os.path.basename(path), _read_md(path)))
        elif path.lower().endswith(".pdf"):
            print(f"Extracting text from PDF: {path}")
            docs.append((os.path.basename(path), _read_pdf(path)))
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Paragraph-aware sliding window chunker (fixed + recursive-ish hybrid:
    splits on paragraph boundaries first, falls back to hard character split
    for paragraphs longer than chunk_size)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(p) > chunk_size:
            # hard-split an oversized paragraph
            for i in range(0, len(p), chunk_size - overlap):
                piece = p[i:i + chunk_size]
                if piece.strip():
                    chunks.append(piece.strip())
            continue
        if len(current) + len(p) + 2 <= chunk_size:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = (tail + "\n\n" + p).strip()
    if current:
        chunks.append(current)
    return chunks


def build_indices(kb_dir: str = KNOWLEDGE_BASE_DIR):
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    docs = load_documents(kb_dir)
    if not docs:
        raise RuntimeError(
            f"No .md or .pdf documents found in {kb_dir}. "
            f"Drop your source files there and re-run."
        )

    all_chunks = []
    for source_file, text in docs:
        title = os.path.splitext(source_file)[0].replace("_", " ").replace("-", " ")
        pieces = chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        for i, piece in enumerate(pieces):
            checksum = hashlib.md5(piece.encode("utf-8")).hexdigest()[:10]
            all_chunks.append(Chunk(
                chunk_id=f"{source_file}::{i}",
                text=piece,
                source_file=source_file,
                doc_title=title,
                chunk_index=i,
                checksum=checksum,
            ))

    print(f"Loaded {len(docs)} documents -> {len(all_chunks)} chunks.")

    # --- Dense embeddings ---
    print(f"Loading embedding model '{ACTIVE_EMBEDDING_MODEL}'...")
    embed_model = SentenceTransformer(ACTIVE_EMBEDDING_MODEL)
    texts = [c.text for c in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for c, e in zip(all_chunks, embeddings):
        c.embedding = e

    with open(DENSE_INDEX_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Saved dense index ({len(all_chunks)} chunks) -> {DENSE_INDEX_PATH}")

    # --- BM25 ---
    tokenized = [c.text.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in all_chunks]}, f)
    print(f"Saved BM25 index -> {BM25_INDEX_PATH}")

    return all_chunks


if __name__ == "__main__":
    build_indices()
