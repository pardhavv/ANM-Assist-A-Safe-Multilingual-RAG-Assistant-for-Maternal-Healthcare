#!/usr/bin/env python3
"""
One-command CLI entry point. This is the "just run it" script.

Usage:
    python run_query.py "BP high, what should I do?"
    python run_query.py --lang-skip "What is the recommended IFA dosage?"

First-time setup (run once):
    pip install -r requirements.txt
    cp .env.example .env   # fill in GEMINI_API_KEY
    export $(cat .env | xargs)
    python -m app.retriever.ingest      # builds the vector store from knowledge_base/
"""
import sys
import json
from dataclasses import asdict

from app.rag.pipeline import run_pipeline


def main():
    args = sys.argv[1:]
    skip_translation = "--lang-skip" in args
    args = [a for a in args if a != "--lang-skip"]

    if not args:
        question = "A pregnant woman at 32 weeks has a severe headache and blurred vision. What should I do?"
        print(f"No question given, using default demo question:\n  {question}\n")
    else:
        question = " ".join(args)

    result = run_pipeline(question, skip_translation=skip_translation)

    print("=" * 70)
    print(f"QUESTION ({result.detected_language}): {result.original_query}")
    print("=" * 70)
    print(f"Emergency detected: {result.is_emergency}")
    print(f"Escalated: {result.escalated}" + (f" ({result.escalation_reason})" if result.escalated else ""))
    print(f"Confidence: {result.confidence_score}  {result.confidence_signals}")
    print(f"Sources: {result.sources}")
    print(f"Latency: retrieval={result.retrieval_latency_ms}ms generation={result.generation_latency_ms}ms total={result.total_latency_ms}ms")
    print(f"Cost: ₹{result.cost_inr}")
    print("-" * 70)
    print(result.answer)
    print("=" * 70)


if __name__ == "__main__":
    main()
