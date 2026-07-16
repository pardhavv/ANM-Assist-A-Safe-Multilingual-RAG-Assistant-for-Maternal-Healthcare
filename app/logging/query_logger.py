"""
SQLite logging: every query gets logged with the fields called out in the
spec (timestamp, original/rewritten query, language, latency breakdown,
tokens, cost, confidence, escalation, sources). SQLite chosen over flat
JSONL for this because the eval/dashboard queries (avg cost by day, escalation
rate by language, etc.) are naturally SQL aggregations — trivial to upgrade
to Postgres later by swapping the connection string, since we only use
standard SQL here.
"""
import sqlite3
import json
import time
from contextlib import contextmanager

from app.config.settings import SQLITE_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    original_query TEXT NOT NULL,
    normalized_query TEXT,
    detected_language TEXT,
    is_emergency INTEGER,
    retrieval_latency_ms REAL,
    generation_latency_ms REAL,
    total_latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    cost_inr REAL,
    confidence_score REAL,
    confidence_signals TEXT,
    escalated INTEGER,
    escalation_reason TEXT,
    sources TEXT,
    answer TEXT
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_query(**fields):
    init_db()
    fields.setdefault("timestamp", time.time())
    if "confidence_signals" in fields and isinstance(fields["confidence_signals"], dict):
        fields["confidence_signals"] = json.dumps(fields["confidence_signals"])
    if "sources" in fields and isinstance(fields["sources"], list):
        fields["sources"] = json.dumps(fields["sources"])

    columns = ", ".join(fields.keys())
    placeholders = ", ".join(["?"] * len(fields))
    with _conn() as conn:
        conn.execute(f"INSERT INTO query_logs ({columns}) VALUES ({placeholders})", list(fields.values()))
        conn.commit()


def get_summary_stats():
    init_db()
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT
                COUNT(*) as n_queries,
                AVG(total_latency_ms) as avg_latency_ms,
                AVG(cost_inr) as avg_cost_inr,
                SUM(cost_inr) as total_cost_inr,
                SUM(escalated) * 1.0 / COUNT(*) as escalation_rate,
                SUM(is_emergency) as emergency_count
            FROM query_logs
        """).fetchone()
        return dict(row) if row else {}
