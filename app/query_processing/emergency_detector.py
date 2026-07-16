"""
Emergency detection: fast, deterministic keyword scan, run BEFORE retrieval.

Design choice: this is intentionally NOT an LLM call. Emergency detection is
the one place in the pipeline where latency directly costs safety margin —
"never delay emergency recommendations because of retrieval" is a stated
requirement, so this stage must be sub-millisecond. A keyword scan trades a
bit of recall for that speed guarantee; it's paired with the confidence
engine downstream, which can still catch danger-sign language the keyword
list misses, so this isn't the only safety net.
"""
import re
from app.config.settings import EMERGENCY_KEYWORDS


def detect_emergency(text: str) -> dict:
    lowered = text.lower()
    matched = [kw for kw in EMERGENCY_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", lowered)]
    return {
        "is_emergency": len(matched) > 0,
        "matched_keywords": matched,
    }


if __name__ == "__main__":
    tests = [
        "She has severe headache and blurred vision",
        "How many ANC visits are recommended",
        "Baby not breathing after delivery",
    ]
    for t in tests:
        print(t, "->", detect_emergency(t))
