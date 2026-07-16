"""
Query normalization: acronym expansion + informal-term normalization.

Design choice: this is a fast, deterministic, dictionary-based pass that runs
BEFORE the (slower, costlier) LLM query rewrite. It handles the common,
predictable cases (BP -> Blood Pressure, "fits" -> seizure) for free and with
zero latency, so the LLM rewrite step only has to handle genuinely ambiguous
phrasing. This mirrors a general pattern worth knowing: cheap deterministic
passes first, expensive LLM calls last.
"""
import re
from app.config.settings import MEDICAL_ACRONYMS, MEDICAL_TERM_NORMALIZATION


def expand_acronyms(text: str) -> str:
    def replace(match):
        word = match.group(0)
        expansion = MEDICAL_ACRONYMS.get(word.upper())
        return f"{word} ({expansion})" if expansion else word

    # Match whole uppercase-ish tokens (2-4 letters) as likely acronyms
    return re.sub(r"\b[A-Za-z]{2,4}\b", lambda m: replace(m) if m.group(0).upper() in MEDICAL_ACRONYMS else m.group(0), text)


def normalize_terms(text: str) -> str:
    lowered = text.lower()
    seen_formals = set()
    # Sort longest-phrase-first so multi-word matches (e.g. "water broke") are
    # checked before shorter substrings that could double-match ("fit" inside "fits").
    for informal, formal in sorted(MEDICAL_TERM_NORMALIZATION.items(), key=lambda kv: -len(kv[0])):
        pattern = r"\b" + re.escape(informal) + r"\b"
        if re.search(pattern, lowered) and formal not in seen_formals:
            text = f"{text} [{formal}]"
            seen_formals.add(formal)
    return text


def normalize_query(text: str) -> str:
    text = expand_acronyms(text)
    text = normalize_terms(text)
    return text


if __name__ == "__main__":
    tests = ["BP high", "She had fits yesterday", "baby not moving since morning", "Hb 6.5, is that low?"]
    for t in tests:
        print(f"{t!r} -> {normalize_query(t)!r}")
