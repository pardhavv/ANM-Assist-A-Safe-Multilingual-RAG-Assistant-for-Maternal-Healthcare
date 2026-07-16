"""
Multilingual layer: language detection + translation, via Gemini.

Design choice — why translate rather than rely purely on the multilingual
embedding model for cross-lingual retrieval:
    The embedding model (paraphrase-multilingual-MiniLM) lets Hindi/Kannada/
    Telugu queries retrieve relevant English source chunks directly, without
    translation — that's the retrieval path. But the LLM's grounded answer
    still needs to come back in the user's language, and free-text medical
    phrasing benefits from an explicit translation pass rather than hoping a
    single multilingual generation prompt handles code-switching cleanly.
    So: retrieval is translation-free (embeddings do the work), but the
    final user-facing response goes through an explicit translate step.
"""
from app.llm.gemini_client import generate
from app.config.settings import SUPPORTED_LANGUAGES

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "te": "Telugu"}


def detect_language(text: str) -> str:
    prompt = (
        f"Detect the language of this text. Respond with ONLY one of these codes: "
        f"{', '.join(SUPPORTED_LANGUAGES)}. Text: {text}"
    )
    result = generate(prompt, max_output_tokens=10, temperature=0.0)
    code = result.text.strip().lower()
    return code if code in SUPPORTED_LANGUAGES else "en"


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text
    prompt = f"Translate this {LANGUAGE_NAMES.get(source_lang, source_lang)} text to English. Respond with ONLY the translation:\n\n{text}"
    result = generate(prompt, max_output_tokens=400, temperature=0.0)
    return result.text.strip()


def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    prompt = f"Translate this English text to {LANGUAGE_NAMES.get(target_lang, target_lang)}. Respond with ONLY the translation:\n\n{text}"
    result = generate(prompt, max_output_tokens=500, temperature=0.0)
    return result.text.strip()
