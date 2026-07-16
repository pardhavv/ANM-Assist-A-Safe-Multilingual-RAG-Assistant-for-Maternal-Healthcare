"""
Central, env-driven settings. Nothing hardcoded, everything overridable.
"""
import os

# --- LLM (Gemini) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Accept both GENERATION_MODEL and MODEL_NAME (some setups use the shorter name)
GENERATION_MODEL = os.environ.get("GENERATION_MODEL") or os.environ.get("MODEL_NAME", "gemini-2.5-flash")

# Approximate Gemini 2.5 Flash pricing (USD per 1K tokens) — update if it changes.
# Used only to produce the cost-per-query estimate in logs/eval.
PRICE_PER_1K_INPUT_TOKENS_USD = float(os.environ.get("PRICE_PER_1K_INPUT_TOKENS_USD", "0.000075"))
PRICE_PER_1K_OUTPUT_TOKENS_USD = float(os.environ.get("PRICE_PER_1K_OUTPUT_TOKENS_USD", "0.0003"))
USD_TO_INR = float(os.environ.get("USD_TO_INR", "87"))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", os.path.join(BASE_DIR, "knowledge_base"))
VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR", os.path.join(BASE_DIR, "vector_store"))
DENSE_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "dense_chunks.pkl")
BM25_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "bm25_index.pkl")
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(BASE_DIR, "logs.db"))

# --- Chunking ---
CHUNK_SIZE_CHARS = int(os.environ.get("CHUNK_SIZE_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "150"))

# --- Embeddings ---
# Multilingual embedding model so Hindi/Kannada/Telugu queries can match English
# source docs (and vice versa) without a translation hop for retrieval itself.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or os.environ.get("MODEL_NAME_EMBED", "paraphrase-multilingual-MiniLM-L12-v2")

# If a fine-tuned embedder exists (see app/training/finetune_embeddings.py) and
# USE_FINETUNED_EMBEDDER=true, retrieval uses it instead of the base model above.
USE_FINETUNED_EMBEDDER = os.environ.get("USE_FINETUNED_EMBEDDER", "false").lower() == "true"
FINETUNED_EMBEDDER_PATH = os.environ.get(
    "FINETUNED_EMBEDDER_PATH", os.path.join(BASE_DIR, "models", "finetuned-embedder")
)
ACTIVE_EMBEDDING_MODEL = (
    FINETUNED_EMBEDDER_PATH if USE_FINETUNED_EMBEDDER and os.path.isdir(FINETUNED_EMBEDDER_PATH)
    else EMBEDDING_MODEL
)

# --- Retrieval ---
# If a generic TOP_K is set (simpler .env convention), use it to size both the
# reranker's final output and, loosely, the retrieval fan-in; the more specific
# DENSE_TOP_K/BM25_TOP_K/FUSED_TOP_K/FINAL_TOP_K env vars still take precedence
# if set explicitly, since they control different stages.
_GENERIC_TOP_K = os.environ.get("TOP_K")

DENSE_TOP_K = int(os.environ.get("DENSE_TOP_K", "10"))
BM25_TOP_K = int(os.environ.get("BM25_TOP_K", "10"))
RRF_K = int(os.environ.get("RRF_K", "60"))          # standard RRF constant
FUSED_TOP_K = int(os.environ.get("FUSED_TOP_K", "8"))   # candidates passed to reranker
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K") or _GENERIC_TOP_K or "4")    # chunks passed to the LLM
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Confidence / escalation ---
# Below this fused confidence score, escalate instead of answering.
# Accept both CONFIDENCE_ESCALATION_THRESHOLD and the shorter CONFIDENCE_THRESHOLD.
CONFIDENCE_ESCALATION_THRESHOLD = float(
    os.environ.get("CONFIDENCE_ESCALATION_THRESHOLD") or os.environ.get("CONFIDENCE_THRESHOLD", "0.45")
)

# --- Telegram (not yet wired to a bot.py in this version — see README) ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# --- Languages ---
SUPPORTED_LANGUAGES = ["en", "hi", "kn", "te"]

# --- Medical acronyms & normalization ---
MEDICAL_ACRONYMS = {
    "ANC": "Antenatal Care",
    "IFA": "Iron and Folic Acid",
    "TT": "Tetanus Toxoid",
    "LMP": "Last Menstrual Period",
    "HB": "Hemoglobin",
    "BP": "Blood Pressure",
    "PNC": "Postnatal Care",
    "KMC": "Kangaroo Mother Care",
    "EDD": "Expected Date of Delivery",
}

MEDICAL_TERM_NORMALIZATION = {
    "sugar": "blood glucose",
    "fits": "seizure",
    "fit": "seizure",
    "water broke": "rupture of membranes",
    "baby not moving": "decreased fetal movement",
    "not moving": "decreased fetal movement",
    "high bp": "high blood pressure",
    "swelling": "edema",
}

EMERGENCY_KEYWORDS = [
    "heavy bleeding", "severe bleeding", "convulsion", "convulsions", "seizure", "fits", "fit",
    "difficulty breathing", "can't breathe", "cannot breathe", "unconscious", "unconsciousness",
    "no fetal movement", "not moving", "severe headache", "blurred vision", "blurry vision",
    "very high blood pressure", "chest pain", "high fever", "not breathing", "baby not breathing",
    "foul smell", "foul-smelling discharge", "soaking a pad",
]
