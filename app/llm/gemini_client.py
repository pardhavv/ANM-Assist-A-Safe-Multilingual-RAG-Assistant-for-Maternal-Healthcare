"""
Thin wrapper around the Gemini API. Kept as a single class so the rest of the
pipeline never imports google.generativeai directly — swapping to a different
provider later means editing only this file.
"""
import time
from dataclasses import dataclass

import google.generativeai as genai

from app.config.settings import GEMINI_API_KEY, GENERATION_MODEL, \
    PRICE_PER_1K_INPUT_TOKENS_USD, PRICE_PER_1K_OUTPUT_TOKENS_USD, USD_TO_INR

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Export it before calling the LLM.")
        genai.configure(api_key=GEMINI_API_KEY)
        _configured = True


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cost_inr: float


def generate(prompt: str, system_instruction: str = None, max_output_tokens: int = 500,
             temperature: float = 0.2) -> LLMResult:
    _ensure_configured()
    t0 = time.time()
    model = genai.GenerativeModel(GENERATION_MODEL, system_instruction=system_instruction)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ),
    )
    latency_ms = (time.time() - t0) * 1000

    text = (response.text or "").strip()

    # Gemini returns usage metadata on response.usage_metadata
    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    cost_usd = (input_tokens / 1000) * PRICE_PER_1K_INPUT_TOKENS_USD + \
               (output_tokens / 1000) * PRICE_PER_1K_OUTPUT_TOKENS_USD
    cost_inr = cost_usd * USD_TO_INR

    return LLMResult(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens,
        latency_ms=latency_ms, cost_usd=cost_usd, cost_inr=cost_inr,
    )
