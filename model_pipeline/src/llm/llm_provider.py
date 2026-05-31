"""
LLM provider abstraction for SavVio.

Active provider: OpenRouter (paid, no rate limits).
  Model: google/gemini-2.5-flash
  Why: strong instruction-following for structured 4-part financial advice prompts,
       production-stable (no deprecation risk), ~$0.30/M input + $2.50/M output tokens.

Fallback: MockProvider (deterministic stub for tests / no-key environments).

Free-tier providers (Gemini direct, Groq) are preserved below but commented out.
Re-enable by swapping get_provider() and uncommenting the class definitions.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared system message — defines SavVio's voice for all real providers.
# ---------------------------------------------------------------------------
_SYSTEM_MESSAGE = (
    "You are SavVio, a fiduciary financial advisor. "
    "Your job is to explain a purchase recommendation to the user in clear, honest, specific language. "
    "Always reference the exact numbers given — never invent or round financial facts. "
    "When customer reviews are provided, quote or paraphrase real details from them. "
    "Follow the output structure instructions in the user message exactly."
)

# ---------------------------------------------------------------------------
# Retry helper — handles transient 429 / 503 errors.
# ---------------------------------------------------------------------------
_RETRYABLE_CODES = {429, 503}
_RETRY_DELAYS = (4, 10)  # seconds; two retries before propagating the error


def _post_with_retry(req: urllib.request.Request, timeout: int = 60) -> bytes:
    """POST the request, retrying on retryable HTTP errors with backoff."""
    last_exc: Exception = RuntimeError("no attempts made")
    delays = list(_RETRY_DELAYS) + [None]
    for attempt, delay in enumerate(delays, start=1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in _RETRYABLE_CODES and delay is not None:
                logger.warning("HTTP %s on attempt %d — retrying in %ds", e.code, attempt, delay)
                time.sleep(delay)
            else:
                raise
    raise last_exc


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Minimal interface used by intent_parser and response_generator."""

    provider_name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Return model text completion for the given prompt."""


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
# Older code paths (and some modules in this repo) import `LLMProvider` from
# this module. The current abstraction is `BaseLLMProvider`, so we alias it.
# This avoids runtime ImportError in the API inference pipeline.
LLMProvider = BaseLLMProvider


# ---------------------------------------------------------------------------
# Mock provider — deterministic stub for tests / no-key environments.
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    provider_name = "mock"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        if "intent" in prompt.lower() and "json" in prompt.lower():
            return '{"intent": "purchase_query", "product_reference": null, "user_context": null}'
        return (
            "Recommendation follows the deterministic engine. "
            "Review your emergency fund and debt obligations before purchasing."
        )


# ---------------------------------------------------------------------------
# OpenRouter provider — active, paid, no rate limits.
# ---------------------------------------------------------------------------

class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter unified API (OpenAI-compatible format).

    Model: google/gemini-2.5-flash
      - Strong instruction-following for structured financial advice prompts
      - ~$0.30 / M input tokens, $2.50 / M output tokens
      - Production-stable (no deprecation), thinking mode off by default
      - No RPM / RPD limits on paid credits

    Switch model via OPENROUTER_MODEL env var if needed, e.g.:
      meta-llama/llama-3.3-70b-instruct  (~$0.12/M in, $0.30/M out)
      mistralai/mistral-small-3.1-24b-instruct  (~$0.10/M in, $0.30/M out)
    """

    provider_name = "openrouter"
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"
    _DEFAULT_MODEL = "google/gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("OPEN_ROUTER_API_KEY is not set")
        self._api_key = key
        self._model = model or os.environ.get("OPENROUTER_MODEL", self._DEFAULT_MODEL)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            self._API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": "https://savvio.ai",
                "X-Title": "SavVio",
            },
            method="POST",
        )
        raw = _post_with_retry(req)
        data = json.loads(raw.decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Free-tier providers — commented out; preserved for reference / fallback.
# ---------------------------------------------------------------------------

# class GeminiProvider(BaseLLMProvider):
#     """
#     Google Gemini direct API (free tier: 15 RPM / 1,500 RPD).
#     Exhausts daily quota quickly under load. Use OpenRouter instead.
#     Re-enable by uncommenting and adding to get_provider() chain.
#     """
#     provider_name = "gemini"
#     _DEFAULT_MODEL = "gemini-2.0-flash"
#
#     def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
#         key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
#         if not key:
#             raise ValueError("GEMINI_API_KEY is not set")
#         self._api_key = key
#         self._model = model or os.environ.get("GEMINI_MODEL", self._DEFAULT_MODEL)
#
#     def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
#         url = (
#             f"https://generativelanguage.googleapis.com/v1beta/models/"
#             f"{self._model}:generateContent?key={self._api_key}"
#         )
#         body = {
#             "system_instruction": {"parts": [{"text": _SYSTEM_MESSAGE}]},
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
#         }
#         req = urllib.request.Request(
#             url,
#             data=json.dumps(body).encode("utf-8"),
#             headers={"Content-Type": "application/json"},
#             method="POST",
#         )
#         raw = _post_with_retry(req)
#         data = json.loads(raw.decode("utf-8"))
#         return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# class GroqProvider(BaseLLMProvider):
#     """
#     Groq free-tier API (llama-3.3-70b-versatile).
#     Good quality but rate-limited; requires User-Agent: SavVio/1.0 to
#     bypass Cloudflare 403. Use OpenRouter for production instead.
#     Re-enable by uncommenting and adding to get_provider() chain.
#     """
#     provider_name = "groq"
#     _DEFAULT_MODEL = "llama-3.3-70b-versatile"
#
#     def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
#         key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
#         if not key:
#             raise ValueError("GROQ_API_KEY is not set")
#         self._api_key = key
#         self._model = model or os.environ.get("GROQ_MODEL", self._DEFAULT_MODEL)
#
#     def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
#         body = {
#             "model": self._model,
#             "messages": [
#                 {"role": "system", "content": _SYSTEM_MESSAGE},
#                 {"role": "user", "content": prompt},
#             ],
#             "max_tokens": max_tokens,
#             "temperature": temperature,
#         }
#         req = urllib.request.Request(
#             "https://api.groq.com/openai/v1/chat/completions",
#             data=json.dumps(body).encode("utf-8"),
#             headers={
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {self._api_key}",
#                 "User-Agent": "SavVio/1.0",  # Required — Cloudflare blocks Python default UA
#             },
#             method="POST",
#         )
#         raw = _post_with_retry(req)
#         data = json.loads(raw.decode("utf-8"))
#         return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider() -> BaseLLMProvider:
    """
    Return the active LLM provider.

    Priority: OpenRouter → Mock (no key configured).
    Free-tier providers (Gemini, Groq) are disabled — see commented classes above.
    """
    if os.environ.get("OPEN_ROUTER_API_KEY", "").strip():
        try:
            p = OpenRouterProvider()
            logger.info("Using LLM provider: %s (%s)", p.provider_name, p._model)
            return p
        except Exception as e:
            logger.warning("Could not init OpenRouterProvider: %s — falling back to mock", e)

    logger.info("Using LLM provider: mock (OPEN_ROUTER_API_KEY not set)")
    return MockProvider()
