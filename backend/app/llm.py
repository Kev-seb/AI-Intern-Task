"""Gemini client helpers (model + key come from environment / backend/.env)."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()      # gemini | groq (OpenAI-compatible)
MODEL = os.getenv("LLM_MODEL") or (os.getenv("GEMINI_MODEL", "gemini-2.5-flash") if PROVIDER == "gemini"
                                   else "openai/gpt-oss-120b")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")


class MissingKeyError(RuntimeError):
    pass


def gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def api_key() -> str | None:
    """Key of the ACTIVE provider (used by the health endpoint)."""
    if PROVIDER == "gemini":
        return gemini_key()
    return os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")


def get_client():
    """Gemini client (also used for the optional Google-Search grounded web lookup)."""
    from google import genai
    from google.genai import types
    key = gemini_key()
    if not key:
        raise MissingKeyError("GEMINI_API_KEY is not set. Copy backend/.env.example to backend/.env and add your key "
                              "(free key: https://aistudio.google.com/apikey).")
    # function calling and search grounding need the v1beta API surface
    return genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta"))


def _retry_delay(msg: str) -> float | None:
    """Parse Google's 'Please retry in 16.5s' / retryDelay hints (free tier is only 5 requests/minute)."""
    m = re.search(r"retry in ([\d.]+)\s*(ms|s)", msg) or re.search(r"'retryDelay': '([\d.]+)()s'", msg)
    if not m:
        return None
    val = float(m.group(1))
    return val / 1000 if m.group(2) == "ms" else val


def with_retry(fn, attempts: int = 6):
    """Retry transient Gemini errors (rate limit / overload), honouring the server's suggested delay."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except MissingKeyError:
            raise
        except Exception as e:  # noqa: BLE001 - SDK raises several error types
            last = e
            msg = str(e)
            low = msg.lower()
            transient = any(t in low for t in ("429", "503", "500", "502", "overloaded", "unavailable", "resource_exhausted",
                                               "timeout", "timed out", "disconnected", "getaddrinfo", "connection",
                                               "remoteprotocol", "readerror", "ssl", "eof occurred", "reset by peer",
                                               "broken pipe", "temporary failure"))
            if not transient or i == attempts - 1:
                raise
            wait = min((_retry_delay(msg) or 2 * (2 ** i)) + 1.5, 70)
            if os.getenv("LLM_DEBUG"):
                print(f"[llm retry {i + 1}/{attempts}] waiting {wait:.0f}s: {msg[:160]}", flush=True)
            time.sleep(wait)
    raise last  # pragma: no cover


def plain_completion(prompt: str) -> str:
    """One-shot answer with NO tools and NO dataset (the baseline in the evaluation)."""
    if PROVIDER == "gemini":
        client = get_client()
        resp = with_retry(lambda: client.models.generate_content(model=MODEL, contents=prompt))
        return (resp.text or "").strip()
    import httpx
    key = api_key()
    if not key:
        raise MissingKeyError("No API key for the selected LLM provider (see backend/.env.example).")

    def call():
        r = httpx.post(f"{BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=90,
                       json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]})
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {r.text}")
        return r.json()["choices"][0]["message"]["content"]

    return (with_retry(call) or "").strip()


def grounded_search(query: str) -> dict:
    """External web lookup through Gemini's Google Search grounding. Always labelled EXTERNAL."""
    from google.genai import types
    client = get_client()

    def call():
        return client.models.generate_content(
            model=MODEL,
            contents=f"{query}\nAnswer in at most 6 sentences. Be specific about uncertainty.",
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.1))

    try:
        resp = with_retry(call)
    except Exception as e:  # noqa: BLE001
        return {"source_type": "EXTERNAL (web)", "error": f"web lookup failed: {e}", "sources": []}
    sources = []
    try:
        for ch in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if ch.web and ch.web.uri:
                sources.append({"title": ch.web.title, "url": ch.web.uri})
    except Exception:  # noqa: BLE001
        pass
    return {"source_type": "EXTERNAL (web) - not verified against the recipe dataset", "answer": resp.text or "",
            "sources": sources[:6]}
