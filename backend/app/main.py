"""FastAPI app. Run:  uvicorn app.main:app --port 8000"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .llm import MODEL, MissingKeyError, api_key
from .session import get_session, reset_session
from .store import get_store

app = FastAPI(title="Recipe Box Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/api/health")
def health():
    store = get_store()
    m = store.meta()
    return {"ok": True, "model": MODEL, "gemini_key_configured": bool(api_key()),
            "embeddings_loaded": store.has_embeddings,
            "dataset": {"raw_records": int(m["raw_records"]), "usable_recipes": int(m["complete"]),
                        "per_source": m["per_source"], "exact_duplicate_records": int(m.get("exact_duplicate_records", 0)),
                        "junk_records": int(m.get("junk_records", 0))}}


@app.post("/api/chat")
def chat_endpoint(body: ChatIn):
    from .agent import chat
    if not body.message.strip():
        raise HTTPException(400, "empty message")
    session = get_session(body.session_id)
    try:
        return chat(session, body.message.strip())
    except MissingKeyError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"LLM call failed: {e}") from e


@app.post("/api/reset")
def reset(body: ChatIn):
    if body.session_id:
        reset_session(body.session_id)
    return {"ok": True}


@app.get("/api/recipe/{recipe_id:path}")
def recipe(recipe_id: str):
    d = get_store().recipe(recipe_id, full=True)
    if not d:
        raise HTTPException(404, "recipe not found")
    return d
