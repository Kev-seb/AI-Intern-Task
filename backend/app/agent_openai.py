"""Tool-calling loop for OpenAI-compatible chat APIs (Groq, OpenRouter, Ollama, Mistral...)."""
from __future__ import annotations

import json

import httpx

from .llm import BASE_URL, MODEL, MissingKeyError, api_key, with_retry
from .router import route_tool
from .session import Session
from .tools import build_tools

MAX_STEPS = 8
MAX_TOOL_CHARS = 4200  # keep tool output inside small free-tier token budgets
ALLOWED = {"type", "description", "properties", "items", "enum", "required"}


def _clean_schema(o):
    """Gemini-style schema -> plain JSON schema (lower-case types, only keys every provider accepts)."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == "properties" and isinstance(v, dict):
                out[k] = {name: _clean_schema(s) for name, s in v.items()}
            elif k in ALLOWED:
                out[k] = v.lower() if k == "type" and isinstance(v, str) else _clean_schema(v)
        return out
    if isinstance(o, list):
        return [_clean_schema(i) for i in o]
    return o


def tool_specs(tools: list) -> list[dict]:
    from google import genai
    from google.genai import types
    dummy = genai.Client(api_key="unused")      # only used to derive schemas from the Python signatures
    specs = []
    for fn in tools:
        decl = types.FunctionDeclaration.from_callable(client=dummy, callable=fn).model_dump(mode="json", exclude_none=True)
        params = _clean_schema(decl.get("parameters") or {"type": "object", "properties": {}})
        for prop in (params.get("properties") or {}).values():     # compact for small free-tier token budgets
            if isinstance(prop.get("description"), str):
                prop["description"] = prop["description"][:110]
        desc = (decl.get("description") or "").strip()
        desc = desc.split("\n\n")[0] if len(desc) > 520 else desc
        specs.append({"type": "function", "function": {"name": decl["name"], "description": desc[:520],
                                                       "parameters": params}})
    return specs


DROP_KEYS = {"sql", "normalization", "dataset", "matched_forms", "components", "indirect_evidence_note",
             "original_instructions", "original_ingredients", "scoring", "source", "weights", "retrieval",
             "your_ingredients_unused", "n_missing", "duplicate_records", "exclusion_note", "similar_recipes",
             "safety_note", "skipped_because_already_in_recipe", "other_or_alternatives_seen_unvetted"}


def _shrink(o, depth: int = 0):
    """Compact a tool result for the model only (the UI trace keeps the full result): drop bulky bookkeeping,
    cap list lengths and long strings so small free-tier token budgets are respected."""
    if isinstance(o, dict):
        return {k: _shrink(v, depth + 1) for k, v in o.items() if k not in DROP_KEYS}
    if isinstance(o, list):
        return [_shrink(v, depth + 1) for v in o[:6]]
    if isinstance(o, str) and len(o) > 150:
        return o[:150] + "…"
    return o


def _compact_match(res: dict) -> dict:
    """Purpose-built short form of a match_by_ingredients result for small token budgets."""
    def short(r: dict) -> dict:
        return {"id": r["id"], "title": r["title"], "steps": r["n_steps"], "score": r["score"], "why": r["why"],
                "matched": [f"{m['you_have']} = {m['recipe_line'][:48]} ({m['match']}; {str(m.get('quantity', ''))[:42]})"
                            for m in r["matched"][:6]],
                "missing": [x["ingredient"] for x in r["missing"]][:8], "pantry_assumed": r.get("assumed_pantry", [])}

    out = {k: res[k] for k in ("hard_constraints", "preferences", "eligible_recipes_containing_any_ingredient", "notes",
                              "no_match", "alternatives_warning", "error") if k in res}
    out["you_have"] = [h["name"] for h in res.get("you_have", [])]
    out["scoring"] = str(res.get("scoring", ""))[:240]
    out["results"] = [short(r) for r in res.get("results", [])]
    if res.get("closest_violating_alternatives"):
        out["closest_violating_alternatives"] = [
            {"violates": a["violates"], "results": [{"id": x["id"], "title": x["title"], "steps": x["n_steps"]} for x in a["results"]]}
            for a in res["closest_violating_alternatives"]]
    return out


def _post(key: str, payload: dict) -> dict:
    def call():
        try:
            r = httpx.post(f"{BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload,
                           timeout=120)
        except httpx.TransportError as e:   # dropped connection / TLS EOF / DNS: safe to retry
            raise RuntimeError(f"connection error: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {r.text}")
        return r.json()
    return with_retry(call)


def chat_openai(session: Session, message: str, system_prompt: str) -> dict:
    key = api_key()
    if not key:
        raise MissingKeyError("No API key for the selected LLM provider. Set GROQ_API_KEY (or LLM_API_KEY) in backend/.env.")
    tools = build_tools(session)
    registry = {fn.__name__: fn for fn in tools}
    specs = tool_specs(tools)
    messages = [{"role": "system", "content": system_prompt + "\n\nSESSION STATE\n" + session.state_summary()}]
    messages += [{"role": "assistant" if h["role"] == "model" else "user", "content": h["text"]} for h in session.history[-16:]]
    messages.append({"role": "user", "content": message})

    trace, reply = [], ""
    forced = route_tool(message)
    for step in range(MAX_STEPS):
        if forced and step == 0:
            # some models ignore tool_choice, so on a forced step only the required tool is offered at all
            step_specs = [s for s in specs if s["function"]["name"] == forced]
            choice = {"type": "function", "function": {"name": forced}}
        else:
            step_specs, choice = specs, "auto"
        try:
            data = _post(key, {"model": MODEL, "messages": messages, "tools": step_specs, "tool_choice": choice,
                               "temperature": 0.2})
        except Exception as e:  # noqa: BLE001
            # a forced step can be rejected when the model insists on another tool; fall back to free tool choice
            if not (forced and step == 0 and "tool_use_failed" in str(e)):
                raise
            data = _post(key, {"model": MODEL, "messages": messages, "tools": specs, "tool_choice": "auto",
                               "temperature": 0.2})
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            reply = (msg.get("content") or "").strip()
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                args = {k: v for k, v in args.items() if v is not None}
                result = registry[name](**args) if name in registry else {"error": f"unknown tool {name}"}
            except Exception as e:  # noqa: BLE001 - let the model see the failure and retry with better arguments
                result = {"error": f"{type(e).__name__}: {e}"}
            result = json.loads(json.dumps(result, default=str))
            trace.append({"name": name, "args": args if isinstance(args, dict) else {}, "result": result})
            model_view = _compact_match(result) if name == "match_by_ingredients" and "results" in result else _shrink(result)
            text = json.dumps(model_view, default=str)
            if len(text) > MAX_TOOL_CHARS:
                text = text[:MAX_TOOL_CHARS] + '... [truncated for length; full result is in the evidence trace]'
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
    if not reply:
        reply = "I could not produce an answer for that. Please rephrase or give more detail."
    session.history.append({"role": "user", "text": message})
    session.history.append({"role": "model", "text": reply})
    return {"reply": reply, "tool_calls": trace, "session_id": session.id, "state": session.state_summary(), "model": MODEL}
