"""In-memory chat sessions: conversation text + the structured state that follow-ups refine."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from .store import Filters


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    history: list[dict] = field(default_factory=list)       # [{"role": "user"|"model", "text": str}]
    search_query: str = ""
    search_filters: Filters = field(default_factory=Filters)
    last_ids: list[str] = field(default_factory=list)         # ids shown in the most recent result list
    last_recipe_id: str | None = None
    match_have: list = field(default_factory=list)
    match_filters: Filters = field(default_factory=Filters)
    match_prefs: dict = field(default_factory=lambda: {"fewest_missing": True, "fewest_steps": False})
    created: float = field(default_factory=time.time)

    def state_summary(self) -> str:
        lines = []
        if self.search_query or not self.search_filters.is_empty():
            lines.append(f"ACTIVE SEARCH: query='{self.search_query}'; filters: {self.search_filters.describe()}")
        if self.last_ids:
            lines.append("LAST RESULT IDS (in order): " + ", ".join(self.last_ids))
        if self.last_recipe_id:
            lines.append(f"LAST DISCUSSED RECIPE: {self.last_recipe_id}")
        if self.match_have:
            names = [h if isinstance(h, str) else h.get("name") for h in self.match_have]
            lines.append(f"ACTIVE INGREDIENT MATCH: have={names}; hard constraints: {self.match_filters.describe()}; "
                         f"prefs={self.match_prefs}")
        return "\n".join(lines) or "No active search or match yet."


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def get_session(session_id: str | None) -> Session:
    with _lock:
        if session_id and session_id in _sessions:
            return _sessions[session_id]
        s = Session(id=session_id or uuid.uuid4().hex[:12])
        _sessions[s.id] = s
        return s


def reset_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
