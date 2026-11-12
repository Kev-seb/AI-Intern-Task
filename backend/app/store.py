"""Data access: SQLite store, ingredient vocabulary resolution, structured filters, hybrid search."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .normalize import match_level, normalize_term, resolve_category, variant_note

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"
STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "without", "for", "of", "in", "on", "to", "me", "i", "can", "cannot",
    "find", "show", "give", "recipe", "recipes", "want", "some", "any", "please", "something", "that", "is", "are",
    "make", "making", "how", "what", "which", "my", "have", "has", "no", "not", "dish", "dishes", "using", "use",
    "good", "best", "easy", "quick", "need", "looking", "suggest", "like", "would", "could", "tell", "about",
}


@dataclass
class Filters:
    """Structured constraints. All lists hold *user-facing terms*; resolution happens in the store."""
    include: list[str] = field(default_factory=list)          # every term must be present (strict match)
    exclude: list[str] = field(default_factory=list)          # ingredient terms that must be absent (broad)
    exclude_categories: list[str] = field(default_factory=list)  # dairy, egg, peanut, gluten, vegan ...
    max_steps: int | None = None
    min_steps: int | None = None
    max_ingredients: int | None = None
    min_ingredients: int | None = None
    sources: list[str] = field(default_factory=list)          # ar | epi | fn
    title_contains: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    def copy(self) -> "Filters":
        return Filters(**json.loads(json.dumps(asdict(self))))

    def describe(self) -> str:
        parts = []
        if self.include:
            parts.append("contains " + " + ".join(self.include))
        if self.exclude:
            parts.append("excludes " + ", ".join(self.exclude))
        if self.exclude_categories:
            parts.append("free of " + ", ".join(self.exclude_categories))
        if self.max_steps is not None:
            parts.append(f"<= {self.max_steps} steps")
        if self.min_steps is not None:
            parts.append(f">= {self.min_steps} steps")
        if self.max_ingredients is not None:
            parts.append(f"<= {self.max_ingredients} ingredients")
        if self.min_ingredients is not None:
            parts.append(f">= {self.min_ingredients} ingredients")
        if self.sources:
            parts.append("source in " + ",".join(self.sources))
        if self.title_contains:
            parts.append("title contains " + ", ".join(self.title_contains))
        return "; ".join(parts) or "no filters"


def _dedupe(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


class Store:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or STORAGE / "recipes.db"
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} not found. Run: python scripts/build_db.py")
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._vocab: dict[str, int] | None = None
        self._inv: dict[str, set[str]] | None = None
        self._emb: np.ndarray | None = None
        self._emb_rids: np.ndarray | None = None
        self._embedder = None
        self._resolve_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------ basics
    def q(self, sql: str, params: tuple | list = ()):
        with self._lock:
            return self.con.execute(sql, params).fetchall()

    def meta(self) -> dict:
        m = {r["key"]: r["value"] for r in self.q("SELECT key, value FROM meta")}
        m["per_source"] = json.loads(m.get("per_source", "{}"))
        return m

    # ------------------------------------------------------------------ vocabulary
    def _load_vocab(self):
        if self._vocab is not None:
            return
        rows = self.q("SELECT canon, COUNT(*) c FROM recipe_ingredients GROUP BY canon")
        self._vocab = {r["canon"]: r["c"] for r in rows}
        inv: dict[str, set[str]] = {}
        for canon in self._vocab:
            for tok in canon.split():
                inv.setdefault(tok, set()).add(canon)
        self._inv = inv

    def resolve_term(self, term: str, mode: str = "strict") -> list[tuple[str, str]]:
        """Resolve a user term to [(canonical ingredient, level)] present in the dataset.

        mode='strict': exact + harmless variants (red onion for onion). Different products never match.
        mode='exact' : only the identical canonical name.
        mode='broad' : every canonical name containing all query tokens (peanut butter for peanut).
        """
        key = (term, mode)
        if key in self._resolve_cache:
            return self._resolve_cache[key]
        self._load_vocab()
        assert self._vocab is not None and self._inv is not None
        canon = normalize_term(term)
        toks = canon.split()
        out: list[tuple[str, str]] = []
        if toks and mode == "exact":
            return [(canon, "exact")] if canon in self._vocab else []
        if toks:
            cands: set[str] = set()
            for t in toks:
                cands |= self._inv.get(t, set())
            if canon in self._vocab:
                out.append((canon, "exact"))
            for c in cands:
                if c == canon:
                    continue
                if mode == "broad":
                    if set(toks) <= set(c.split()):
                        out.append((c, "broad"))
                else:
                    lvl = match_level(canon, c)
                    if lvl:
                        out.append((c, lvl))
        out.sort(key=lambda x: -self._vocab.get(x[0], 0))
        self._resolve_cache[key] = out
        return out

    def display_forms(self, term: str, mode: str = "strict", limit: int = 6) -> list[str]:
        return [c for c, _ in self.resolve_term(term, mode)[:limit]]

    # ------------------------------------------------------------------ filters -> SQL
    def _where(self, f: Filters, include_mode: str = "strict") -> tuple[str, list]:
        clauses = ["r.complete = 1"]
        params: list = []
        if f.sources:
            clauses.append(f"r.source IN ({','.join('?' * len(f.sources))})")
            params += f.sources
        for col, op, val in (("n_steps", "<=", f.max_steps), ("n_steps", ">=", f.min_steps),
                             ("n_ingredients", "<=", f.max_ingredients), ("n_ingredients", ">=", f.min_ingredients)):
            if val is not None:
                clauses.append(f"r.{col} {op} ?")
                params.append(val)
        for t in f.title_contains:
            clauses.append("LOWER(r.title) LIKE ?")
            params.append(f"%{t.lower()}%")
        for term in f.include:
            forms = [c for c, _ in self.resolve_term(term, include_mode)]
            if not forms:
                clauses.append("0")   # an unknown ingredient can never be satisfied: no silent relaxation
                continue
            clauses.append(f"r.rid IN (SELECT i.rid FROM recipe_ingredients i WHERE i.canon IN ({','.join('?' * len(forms))}))")
            params += forms
        for term in f.exclude:
            forms = [c for c, _ in self.resolve_term(term, "broad")]
            cats = resolve_category(term)
            if forms:
                clauses.append(f"r.rid NOT IN (SELECT i.rid FROM recipe_ingredients i WHERE i.canon IN ({','.join('?' * len(forms))}))")
                params += forms
            if cats:
                clauses.append(f"r.rid NOT IN (SELECT c.rid FROM recipe_cat c WHERE c.level = 'definite' AND c.cat IN ({','.join('?' * len(cats))}))")
                params += cats
        cats_all: list[str] = []
        for c in f.exclude_categories:
            cats_all += resolve_category(c)
        cats_all = _dedupe(cats_all)
        if cats_all:
            clauses.append(f"r.rid NOT IN (SELECT c.rid FROM recipe_cat c WHERE c.level = 'definite' AND c.cat IN ({','.join('?' * len(cats_all))}))")
            params += cats_all
        return " AND ".join(clauses), params

    def filtered_rids(self, f: Filters, include_mode: str = "strict") -> list[int]:
        where, params = self._where(f, include_mode)
        return [r[0] for r in self.q(f"SELECT r.rid FROM recipes r WHERE {where}", params)]

    def count(self, f: Filters, include_mode: str = "strict") -> int:
        where, params = self._where(f, include_mode)
        return self.q(f"SELECT COUNT(*) FROM recipes r WHERE {where}", params)[0][0]

    def meta_table(self) -> dict[int, tuple]:
        """rid -> (id, title, source, n_steps, n_ingredients, dup_of) for complete recipes (cached in memory)."""
        if getattr(self, "_meta_table", None) is None:
            rows = self.q("SELECT rid, id, title, source, n_steps, n_ingredients, dup_of FROM recipes WHERE complete = 1")
            self._meta_table = {r[0]: tuple(r[1:]) for r in rows}
        return self._meta_table

    # ------------------------------------------------------------------ recipes
    def recipe(self, ident: str | int, full: bool = True) -> dict | None:
        if isinstance(ident, int) or str(ident).isdigit():
            rows = self.q("SELECT * FROM recipes WHERE rid = ?", (int(ident),))
        else:
            rows = self.q("SELECT * FROM recipes WHERE id = ?", (ident,))
            if not rows:  # allow unique-prefix lookup of an id
                rows = self.q("SELECT * FROM recipes WHERE id LIKE ? LIMIT 2", (f"{ident}%",))
                if len(rows) != 1:
                    rows = []
        if not rows:
            return None
        r = rows[0]
        d = {"rid": r["rid"], "id": r["id"], "source": r["source"], "title": r["title"],
             "n_ingredients": r["n_ingredients"], "n_steps": r["n_steps"], "complete": bool(r["complete"])}
        if full:
            d["ingredients"] = json.loads(r["ingredients_json"] or "[]")
            d["instructions"] = r["instructions"]
        return d

    def recipes_by_rids(self, rids: list[int], full: bool = False) -> list[dict]:
        out = []
        for rid in rids:
            d = self.recipe(int(rid), full=full)
            if d:
                out.append(d)
        return out

    def canon_ingredients(self, rid: int) -> list[dict]:
        rows = self.q("SELECT pos, raw, canon, qty, unit FROM recipe_ingredients WHERE rid = ? ORDER BY pos", (rid,))
        return [dict(r) for r in rows]

    def flags(self, rid: int) -> list[dict]:
        return [dict(r) for r in self.q("SELECT cat, level, canon FROM recipe_cat WHERE rid = ?", (rid,))]

    # ------------------------------------------------------------------ search
    def _fts_query(self, text: str) -> str | None:
        words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS and len(w) > 2]
        words = _dedupe(words)[:12]
        return " OR ".join(f'"{w}"' for w in words) if words else None

    def _ensure_emb(self):
        if self._emb is None:
            ep, rp = STORAGE / "emb.npy", STORAGE / "emb_rids.npy"
            if ep.exists() and rp.exists():
                self._emb = np.load(ep).astype(np.float32)
                self._emb_rids = np.load(rp)

    @property
    def has_embeddings(self) -> bool:
        self._ensure_emb()
        return self._emb is not None

    def _embed_query(self, text: str) -> np.ndarray | None:
        self._ensure_emb()
        if self._emb is None:
            return None
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        v = np.array(list(self._embedder.embed([text])), dtype=np.float32)[0]
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, text: str, f: Filters | None = None, k: int = 10, pool: int = 300) -> dict:
        """Hybrid BM25 + embedding search restricted to the *complete* filtered set.

        Returns {"results": [...], "filtered_total": int, "method": str}. Fusion = reciprocal rank (k=60).
        """
        f = f or Filters()
        allowed = None if f.is_empty() else set(self.filtered_rids(f))
        total = len(allowed) if allowed is not None else int(self.meta().get("complete", 0))
        scores: dict[int, float] = {}
        methods = []
        fq = self._fts_query(text) if text else None
        if fq:
            rows = self.q("SELECT rowid FROM recipes_fts WHERE recipes_fts MATCH ? ORDER BY bm25(recipes_fts, 6.0, 1.0) LIMIT ?",
                          (fq, 3000 if allowed is not None else pool))
            rank = 0
            for (rid,) in rows:
                if allowed is not None and rid not in allowed:
                    continue
                rank += 1
                scores[rid] = scores.get(rid, 0) + 1.0 / (60 + rank)
                if rank >= pool:
                    break
            methods.append("BM25")
        qv = self._embed_query(text) if text else None
        if qv is not None:
            sims = self._emb @ qv
            if allowed is not None:
                mask = np.isin(self._emb_rids, np.fromiter(allowed, dtype=np.int32, count=len(allowed)))
                sims = np.where(mask, sims, -9.0)
            top = np.argpartition(-sims, min(pool, len(sims) - 1))[:pool]
            top = top[np.argsort(-sims[top])]
            rank = 0
            for idx in top:
                if sims[idx] <= -5:
                    break
                rank += 1
                rid = int(self._emb_rids[idx])
                scores[rid] = scores.get(rid, 0) + 1.0 / (60 + rank)
            methods.append("embeddings")
        if not scores and allowed is not None and not text:
            # filters only: deterministic order
            for rid in sorted(allowed)[:k]:
                scores[rid] = 1.0
            methods.append("filter-only")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        results, seen = [], set()
        for rid, sc in ranked:
            d = self.recipe(rid, full=False)
            if not d or not d["complete"]:
                continue
            prim = self.q("SELECT dup_of FROM recipes WHERE rid = ?", (rid,))[0][0] or rid
            if prim in seen:
                continue
            seen.add(prim)
            if prim != rid:
                d = self.recipe(prim, full=False) or d
            dups = [r[0] for r in self.q("SELECT id FROM recipes WHERE dup_of = ? AND rid != ?", (prim, prim))]
            if dups:
                d["duplicate_records"] = dups
            d["score"] = round(sc, 5)
            results.append(d)
            if len(results) >= k:
                break
        return {"results": results, "filtered_total": total, "method": "+".join(methods) or "none"}

    # ------------------------------------------------------------------ evidence
    def constraint_evidence(self, rid: int, f: Filters) -> dict:
        """Evidence lines proving a recipe satisfies the filters: matched ingredient lines for includes,
        and 'possible' allergen warnings that were NOT used for exclusion."""
        ings = self.canon_ingredients(rid)
        ev: dict = {"matched": {}, "warnings": []}
        for term in f.include:
            forms = {c for c, _ in self.resolve_term(term, "strict")}
            ev["matched"][term] = [i["raw"] for i in ings if i["canon"] in forms][:3]
        wanted = []
        for c in f.exclude_categories:
            wanted += resolve_category(c)
        for term in f.exclude:
            wanted += resolve_category(term)
        for fl in self.flags(rid):
            if fl["level"] == "possible" and fl["cat"] in wanted:
                ev["warnings"].append(f"possible {fl['cat']} in '{fl['canon']}' (check the package label)")
        return ev

    def variant_note(self, query: str, ing: str) -> str:
        return variant_note(query, ing)


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
