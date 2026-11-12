"""Build storage/recipes.db from the three Recipe Box JSON files.

Usage:  python scripts/build_db.py [--data ../data]

Creates
  recipes(rid, id, source, src_key, title, ingredients_json, instructions, picture_link,
          n_ingredients, n_steps, complete)
  recipe_ingredients(rid, pos, raw, canon, qty, unit)
  recipe_cat(rid, cat, level, canon)        -- allergen/diet flags (definite | possible)
  recipes_fts(title, ingr)                   -- FTS5 index (BM25) over title + canonical ingredients
  meta(key, value)
"""
from __future__ import annotations

import argparse
import functools
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.normalize import categories, count_steps, parse_ingredients  # noqa: E402
from aux_tables import build_aux_tables  # noqa: E402
from mark_junk import mark_junk  # noqa: E402

SOURCES = {
    "ar": "recipes_raw_nosource_ar.json",    # AllRecipes
    "epi": "recipes_raw_nosource_epi.json",  # Epicurious
    "fn": "recipes_raw_nosource_fn.json",    # Food Network
}

SCHEMA = """
DROP TABLE IF EXISTS recipes;
DROP TABLE IF EXISTS recipe_ingredients;
DROP TABLE IF EXISTS recipe_cat;
DROP TABLE IF EXISTS recipes_fts;
DROP TABLE IF EXISTS meta;
CREATE TABLE recipes(
  rid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, source TEXT NOT NULL, src_key TEXT NOT NULL,
  title TEXT, ingredients_json TEXT, instructions TEXT, picture_link TEXT,
  n_ingredients INTEGER NOT NULL, n_steps INTEGER NOT NULL, complete INTEGER NOT NULL, dup_of INTEGER);
CREATE TABLE recipe_ingredients(rid INTEGER NOT NULL, pos INTEGER NOT NULL, raw TEXT, canon TEXT NOT NULL,
  qty REAL, unit TEXT);
CREATE TABLE recipe_cat(rid INTEGER NOT NULL, cat TEXT NOT NULL, level TEXT NOT NULL, canon TEXT NOT NULL);
CREATE VIRTUAL TABLE recipes_fts USING fts5(title, ingr, tokenize='porter unicode61');
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""
INDEXES = """
CREATE INDEX ix_ri_canon ON recipe_ingredients(canon, rid);
CREATE INDEX ix_ri_rid ON recipe_ingredients(rid);
CREATE INDEX ix_cat ON recipe_cat(cat, level, rid);
CREATE INDEX ix_cat_rid ON recipe_cat(rid);
CREATE INDEX ix_recipes_complete ON recipes(complete, source);
"""


def mark_duplicates(con: sqlite3.Connection) -> int:
    """dup_of = rid of the first recipe sharing the exact same title + raw ingredient list (complete recipes only)."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(recipes)")}
    if "dup_of" not in cols:
        con.execute("ALTER TABLE recipes ADD COLUMN dup_of INTEGER")
    first: dict[tuple[str, str], int] = {}
    updates = []
    for rid, title, ij in con.execute("SELECT rid, title, ingredients_json FROM recipes WHERE complete = 1 ORDER BY rid"):
        key = (title or "", ij or "")
        first.setdefault(key, rid)
        updates.append((first[key], rid))
    con.executemany("UPDATE recipes SET dup_of = ? WHERE rid = ?", updates)
    con.execute("CREATE INDEX IF NOT EXISTS ix_dup ON recipes(dup_of)")
    con.commit()
    return sum(1 for a, b in updates if a != b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data"))
    ap.add_argument("--out", default=str(ROOT / "storage" / "recipes.db"))
    args = ap.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cat_cache = functools.lru_cache(maxsize=None)(categories)
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    t0 = time.time()
    rid = 0
    stats = {"raw_records": 0, "complete": 0}
    per_source: dict[str, dict[str, int]] = {}

    for src, fname in SOURCES.items():
        path = data_dir / fname
        if not path.exists():
            sys.exit(f"missing {path} - run scripts/download_data.py first")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        ps = per_source[src] = {"raw": len(raw), "complete": 0}
        recipes, ings, cats, fts = [], [], [], []
        for key, rec in raw.items():
            rid += 1
            stats["raw_records"] += 1
            title = (rec.get("title") or "").strip()
            ingredient_lines = [x for x in (rec.get("ingredients") or []) if isinstance(x, str)]
            instructions = rec.get("instructions") or ""
            parsed = []
            for line in ingredient_lines:
                parsed.extend(parse_ingredients(line))
            n_steps = count_steps(instructions)
            complete = int(bool(title) and bool(parsed) and n_steps > 0)
            ps["complete"] += complete
            stats["complete"] += complete
            recipes.append((rid, f"{src}:{key}", src, key, title or None,
                            json.dumps(ingredient_lines, ensure_ascii=False), instructions or None,
                            rec.get("picture_link"), len(parsed), n_steps, complete))
            seen_cat: set[tuple[str, str, str]] = set()
            for pos, ing in enumerate(parsed):
                ings.append((rid, pos, ing.raw, ing.canon, ing.qty, ing.unit))
                for name in (ing.canon, *ing.alts):
                    definite, possible = cat_cache(name)
                    for c in definite:
                        seen_cat.add((c, "definite", name))
                    for c in possible:
                        seen_cat.add((c, "possible", name))
            # an alternative like "butter or margarine" is flagged via the primary name only when
            # every option trips the category; keep primary-only flags to avoid over-excluding.
            for c, lvl, name in seen_cat:
                cats.append((rid, c, lvl, name))
            if complete:
                fts.append((rid, title, " ; ".join(i.canon for i in parsed)))
        con.executemany("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)", recipes)
        con.executemany("INSERT INTO recipe_ingredients VALUES (?,?,?,?,?,?)", ings)
        con.executemany("INSERT INTO recipe_cat VALUES (?,?,?,?)", cats)
        con.executemany("INSERT INTO recipes_fts(rowid,title,ingr) VALUES (?,?,?)", fts)
        con.commit()
        print(f"{src}: {ps} ({time.time() - t0:.0f}s)")

    con.executescript(INDEXES)
    n_junk = mark_junk(con)
    stats["complete"] -= n_junk
    n_dups = mark_duplicates(con)
    print('aux tables (subst_sentences, or_lines):', build_aux_tables(con))
    meta = {"junk_records": n_junk, "exact_duplicate_records": n_dups, "raw_records": stats["raw_records"], "complete": stats["complete"],
            "per_source": json.dumps(per_source), "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    con.commit()
    con.close()
    print(f"done: {stats} in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
