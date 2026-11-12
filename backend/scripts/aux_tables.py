"""Precompute small lookup tables used by the substitution engine (so queries never scan 120k instruction texts).

  subst_sentences(rid, sentence) : instruction sentences that talk about replacing/substituting something
  or_lines(rid, raw)             : ingredient lines of the form "X or Y"
"""
import re
import sqlite3
from pathlib import Path

KW = re.compile(r"instead of|substitut|in place of|replac", re.I)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


def build_aux_tables(con: sqlite3.Connection) -> tuple[int, int]:
    con.executescript(
        """DROP TABLE IF EXISTS subst_sentences; DROP TABLE IF EXISTS or_lines;
           CREATE TABLE subst_sentences(rid INTEGER NOT NULL, sentence TEXT NOT NULL);
           CREATE TABLE or_lines(rid INTEGER NOT NULL, raw TEXT NOT NULL);"""
    )
    batch = []
    for rid, text in con.execute("SELECT rid, instructions FROM recipes WHERE complete = 1 AND instructions IS NOT NULL"):
        if not KW.search(text):
            continue
        for sent in SPLIT.split(text):
            if KW.search(sent):
                batch.append((rid, sent.strip()[:400]))
    con.executemany("INSERT INTO subst_sentences VALUES (?,?)", batch)
    n1 = len(batch)
    con.execute(
        "INSERT INTO or_lines SELECT i.rid, i.raw FROM recipe_ingredients i JOIN recipes r ON r.rid = i.rid "
        "WHERE r.complete = 1 AND i.raw LIKE '% or %'"
    )
    n2 = con.execute("SELECT COUNT(*) FROM or_lines").fetchone()[0]
    con.execute("CREATE INDEX ix_subst_rid ON subst_sentences(rid)")
    con.commit()
    return n1, n2


if __name__ == "__main__":
    con = sqlite3.connect(Path(__file__).resolve().parents[1] / "storage" / "recipes.db")
    print("subst_sentences, or_lines:", build_aux_tables(con))
