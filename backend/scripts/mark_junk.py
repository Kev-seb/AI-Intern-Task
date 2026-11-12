"""Flag obvious junk/test records (titles like 'Tst', 'Tstcgdgdgdgdgdgdgd') as not usable."""
import re
import sqlite3
from pathlib import Path

JUNK = re.compile(r"^\s*(tst|test\b|asdf)|(\w{2,3})\2{3,}", re.I)


def mark_junk(con: sqlite3.Connection) -> int:
    rows = con.execute("SELECT rid, title FROM recipes WHERE complete = 1").fetchall()
    bad = [(r,) for r, t in rows if JUNK.search(t or "") and not (t or "").lower().startswith("aaahh")]
    con.executemany("UPDATE recipes SET complete = 0 WHERE rid = ?", bad)
    con.commit()
    return len(bad)


if __name__ == "__main__":
    con = sqlite3.connect(Path(__file__).resolve().parents[1] / "storage" / "recipes.db")
    n = mark_junk(con)
    cur = con.execute("SELECT COUNT(*) FROM recipes WHERE complete = 1").fetchone()[0]
    con.execute("INSERT OR REPLACE INTO meta VALUES ('complete', ?)", (str(cur),))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('junk_records', ?)", (str(n),))
    con.commit()
    print("junk flagged:", n, "usable now:", cur)
