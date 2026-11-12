"""Embed every usable recipe (title + canonical ingredients) with a small local model.

Usage:  python scripts/build_embeddings.py

Output: storage/emb.npy (float16, L2-normalised, shape [N, 384]) and storage/emb_rids.npy (int32).
Model:  sentence-transformers/all-MiniLM-L6-v2 via fastembed (ONNX, CPU, no torch, no API quota).
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "storage"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 512


def main() -> None:
    from fastembed import TextEmbedding

    con = sqlite3.connect(STORE / "recipes.db")
    rows = con.execute(
        """SELECT r.rid, r.title, GROUP_CONCAT(DISTINCT i.canon)
           FROM recipes r JOIN recipe_ingredients i ON i.rid = r.rid
           WHERE r.complete = 1 GROUP BY r.rid ORDER BY r.rid"""
    ).fetchall()
    print("recipes to embed:", len(rows))
    texts = []
    for _, title, canons in rows:
        ings = ", ".join((canons or "").split(",")[:14])
        texts.append(f"{title}. Ingredients: {ings}")
    rids = np.array([r[0] for r in rows], dtype=np.int32)

    model = TextEmbedding(MODEL)
    out = np.zeros((len(texts), 384), dtype=np.float16)
    t0 = time.time()
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        vecs = np.array(list(model.embed(chunk, batch_size=128)), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        out[start:start + len(chunk)] = vecs.astype(np.float16)
        done = start + len(chunk)
        if (start // BATCH) % 20 == 0:
            rate = done / (time.time() - t0)
            print(f"{done}/{len(texts)}  {rate:.0f}/s  eta {(len(texts) - done) / rate / 60:.1f} min", flush=True)
    np.save(STORE / "emb.npy", out)
    np.save(STORE / "emb_rids.npy", rids)
    print("saved", out.shape, f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
