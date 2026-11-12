"""One-shot data setup: download -> parse/normalise into SQLite -> embeddings.

    python scripts/setup_data.py                      # everything (embeddings take ~25-30 min on CPU)
    python scripts/setup_data.py --skip-embeddings    # keyword (BM25) search only; much faster (~2 min)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()
    run("download_data.py")
    run("build_db.py")
    if args.skip_embeddings:
        print("\nSkipped embeddings: search falls back to BM25 only. Run scripts/build_embeddings.py later to enable hybrid search.")
    else:
        run("build_embeddings.py")
    print("\nDone. Next: copy .env.example to .env, add GEMINI_API_KEY, then `uvicorn app.main:app --port 8000`.")
