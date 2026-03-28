#!/usr/bin/env python3
"""Backfill structured_facts for existing source chunks in the eval DB.

Uses concurrent workers to speed up LLM calls.
"""

import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")
os.environ.setdefault("ULTRAMEMORY_DB_PATH", "/tmp/memorybench_eval.db")
os.environ.setdefault("ULTRAMEMORY_EMBEDDING_PROVIDER", "litellm")
os.environ.setdefault("ULTRAMEMORY_EMBEDDING_MODEL", "gemini/gemini-embedding-2-preview")
os.environ.setdefault("ULTRAMEMORY_MODEL", "gemini/gemini-2.5-flash")

from ultramemory.engine import MemoryEngine

WORKERS = 8
DB_PATH = os.environ.get("ULTRAMEMORY_DB_PATH", "/tmp/memorybench_eval.db")

conn = sqlite3.connect(DB_PATH, timeout=30)
conn.row_factory = sqlite3.Row

chunks = conn.execute("""
    SELECT sc.id, sc.content, sc.session_key, sc.document_date
    FROM source_chunks sc
    WHERE sc.id NOT IN (SELECT DISTINCT source_chunk_id FROM structured_facts WHERE source_chunk_id IS NOT NULL)
    ORDER BY sc.created_at
""").fetchall()
conn.close()

# Convert to plain dicts for thread safety
chunks = [dict(c) for c in chunks]
print(f"Chunks to process: {len(chunks)} with {WORKERS} workers")

total_facts = 0
errors = 0
completed = 0
start = time.time()


def process_chunk(chunk):
    """Process a single chunk — each thread gets its own engine instance."""
    engine = MemoryEngine(db_path=DB_PATH)
    try:
        facts = engine.extract_facts(
            text=chunk["content"],
            session_key=chunk["session_key"],
            chunk_id=chunk["id"],
            document_date=chunk["document_date"],
        )
        return len(facts), None
    except Exception as e:
        return 0, str(e)


with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(process_chunk, c): c for c in chunks}

    for future in as_completed(futures):
        chunk = futures[future]
        n_facts, err = future.result()
        completed += 1

        if err:
            errors += 1
            if errors <= 5 or errors % 50 == 0:
                print(f"[{completed}/{len(chunks)}] ERROR ({chunk['session_key'][:40]}): {err}")
        else:
            total_facts += n_facts

        if completed % 25 == 0 or n_facts > 0:
            elapsed = time.time() - start
            rate = completed / elapsed * 60
            eta = (len(chunks) - completed) / rate if rate > 0 else 0
            print(
                f"[{completed}/{len(chunks)}] facts={total_facts} errors={errors} rate={rate:.0f}/min ETA={eta:.0f}min"
            )

elapsed = time.time() - start
print(f"\nDone! {total_facts} facts from {len(chunks)} chunks in {elapsed:.0f}s ({errors} errors)")
