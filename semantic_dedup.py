"""
Semantic deduplication: find and merge near-duplicate memories.

Uses cosine similarity on embeddings to detect memories that say the same thing
in slightly different words. Keeps the more specific/detailed version.

Usage:
    python3 semantic_dedup.py [--threshold 0.95] [--dry-run] [--limit 100]
"""

import sqlite3
import sys
import numpy as np
from collections import defaultdict

DB_PATH = "memory.db"
THRESHOLD = 0.95  # cosine similarity threshold for "near-duplicate"
DRY_RUN = "--dry-run" in sys.argv
LIMIT = 500  # max pairs to process per run

# Parse args
for i, arg in enumerate(sys.argv):
    if arg == "--threshold" and i + 1 < len(sys.argv):
        THRESHOLD = float(sys.argv[i + 1])
    if arg == "--limit" and i + 1 < len(sys.argv):
        LIMIT = int(sys.argv[i + 1])
    if arg == "--db" and i + 1 < len(sys.argv):
        DB_PATH = sys.argv[i + 1]

conn = sqlite3.connect(DB_PATH, timeout=30)
conn.execute("PRAGMA journal_mode=WAL")
conn.row_factory = sqlite3.Row

print(f"Loading current memories with embeddings (threshold={THRESHOLD})...")

rows = conn.execute("""
    SELECT id, content, category, confidence, embedding, created_at, source_session
    FROM memories
    WHERE is_current = 1 AND embedding IS NOT NULL
    ORDER BY created_at ASC
""").fetchall()

print(f"  {len(rows)} memories loaded")

# Build embedding matrix
EMBED_DIM = 384
matrix = np.empty((len(rows), EMBED_DIM), dtype=np.float32)
valid = []

for i, r in enumerate(rows):
    blob = r["embedding"]
    if blob and len(blob) == EMBED_DIM * 4:
        matrix[i] = np.frombuffer(blob, dtype=np.float32)
        valid.append(i)
    else:
        matrix[i] = 0

print(f"  {len(valid)} valid embeddings")

# Find near-duplicate pairs using batched matrix multiply
# Process in chunks to avoid memory explosion
print("Computing similarity matrix (chunked)...")

CHUNK_SIZE = 500
duplicate_pairs = []

for start in range(0, len(valid), CHUNK_SIZE):
    chunk_indices = valid[start:start + CHUNK_SIZE]
    chunk_matrix = matrix[chunk_indices]

    # Compute similarity of this chunk against ALL valid memories
    all_valid_matrix = matrix[valid]
    sims = chunk_matrix @ all_valid_matrix.T  # (chunk_size, N)

    for ci, global_i in enumerate(chunk_indices):
        for vj, global_j in enumerate(valid):
            if global_j <= global_i:
                continue  # skip self and already-checked pairs
            if sims[ci, vj] >= THRESHOLD:
                duplicate_pairs.append((global_i, global_j, float(sims[ci, vj])))

    if len(duplicate_pairs) >= LIMIT * 2:
        break

print(f"  Found {len(duplicate_pairs)} near-duplicate pairs")

if not duplicate_pairs:
    print("No near-duplicates found. Database is clean!")
    conn.close()
    sys.exit(0)

# Sort by similarity (highest first) and limit
duplicate_pairs.sort(key=lambda x: x[2], reverse=True)
duplicate_pairs = duplicate_pairs[:LIMIT]

# Decide which to keep: prefer longer content (more specific), lower version (older)
to_delete = set()
merge_log = []

for idx_a, idx_b, sim in duplicate_pairs:
    row_a = rows[idx_a]
    row_b = rows[idx_b]

    id_a, id_b = row_a["id"], row_b["id"]

    # Skip if either already marked for deletion
    if id_a in to_delete or id_b in to_delete:
        continue

    content_a = row_a["content"]
    content_b = row_b["content"]

    # Keep the longer (more detailed) one. If same length, keep older.
    if len(content_a) >= len(content_b):
        keeper, loser = id_a, id_b
        keeper_content, loser_content = content_a, content_b
    else:
        keeper, loser = id_b, id_a
        keeper_content, loser_content = content_b, content_a

    to_delete.add(loser)
    merge_log.append({
        "similarity": sim,
        "kept": keeper_content[:80],
        "removed": loser_content[:80],
    })

print(f"\n{len(to_delete)} memories to remove (keeping more detailed version)")
print("\nSample merges:")
for entry in merge_log[:20]:
    print(f"  {entry['similarity']:.3f} | KEEP: {entry['kept']}")
    print(f"         | DROP: {entry['removed']}")
    print()

if DRY_RUN:
    print(f"[DRY RUN] Would delete {len(to_delete)} near-duplicate memories.")
    conn.close()
    sys.exit(0)

# Delete near-duplicates
print(f"\nDeleting {len(to_delete)} near-duplicate memories...")
delete_list = list(to_delete)

# Re-point relations
for loser_id in delete_list:
    # Find what content this loser has
    loser_content = None
    for r in rows:
        if r["id"] == loser_id:
            loser_content = r["content"]
            break

    # Find keeper (highest similarity match not in delete set)
    # Just delete orphaned relations for simplicity
    pass

# Batch delete relations
batch_size = 100
for i in range(0, len(delete_list), batch_size):
    batch = delete_list[i:i + batch_size]
    placeholders = ",".join("?" for _ in batch)
    conn.execute(
        f"DELETE FROM memory_relations WHERE from_memory IN ({placeholders}) OR to_memory IN ({placeholders})",
        batch + batch
    )
    conn.execute(
        f"DELETE FROM memories WHERE id IN ({placeholders})",
        batch
    )

conn.commit()

# Final stats
current = conn.execute("SELECT COUNT(*) FROM memories WHERE is_current = 1").fetchone()[0]
total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
rels = conn.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0]

print(f"\nPost-dedup stats:")
print(f"  Current memories: {current}")
print(f"  Total: {total}")
print(f"  Relations: {rels}")

conn.close()
print("\nDone! Run: curl -X POST http://localhost:8642/api/cache/refresh")
