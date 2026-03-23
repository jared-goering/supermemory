"""
Deduplicate memory database.

Strategy:
1. Exact content dedup: keep the oldest memory (first observed), delete newer exact copies
2. Clean up orphaned relations pointing to deleted memories
3. Rebuild embedding cache signal to API

Preserves: relational versioning chains (UPDATE/EXTEND/CONTRADICT), 
           superseded memories, source_session diversity
"""

import sqlite3
import sys

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "memory.db"
DRY_RUN = "--dry-run" in sys.argv

conn = sqlite3.connect(DB_PATH, timeout=30)
conn.execute("PRAGMA journal_mode=WAL")
conn.row_factory = sqlite3.Row

# --- Phase 1: Find exact content duplicates among current memories ---
print("Phase 1: Finding exact content duplicates...")

dupes = conn.execute("""
    SELECT content, GROUP_CONCAT(id) as ids, COUNT(*) as cnt,
           MIN(created_at) as first_seen
    FROM memories 
    WHERE is_current = 1
    GROUP BY content 
    HAVING cnt > 1
    ORDER BY cnt DESC
""").fetchall()

ids_to_delete = []
for row in dupes:
    all_ids = row["ids"].split(",")
    
    # Keep the oldest one (first ingested), delete the rest
    # But prefer keeping one that has relations
    ids_with_relations = set()
    for mid in all_ids:
        rel_count = conn.execute(
            "SELECT COUNT(*) FROM memory_relations WHERE from_memory = ? OR to_memory = ?",
            (mid, mid)
        ).fetchone()[0]
        if rel_count > 0:
            ids_with_relations.add(mid)
    
    # Pick keeper: prefer one with relations, else oldest
    if ids_with_relations:
        keeper = sorted(ids_with_relations)[0]  # deterministic pick
    else:
        # Get oldest by created_at
        oldest = conn.execute(
            "SELECT id FROM memories WHERE id IN ({}) ORDER BY created_at ASC LIMIT 1".format(
                ",".join("?" for _ in all_ids)
            ), all_ids
        ).fetchone()
        keeper = oldest["id"]
    
    to_delete = [mid for mid in all_ids if mid != keeper]
    ids_to_delete.extend(to_delete)

print(f"  Found {len(dupes)} duplicate groups")
print(f"  Memories to delete: {len(ids_to_delete)}")

if DRY_RUN:
    print("\n[DRY RUN] Would delete the above. Run without --dry-run to execute.")
    conn.close()
    sys.exit(0)

# --- Phase 2: Delete duplicates ---
print("\nPhase 2: Deleting duplicate memories...")

# First, re-point any relations from deleted memories to their keeper
for row in dupes:
    all_ids = row["ids"].split(",")
    keeper = [mid for mid in all_ids if mid not in ids_to_delete][0]
    deletable = [mid for mid in all_ids if mid in ids_to_delete]
    
    for mid in deletable:
        # Re-point relations to keeper (avoid creating duplicate relations)
        conn.execute("""
            UPDATE memory_relations SET from_memory = ? 
            WHERE from_memory = ? AND NOT EXISTS (
                SELECT 1 FROM memory_relations mr2 
                WHERE mr2.from_memory = ? AND mr2.to_memory = memory_relations.to_memory
                  AND mr2.relation = memory_relations.relation
            )
        """, (keeper, mid, keeper))
        
        conn.execute("""
            UPDATE memory_relations SET to_memory = ?
            WHERE to_memory = ? AND NOT EXISTS (
                SELECT 1 FROM memory_relations mr2
                WHERE mr2.to_memory = ? AND mr2.from_memory = memory_relations.from_memory
                  AND mr2.relation = memory_relations.relation
            )
        """, (keeper, mid, keeper))

# Delete orphaned relations (pointing to memories we're about to delete)
conn.execute("""
    DELETE FROM memory_relations 
    WHERE from_memory IN ({ids}) OR to_memory IN ({ids})
""".format(ids=",".join("?" for _ in ids_to_delete)), ids_to_delete + ids_to_delete)

# Delete the duplicate memories
batch_size = 100
deleted = 0
for i in range(0, len(ids_to_delete), batch_size):
    batch = ids_to_delete[i:i+batch_size]
    conn.execute(
        "DELETE FROM memories WHERE id IN ({})".format(",".join("?" for _ in batch)),
        batch
    )
    deleted += len(batch)

conn.commit()
print(f"  Deleted {deleted} duplicate memories")

# --- Phase 3: Clean up noise memories (low-value patterns) ---
print("\nPhase 3: Removing noise memories...")

NOISE_PATTERNS = [
    "No new volunteers were enriched%",
    "There is nothing to report to the%",
    "Nothing needs to be reported to the%",
    "Changes were detected and committed%",
    "Changes were pushed to the main branch%",
    "A process was killed with SIGTERM%",
    "A retry was attempted%",
    "A second attempt still produced no output%",
    "A script existence and functionality check%",
    "An initial attempt to run a process produced no output%",
]

noise_deleted = 0
for pattern in NOISE_PATTERNS:
    # Keep at most 1 instance of each noise pattern
    noise_ids = conn.execute(
        "SELECT id FROM memories WHERE content LIKE ? AND is_current = 1 ORDER BY created_at ASC",
        (pattern,)
    ).fetchall()
    
    if len(noise_ids) > 1:
        to_remove = [r["id"] for r in noise_ids[1:]]  # keep first, delete rest
        # Clean relations first
        conn.execute(
            "DELETE FROM memory_relations WHERE from_memory IN ({ids}) OR to_memory IN ({ids})".format(
                ids=",".join("?" for _ in to_remove)
            ), to_remove + to_remove
        )
        conn.execute(
            "DELETE FROM memories WHERE id IN ({})".format(",".join("?" for _ in to_remove)),
            to_remove
        )
        noise_deleted += len(to_remove)

conn.commit()
print(f"  Removed {noise_deleted} noise memories")

# --- Phase 4: Stats ---
print("\nPhase 4: Post-dedup stats...")
stats = conn.execute("SELECT COUNT(*) FROM memories WHERE is_current = 1").fetchone()[0]
total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
rels = conn.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0]
profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]

print(f"  Current memories: {stats}")
print(f"  Total memories (incl superseded): {total}")
print(f"  Relations: {rels}")
print(f"  Profiles: {profiles}")

# Vacuum
print("\nVacuuming database...")
conn.execute("VACUUM")
conn.close()

print("\nDone! Remember to refresh the API cache: curl -X POST http://localhost:8642/api/cache/refresh")
