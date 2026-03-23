"""
Deduplicate memory database - both exact content and semantic deduplication.

Combines two deduplication strategies:
1. Exact content dedup: keep the oldest memory (first observed), delete newer exact copies
2. Semantic dedup: find and merge near-duplicate memories using embedding similarity
"""

import sqlite3
import sys
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set


def exact_content_dedup(db_path: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Remove exact content duplicates.

    Strategy:
    1. Exact content dedup: keep the oldest memory (first observed), delete newer exact copies
    2. Clean up orphaned relations pointing to deleted memories
    3. Remove noise memories (low-value patterns)

    Preserves: relational versioning chains (UPDATE/EXTEND/CONTRADICT),
               superseded memories, source_session diversity

    Returns dict with stats about what was removed.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    result = {
        "exact_duplicates_removed": 0,
        "noise_memories_removed": 0,
        "duplicate_groups": 0,
        "final_stats": {}
    }

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

    result["duplicate_groups"] = len(dupes)
    result["exact_duplicates_removed"] = len(ids_to_delete)
    print(f"  Found {len(dupes)} duplicate groups")
    print(f"  Memories to delete: {len(ids_to_delete)}")

    if dry_run:
        print("\n[DRY RUN] Would delete the above. Run without --dry-run to execute.")
        conn.close()
        return result

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
    if ids_to_delete:
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
            if to_remove:
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
    result["noise_memories_removed"] = noise_deleted
    print(f"  Removed {noise_deleted} noise memories")

    # --- Phase 4: Stats ---
    print("\nPhase 4: Post-dedup stats...")
    stats = conn.execute("SELECT COUNT(*) FROM memories WHERE is_current = 1").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    rels = conn.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0]
    profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]

    result["final_stats"] = {
        "current_memories": stats,
        "total_memories": total,
        "relations": rels,
        "profiles": profiles
    }

    print(f"  Current memories: {stats}")
    print(f"  Total memories (incl superseded): {total}")
    print(f"  Relations: {rels}")
    print(f"  Profiles: {profiles}")

    # Vacuum
    print("\nVacuuming database...")
    conn.execute("VACUUM")
    conn.close()

    print("\nDone! Remember to refresh the API cache: curl -X POST http://localhost:8642/api/cache/refresh")
    return result


def semantic_dedup(db_path: str, threshold: float = 0.95, limit: int = 500, dry_run: bool = False) -> Dict[str, Any]:
    """
    Semantic deduplication: find and merge near-duplicate memories.

    Uses cosine similarity on embeddings to detect memories that say the same thing
    in slightly different words. Keeps the more specific/detailed version.

    Args:
        db_path: Path to SQLite database
        threshold: Cosine similarity threshold for "near-duplicate" (default 0.95)
        limit: Max pairs to process per run (default 500)
        dry_run: If True, don't actually delete anything

    Returns dict with stats about what was removed.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    result = {
        "pairs_found": 0,
        "memories_removed": 0,
        "threshold": threshold,
        "final_stats": {}
    }

    print(f"Loading current memories with embeddings (threshold={threshold})...")

    rows = conn.execute("""
        SELECT id, content, category, confidence, embedding, created_at, source_session
        FROM memories
        WHERE is_current = 1 AND embedding IS NOT NULL
        ORDER BY created_at ASC
    """).fetchall()

    print(f"  {len(rows)} memories loaded")

    if len(rows) < 2:
        print("Not enough memories with embeddings for semantic dedup")
        conn.close()
        return result

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

    if len(valid) < 2:
        print("Not enough valid embeddings for semantic dedup")
        conn.close()
        return result

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
                if sims[ci, vj] >= threshold:
                    duplicate_pairs.append((global_i, global_j, float(sims[ci, vj])))

        if len(duplicate_pairs) >= limit * 2:
            break

    result["pairs_found"] = len(duplicate_pairs)
    print(f"  Found {len(duplicate_pairs)} near-duplicate pairs")

    if not duplicate_pairs:
        print("No near-duplicates found. Database is clean!")
        conn.close()
        return result

    # Sort by similarity (highest first) and limit
    duplicate_pairs.sort(key=lambda x: x[2], reverse=True)
    duplicate_pairs = duplicate_pairs[:limit]

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

    result["memories_removed"] = len(to_delete)
    print(f"\n{len(to_delete)} memories to remove (keeping more detailed version)")
    print("\nSample merges:")
    for entry in merge_log[:20]:
        print(f"  {entry['similarity']:.3f} | KEEP: {entry['kept']}")
        print(f"         | DROP: {entry['removed']}")
        print()

    if dry_run:
        print(f"[DRY RUN] Would delete {len(to_delete)} near-duplicate memories.")
        conn.close()
        return result

    # Delete near-duplicates
    print(f"\nDeleting {len(to_delete)} near-duplicate memories...")
    delete_list = list(to_delete)

    if delete_list:
        # Batch delete relations and memories
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

    result["final_stats"] = {
        "current_memories": current,
        "total_memories": total,
        "relations": rels
    }

    print(f"\nPost-dedup stats:")
    print(f"  Current memories: {current}")
    print(f"  Total: {total}")
    print(f"  Relations: {rels}")

    conn.close()
    print("\nDone! Run: curl -X POST http://localhost:8642/api/cache/refresh")
    return result


def main():
    """Command-line interface for deduplication tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Memory deduplication tools")
    parser.add_argument("db_path", nargs="?", default="memory.db", help="Path to SQLite database")
    parser.add_argument("--mode", choices=["exact", "semantic", "both"], default="exact",
                        help="Deduplication mode")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Similarity threshold for semantic dedup")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max pairs to process for semantic dedup")

    args = parser.parse_args()

    if args.mode in ["exact", "both"]:
        print("=== EXACT CONTENT DEDUPLICATION ===")
        exact_content_dedup(args.db_path, args.dry_run)
        print()

    if args.mode in ["semantic", "both"]:
        print("=== SEMANTIC DEDUPLICATION ===")
        semantic_dedup(args.db_path, args.threshold, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
