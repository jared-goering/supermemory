"""
Migrate source_chunk data from denormalized memories.source_chunk
to normalized source_chunks table.

Groups memories by (source_chunk content hash, source_session) to create
one source_chunks row per unique chunk, then sets source_chunk_id FK.

Usage:
    python -m supermemory.migrate_chunks [--dry-run]
"""

import hashlib
import sqlite3
import sys
import uuid


def migrate(db_path: str, dry_run: bool = False):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    # Ensure new tables/columns exist
    conn.execute(
        """CREATE TABLE IF NOT EXISTS source_chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            session_key TEXT,
            agent_id TEXT,
            document_date TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )"""
    )

    # Add source_chunk_id column if missing
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "source_chunk_id" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN source_chunk_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_chunk ON memories(source_chunk_id)")
        conn.commit()

    # Check if old source_chunk column exists
    if "source_chunk" not in cols:
        print("No source_chunk column found. Nothing to migrate.")
        conn.close()
        return

    # Count memories with source_chunk but no source_chunk_id
    to_migrate = conn.execute(
        "SELECT COUNT(*) as c FROM memories "
        "WHERE source_chunk IS NOT NULL AND source_chunk != '' "
        "AND (source_chunk_id IS NULL OR source_chunk_id = '')"
    ).fetchone()["c"]

    already_migrated = conn.execute(
        "SELECT COUNT(*) as c FROM memories WHERE source_chunk_id IS NOT NULL AND source_chunk_id != ''"
    ).fetchone()["c"]

    existing_chunks = conn.execute("SELECT COUNT(*) as c FROM source_chunks").fetchone()["c"]

    print(f"Memories to migrate: {to_migrate}")
    print(f"Already migrated: {already_migrated}")
    print(f"Existing source_chunks: {existing_chunks}")

    if to_migrate == 0:
        print("Nothing to migrate.")
        conn.close()
        return

    # Group by unique (content_hash, session, agent, date)
    rows = conn.execute(
        """SELECT id, source_chunk, source_session, source_agent, document_date
           FROM memories
           WHERE source_chunk IS NOT NULL AND source_chunk != ''
           AND (source_chunk_id IS NULL OR source_chunk_id = '')
           ORDER BY created_at"""
    ).fetchall()

    # Build chunk groups by content hash
    chunk_map: dict[str, str] = {}  # content_hash -> chunk_id
    updates: list[tuple[str, str]] = []  # (chunk_id, memory_id)
    chunks_created = 0

    for r in rows:
        content = r["source_chunk"]
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        key = f"{content_hash}:{r['source_session'] or ''}"

        if key not in chunk_map:
            chunk_id = str(uuid.uuid4())
            chunk_map[key] = chunk_id
            if not dry_run:
                conn.execute(
                    "INSERT OR IGNORE INTO source_chunks "
                    "(id, content, session_key, agent_id, document_date) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, content, r["source_session"], r["source_agent"], r["document_date"]),
                )
            chunks_created += 1

        updates.append((chunk_map[key], r["id"]))

    if not dry_run:
        # Commit the chunk inserts first
        conn.commit()
        # Batch update in chunks of 500
        for i in range(0, len(updates), 500):
            batch = updates[i : i + 500]
            for chunk_id, mem_id in batch:
                conn.execute(
                    "UPDATE memories SET source_chunk_id = ? WHERE id = ?",
                    (chunk_id, mem_id),
                )
            conn.commit()
            print(f"  Updated {min(i + 500, len(updates))}/{len(updates)} memories")

    print(f"\nDone. {chunks_created} unique chunks {'would be ' if dry_run else ''}created.")
    print(f"{len(updates)} memories {'would be ' if dry_run else ''}linked.")

    # Show storage savings estimate
    if not dry_run and chunks_created > 0:
        old_size = conn.execute(
            "SELECT SUM(LENGTH(source_chunk)) as s FROM memories WHERE source_chunk IS NOT NULL"
        ).fetchone()["s"]
        new_size = conn.execute("SELECT SUM(LENGTH(content)) as s FROM source_chunks").fetchone()[
            "s"
        ]
        if old_size and new_size:
            saved = old_size - new_size
            print(
                f"\nStorage: {old_size / 1024 / 1024:.1f}MB denormalized -> "
                f"{new_size / 1024 / 1024:.1f}MB normalized "
                f"({saved / 1024 / 1024:.1f}MB saved, {saved / old_size * 100:.0f}% reduction)"
            )

    conn.close()


if __name__ == "__main__":
    from supermemory.config import get_config

    cfg = get_config()
    _dry_run = "--dry-run" in sys.argv
    _db = cfg["db_path"]
    # Allow explicit DB path
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            _db = sys.argv[i + 1]
    print(f"Database: {_db}")
    migrate(_db, dry_run=_dry_run)
