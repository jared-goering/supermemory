"""
Backfill memory_entities join table from existing memories.

Uses a fast LLM pass to extract entity names from memory content,
then populates the join table. Processes in batches to stay within
token limits and minimize LLM calls.

Usage:
    python -m supermemory.backfill_entities [--batch-size 50] [--dry-run]
"""

import json
import sqlite3
import sys

import litellm

from supermemory.config import get_config


def backfill(batch_size: int = 50, dry_run: bool = False):
    cfg = get_config()
    db_path = cfg["db_path"]
    model = cfg["model"]

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    # Ensure table exists
    conn.execute(
        """CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id TEXT NOT NULL,
            entity_name TEXT NOT NULL,
            entity_type TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id),
            PRIMARY KEY (memory_id, entity_name)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name)"
    )
    conn.commit()

    # Find memories not yet in the join table
    total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    already = conn.execute("SELECT COUNT(DISTINCT memory_id) as c FROM memory_entities").fetchone()[
        "c"
    ]
    remaining = conn.execute(
        """SELECT m.id, m.content FROM memories m
           WHERE m.id NOT IN (SELECT DISTINCT memory_id FROM memory_entities)
           ORDER BY m.created_at"""
    ).fetchall()

    print(f"Total memories: {total}")
    print(f"Already linked: {already}")
    print(f"To backfill: {len(remaining)}")

    if not remaining:
        print("Nothing to backfill.")
        return

    # Process in batches
    processed = 0
    entities_added = 0

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]

        # Build a single prompt for the batch
        items = []
        for r in batch:
            items.append(f"ID: {r['id']}\nContent: {r['content']}")

        prompt = (
            "Extract entity names (people, organizations, places, products) from each memory below.\n"
            "Return a JSON object mapping memory ID to a list of entity objects.\n"
            'Each entity object has: "name" (string), "type" (person|org|place|product|other).\n'
            "If a memory has no entities, map it to an empty list.\n\n"
            + "\n---\n".join(items)
            + "\n\nReturn ONLY a JSON object, no other text."
        )

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            result = json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Batch {batch_start}-{batch_start + len(batch)}: LLM error: {e}")
            continue

        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")

        batch_entities = 0
        for mem_id, entity_list in result.items():
            if not isinstance(entity_list, list):
                continue
            for entity in entity_list:
                if isinstance(entity, dict):
                    name = entity.get("name", "").strip()
                    etype = entity.get("type")
                else:
                    name = str(entity).strip()
                    etype = None

                if not name:
                    continue

                # Check alias table
                alias_row = conn.execute(
                    "SELECT canonical FROM entity_aliases WHERE alias = ?",
                    (name.lower(),),
                ).fetchone()
                if alias_row:
                    name = alias_row["canonical"]

                if not dry_run:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_entities "
                        "(memory_id, entity_name, entity_type) VALUES (?, ?, ?)",
                        (mem_id, name, etype),
                    )
                batch_entities += 1

        if not dry_run:
            conn.commit()

        processed += len(batch)
        entities_added += batch_entities
        print(
            f"  Batch {batch_start + 1}-{batch_start + len(batch)}: "
            f"{batch_entities} entities extracted ({processed}/{len(remaining)} memories done)"
        )

    conn.close()
    print(f"\nDone. {entities_added} entity links {'would be ' if dry_run else ''}added.")


if __name__ == "__main__":
    _dry_run = "--dry-run" in sys.argv
    _batch_size = 50
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            _batch_size = int(sys.argv[i + 1])
    backfill(batch_size=_batch_size, dry_run=_dry_run)
