"""
OpenClaw Memory Engine — Phase 1
Local-first structured memory with temporal versioning, relations, and hybrid search.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import litellm

from config import get_config


# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT,
    confidence REAL DEFAULT 1.0,
    document_date TEXT NOT NULL,
    event_date TEXT,
    source_session TEXT,
    source_agent TEXT,
    source_chunk TEXT,
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT 1,
    superseded_by TEXT,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_memory TEXT NOT NULL,
    to_memory TEXT NOT NULL,
    relation TEXT NOT NULL,
    context TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (from_memory) REFERENCES memories(id),
    FOREIGN KEY (to_memory) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    entity_name TEXT NOT NULL,
    static_profile TEXT,
    dynamic_profile TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_current ON memories(is_current);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(source_session);
CREATE INDEX IF NOT EXISTS idx_relations_from ON memory_relations(from_memory);
CREATE INDEX IF NOT EXISTS idx_relations_to ON memory_relations(to_memory);
CREATE INDEX IF NOT EXISTS idx_profiles_entity ON profiles(entity_name);
"""

# ── LLM Prompts ──────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """Extract atomic memories (single facts) from this conversation text.
Return a JSON array of objects, each with:
- "content": one atomic fact as a single sentence
- "category": one of "person", "preference", "project", "decision", "event", "insight"
- "event_date": ISO date string if a specific date is mentioned or inferable, otherwise null
- "confidence": float 0-1 indicating how certain this fact is (1.0 = explicitly stated, 0.7 = implied, 0.5 = uncertain)
- "entities": list of entity names (people, orgs, places) mentioned in this fact

Be thorough — extract every distinct fact. One fact per item. Do NOT combine multiple facts.

Conversation text:
---
{text}
---

Return ONLY a JSON array, no other text."""

RELATE_PROMPT = """Given a NEW memory and a list of EXISTING memories, determine if the new memory has any relationship to the existing ones.

NEW MEMORY:
{new_memory}

EXISTING MEMORIES:
{existing_memories}

For each existing memory that has a relationship to the new one, return a JSON array of objects with:
- "existing_id": the id of the existing memory
- "relation": one of "updates" (new fact replaces old), "extends" (adds detail to existing), "contradicts" (conflicts with existing), "supports" (confirms existing), "derives" (inferred from existing)
- "context": brief explanation of why this relationship exists

If the new memory UPDATES an existing one (same topic, newer info), the old one should be superseded.

Return ONLY a JSON array (empty array [] if no relationships found), no other text."""

PROFILE_PROMPT = """Given these memories about the entity "{entity_name}", build a profile.

MEMORIES:
{memories}

Return a JSON object with:
- "static_profile": object with stable/core facts (name, role, location, etc.)
- "dynamic_profile": object with recent/evolving facts (current projects, recent preferences, etc.)

Return ONLY a JSON object, no other text."""


# ── Engine ───────────────────────────────────────────────────────────────────

class MemoryEngine:
    """Local-first structured memory engine with temporal versioning and relations."""

    def __init__(self, db_path: str | None = None, model_name: str | None = None):
        cfg = get_config()
        self.db_path = db_path or cfg["db_path"]
        self.model_name = model_name or cfg["model"]
        self._embedding_model = cfg["embedding_model"]
        self._embedding_dim = cfg["embedding_dim"]
        self._dedup_threshold = cfg["dedup_threshold"]
        self._embedder = None
        self._init_db()

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executescript(SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Embedding helpers ────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True).astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True).astype(np.float32)

    @staticmethod
    def _blob_to_vec(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def _vec_to_blob(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # ── LLM helper ───────────────────────────────────────────────────────

    def _llm_call(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def _parse_json(self, text: str):
        """Extract JSON from LLM response, handling markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    # ── Ingest ───────────────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        session_key: str,
        agent_id: str,
        document_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Extract atomic memories from text, detect relations, embed and store.
        Returns list of created memory dicts.
        """
        if document_date is None:
            document_date = datetime.now().isoformat()[:10]

        # LLM Call 1: Extract atomic memories
        extract_response = self._llm_call(EXTRACT_PROMPT.format(text=text))
        try:
            extracted = self._parse_json(extract_response)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"LLM extraction returned invalid JSON: {extract_response}")

        if not extracted:
            return []

        # Embed all new memories in one batch
        contents = [m["content"] for m in extracted]
        embeddings = self._embed_batch(contents)

        created_memories = []

        with self._conn() as conn:
            # Load existing embeddings for semantic dedup check
            existing_rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE is_current = 1 AND embedding IS NOT NULL"
            ).fetchall()
            if existing_rows:
                _existing_matrix = np.empty((len(existing_rows), self._embedding_dim), dtype=np.float32)
                for ei, er in enumerate(existing_rows):
                    blob = er["embedding"]
                    if blob and len(blob) == self._embedding_dim * 4:
                        _existing_matrix[ei] = np.frombuffer(blob, dtype=np.float32)
                    else:
                        _existing_matrix[ei] = 0
            else:
                _existing_matrix = None

            # Build list of new memory records
            for i, mem_data in enumerate(extracted):
                # Skip exact content duplicates
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content = ? AND is_current = 1 LIMIT 1",
                    (mem_data["content"],)
                ).fetchone()
                if existing:
                    continue

                mem_id = str(uuid.uuid4())
                embedding = embeddings[i]

                # Skip semantic near-duplicates (>0.97 cosine similarity)
                if _existing_matrix is not None and len(existing_rows) > 0:
                    sims = _existing_matrix @ embedding
                    max_sim = float(np.max(sims))
                    if max_sim > self._dedup_threshold:
                        continue

                conn.execute(
                    """INSERT INTO memories
                       (id, content, category, confidence, document_date, event_date,
                        source_session, source_agent, source_chunk, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        mem_id,
                        mem_data["content"],
                        mem_data.get("category"),
                        mem_data.get("confidence", 1.0),
                        document_date,
                        mem_data.get("event_date"),
                        session_key,
                        agent_id,
                        text,
                        self._vec_to_blob(embedding),
                    ),
                )

                created_memories.append({
                    "id": mem_id,
                    "content": mem_data["content"],
                    "category": mem_data.get("category"),
                    "confidence": mem_data.get("confidence", 1.0),
                    "entities": mem_data.get("entities", []),
                    "embedding": embedding,
                })

            # LLM Call 2: Detect relations to existing memories
            # For each new memory, find top-5 similar existing memories and ask LLM
            existing_rows = conn.execute(
                "SELECT id, content, embedding FROM memories WHERE is_current = 1"
            ).fetchall()

            # Build existing embeddings matrix (exclude just-created memories)
            new_ids = {m["id"] for m in created_memories}
            existing = [
                (r["id"], r["content"], self._blob_to_vec(r["embedding"]))
                for r in existing_rows
                if r["id"] not in new_ids and r["embedding"] is not None
            ]

            if existing:
                # Batch all new memories' relation checks into one LLM call
                all_relation_items = []
                for mem in created_memories:
                    # Find top-5 similar existing memories
                    similarities = [
                        (eid, econtent, self._cosine_similarity(mem["embedding"], evec))
                        for eid, econtent, evec in existing
                    ]
                    similarities.sort(key=lambda x: x[2], reverse=True)
                    top_5 = similarities[:5]

                    if top_5 and top_5[0][2] > 0.3:  # Only check if there's reasonable similarity
                        all_relation_items.append((mem, top_5))

                if all_relation_items:
                    # Build a single prompt for all relation checks
                    relation_blocks = []
                    for mem, top_5 in all_relation_items:
                        existing_desc = "\n".join(
                            f"  - id: {eid}, content: \"{econtent}\" (similarity: {sim:.2f})"
                            for eid, econtent, sim in top_5
                        )
                        relation_blocks.append(
                            f"NEW MEMORY (id: {mem['id']}):\n\"{mem['content']}\"\n\nCANDIDATE EXISTING MEMORIES:\n{existing_desc}"
                        )

                    combined_prompt = (
                        "For each new memory below, determine if it has relationships to the candidate existing memories.\n\n"
                        + "\n---\n".join(relation_blocks)
                        + "\n\nReturn a JSON array of objects, each with: \"new_id\" (the new memory id), \"existing_id\", \"relation\" (updates/extends/contradicts/supports/derives), \"context\".\n"
                        "Return ONLY a JSON array (empty [] if no relationships), no other text."
                    )

                    relate_response = self._llm_call(combined_prompt)
                    try:
                        relations = self._parse_json(relate_response)
                    except (json.JSONDecodeError, ValueError):
                        relations = []

                    # Process relations
                    for rel in relations:
                        new_id = rel.get("new_id")
                        existing_id = rel.get("existing_id")
                        relation_type = rel.get("relation")
                        context = rel.get("context", "")

                        if not new_id or not existing_id or not relation_type:
                            continue

                        conn.execute(
                            """INSERT INTO memory_relations
                               (from_memory, to_memory, relation, context)
                               VALUES (?, ?, ?, ?)""",
                            (new_id, existing_id, relation_type, context),
                        )

                        # If this updates an existing memory, mark old as superseded
                        if relation_type == "updates":
                            # Get the version of the existing memory
                            old_row = conn.execute(
                                "SELECT version FROM memories WHERE id = ?",
                                (existing_id,),
                            ).fetchone()
                            old_version = old_row["version"] if old_row else 1

                            conn.execute(
                                "UPDATE memories SET is_current = 0, superseded_by = ?, updated_at = datetime('now') WHERE id = ?",
                                (new_id, existing_id),
                            )
                            conn.execute(
                                "UPDATE memories SET version = ?, updated_at = datetime('now') WHERE id = ?",
                                (old_version + 1, new_id),
                            )

            # LLM Call 3: Update profiles for mentioned entities
            all_entities = set()
            for mem in created_memories:
                for entity in mem.get("entities", []):
                    all_entities.add(entity)

            if all_entities:
                for entity_name in all_entities:
                    self._update_profile(conn, entity_name)

        # Return created memories (without embedding arrays for cleanliness)
        return [
            {k: v for k, v in m.items() if k != "embedding"}
            for m in created_memories
        ]

    def _update_profile(self, conn: sqlite3.Connection, entity_name: str):
        """Update or create profile for an entity based on current memories."""
        # Find all current memories mentioning this entity
        rows = conn.execute(
            "SELECT content, category, confidence, document_date FROM memories WHERE is_current = 1 AND content LIKE ?",
            (f"%{entity_name}%",),
        ).fetchall()

        if not rows:
            return

        memories_text = "\n".join(
            f"- [{r['category']}] {r['content']} (date: {r['document_date']}, confidence: {r['confidence']})"
            for r in rows
        )

        profile_response = self._llm_call(
            PROFILE_PROMPT.format(entity_name=entity_name, memories=memories_text)
        )

        try:
            profile_data = self._parse_json(profile_response)
        except (json.JSONDecodeError, ValueError):
            return

        static = json.dumps(profile_data.get("static_profile", {}))
        dynamic = json.dumps(profile_data.get("dynamic_profile", {}))

        # Upsert profile
        existing = conn.execute(
            "SELECT id FROM profiles WHERE entity_name = ?", (entity_name,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE profiles SET static_profile = ?, dynamic_profile = ?, updated_at = datetime('now') WHERE entity_name = ?",
                (static, dynamic, entity_name),
            )
        else:
            conn.execute(
                "INSERT INTO profiles (id, entity_name, static_profile, dynamic_profile) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), entity_name, static, dynamic),
            )

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        current_only: bool = True,
        as_of_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search: embed query → cosine similarity → temporal filter → expand relations.
        """
        query_vec = self._embed(query)

        with self._conn() as conn:
            if as_of_date:
                # For as_of_date queries, we need memories that were current at that time.
                # A memory was current at date X if:
                #   - It was created on or before X, AND
                #   - It was not superseded before X (either still current OR superseded after X)
                rows = conn.execute(
                    """SELECT id, content, category, confidence, document_date, event_date,
                              source_session, source_agent, source_chunk, version,
                              is_current, superseded_by, embedding, created_at
                       FROM memories
                       WHERE document_date <= ?
                         AND embedding IS NOT NULL""",
                    (as_of_date,),
                ).fetchall()

                # Filter: keep memories that were current as of that date
                # A memory superseded before as_of_date should be excluded if the superseding
                # memory also existed by then. We approximate by checking if superseded_by exists
                # and that superseding memory's document_date <= as_of_date.
                filtered_rows = []
                for r in rows:
                    if r["is_current"]:
                        filtered_rows.append(r)
                    elif r["superseded_by"]:
                        # Check if the superseding memory was created after as_of_date
                        sup = conn.execute(
                            "SELECT document_date FROM memories WHERE id = ?",
                            (r["superseded_by"],),
                        ).fetchone()
                        if sup and sup["document_date"] > as_of_date:
                            # Superseding memory didn't exist yet — this was still current
                            filtered_rows.append(r)
                    else:
                        filtered_rows.append(r)
                rows = filtered_rows
            elif current_only:
                rows = conn.execute(
                    """SELECT id, content, category, confidence, document_date, event_date,
                              source_session, source_agent, source_chunk, version,
                              is_current, superseded_by, embedding
                       FROM memories
                       WHERE is_current = 1
                         AND embedding IS NOT NULL"""
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, content, category, confidence, document_date, event_date,
                              source_session, source_agent, source_chunk, version,
                              is_current, superseded_by, embedding
                       FROM memories
                       WHERE embedding IS NOT NULL"""
                ).fetchall()

            if not rows:
                return []

            # Batch vectorized similarity: stack all embeddings into matrix, single matmul
            embed_dim = len(query_vec)
            embeddings = np.empty((len(rows), embed_dim), dtype=np.float32)
            valid_mask = []
            for i, r in enumerate(rows):
                blob = r["embedding"]
                if blob and len(blob) == embed_dim * 4:
                    embeddings[i] = np.frombuffer(blob, dtype=np.float32)
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)

            # Single matrix-vector multiply for all similarities at once
            similarities = embeddings @ query_vec  # (N,) dot products

            # Build results with scores
            scored = []
            for i, r in enumerate(rows):
                if not valid_mask[i]:
                    continue
                scored.append((similarities[i], i))

            # Partial sort: only need top_k
            scored.sort(key=lambda x: x[0], reverse=True)
            top_indices = scored[:top_k]

            results = []
            for sim, i in top_indices:
                r = rows[i]
                results.append({
                    "id": r["id"],
                    "content": r["content"],
                    "category": r["category"],
                    "confidence": r["confidence"],
                    "document_date": r["document_date"],
                    "event_date": r["event_date"],
                    "source_session": r["source_session"],
                    "source_chunk": r["source_chunk"],
                    "version": r["version"],
                    "is_current": bool(r["is_current"]),
                    "similarity": float(sim),
                })

            # Expand relations for each result
            for result in results:
                relations = conn.execute(
                    """SELECT mr.relation, mr.context, m.content as related_content, m.id as related_id
                       FROM memory_relations mr
                       JOIN memories m ON (
                           CASE WHEN mr.from_memory = ? THEN mr.to_memory ELSE mr.from_memory END
                       ) = m.id
                       WHERE mr.from_memory = ? OR mr.to_memory = ?""",
                    (result["id"], result["id"], result["id"]),
                ).fetchall()

                result["relations"] = [
                    {
                        "relation": rel["relation"],
                        "context": rel["context"],
                        "related_content": rel["related_content"],
                        "related_id": rel["related_id"],
                    }
                    for rel in relations
                ]

            return results

    # ── History ──────────────────────────────────────────────────────────

    def get_history(self, entity_name: str) -> list[dict]:
        """Return version chain for all memories mentioning an entity."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, content, category, confidence, document_date, event_date,
                          version, is_current, superseded_by, created_at
                   FROM memories
                   WHERE content LIKE ?
                   ORDER BY document_date ASC, created_at ASC""",
                (f"%{entity_name}%",),
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "category": r["category"],
                    "confidence": r["confidence"],
                    "document_date": r["document_date"],
                    "event_date": r["event_date"],
                    "version": r["version"],
                    "is_current": bool(r["is_current"]),
                    "superseded_by": r["superseded_by"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    # ── Profile ──────────────────────────────────────────────────────────

    def get_profile(self, entity_name: str) -> Optional[dict]:
        """Return static + dynamic profile for an entity."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE entity_name = ?",
                (entity_name,),
            ).fetchone()

            if not row:
                return None

            return {
                "id": row["id"],
                "entity_name": row["entity_name"],
                "static_profile": json.loads(row["static_profile"]) if row["static_profile"] else {},
                "dynamic_profile": json.loads(row["dynamic_profile"]) if row["dynamic_profile"] else {},
                "updated_at": row["updated_at"],
            }

    # ── Relations ────────────────────────────────────────────────────────

    def get_relations(self, memory_id: str) -> list[dict]:
        """Return all relations for a memory."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT mr.*,
                          m1.content as from_content,
                          m2.content as to_content
                   FROM memory_relations mr
                   JOIN memories m1 ON mr.from_memory = m1.id
                   JOIN memories m2 ON mr.to_memory = m2.id
                   WHERE mr.from_memory = ? OR mr.to_memory = ?""",
                (memory_id, memory_id),
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "from_memory": r["from_memory"],
                    "to_memory": r["to_memory"],
                    "relation": r["relation"],
                    "context": r["context"],
                    "from_content": r["from_content"],
                    "to_content": r["to_content"],
                }
                for r in rows
            ]

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return counts of memories, relations, entities, profiles."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            current = conn.execute("SELECT COUNT(*) as c FROM memories WHERE is_current = 1").fetchone()["c"]
            superseded = conn.execute("SELECT COUNT(*) as c FROM memories WHERE is_current = 0").fetchone()["c"]
            relations = conn.execute("SELECT COUNT(*) as c FROM memory_relations").fetchone()["c"]
            profiles = conn.execute("SELECT COUNT(*) as c FROM profiles").fetchone()["c"]
            sessions = conn.execute("SELECT COUNT(DISTINCT source_session) as c FROM memories").fetchone()["c"]

            categories = conn.execute(
                "SELECT category, COUNT(*) as c FROM memories WHERE is_current = 1 GROUP BY category"
            ).fetchall()

            return {
                "total_memories": total,
                "current_memories": current,
                "superseded_memories": superseded,
                "relations": relations,
                "profiles": profiles,
                "sessions": sessions,
                "categories": {r["category"]: r["c"] for r in categories},
            }
