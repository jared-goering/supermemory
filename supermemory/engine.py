"""
OpenClaw Memory Engine — Phase 1
Local-first structured memory with temporal versioning, relations, and hybrid search.
"""

import json
import sqlite3
import uuid
from datetime import datetime

import litellm
import numpy as np

# Lazy import — only loaded when using local embeddings
SentenceTransformer = None


def _get_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST

        SentenceTransformer = ST
    return SentenceTransformer


from supermemory.config import get_config

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

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    PRIMARY KEY (memory_id, entity_name)
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_current ON memories(is_current);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(source_session);
CREATE INDEX IF NOT EXISTS idx_relations_from ON memory_relations(from_memory);
CREATE INDEX IF NOT EXISTS idx_relations_to ON memory_relations(to_memory);
CREATE INDEX IF NOT EXISTS idx_profiles_entity ON profiles(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical);
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
        self._embedding_provider = cfg.get("embedding_provider", "local")
        self._dedup_threshold = cfg["dedup_threshold"]
        self._embedder = None
        self._init_db()

    @property
    def embedder(self):
        """Lazy-load local SentenceTransformer model (only used when embedding_provider='local')."""
        if self._embedder is None:
            ST = _get_sentence_transformer()
            self._embedder = ST(self._embedding_model)
        return self._embedder

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Embedding helpers ────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        if self._embedding_provider == "local":
            return self.embedder.encode(texts, normalize_embeddings=True).astype(np.float32)

        # API-based embeddings via litellm (supports openai, cohere, voyage, etc.)
        response = litellm.embedding(
            model=self._embedding_model,
            input=texts,
        )
        vecs = np.array(
            [item["embedding"] for item in response.data],
            dtype=np.float32,
        )
        # Normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vecs / norms).astype(np.float32)

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

    # ── Entity helpers ─────────────────────────────────────────────────

    def _resolve_entity(self, conn: sqlite3.Connection, name: str) -> str:
        """Resolve an entity name through aliases to its canonical form."""
        row = conn.execute(
            "SELECT canonical FROM entity_aliases WHERE alias = ?",
            (name.lower().strip(),),
        ).fetchone()
        return row["canonical"] if row else name.strip()

    def _store_entities(self, conn: sqlite3.Connection, memory_id: str, entities: list[dict | str]):
        """Store entity-memory links in the join table."""
        for entity in entities:
            if isinstance(entity, dict):
                name = entity.get("name", "")
                etype = entity.get("type")
            else:
                name = str(entity)
                etype = None

            if not name:
                continue

            canonical = self._resolve_entity(conn, name)
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_name, entity_type) "
                "VALUES (?, ?, ?)",
                (memory_id, canonical, etype),
            )

    def add_entity_alias(self, alias: str, canonical: str):
        """Register an alias that maps to a canonical entity name."""
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias, canonical) VALUES (?, ?)",
                (alias.lower().strip(), canonical.strip()),
            )
            conn.commit()
        finally:
            conn.close()

    def merge_entities(self, old_name: str, new_name: str):
        """Merge old_name into new_name: update join table + add alias."""
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Update all memory_entities rows
            conn.execute(
                "UPDATE memory_entities SET entity_name = ? WHERE entity_name = ?",
                (new_name.strip(), old_name.strip()),
            )
            # Add alias mapping
            conn.execute(
                "INSERT OR REPLACE INTO entity_aliases (alias, canonical) VALUES (?, ?)",
                (old_name.lower().strip(), new_name.strip()),
            )
            # Update profile if exists
            conn.execute(
                "UPDATE profiles SET entity_name = ? WHERE entity_name = ?",
                (new_name.strip(), old_name.strip()),
            )
            conn.commit()
        finally:
            conn.close()

    def list_entities(self, min_mentions: int = 1) -> list[dict]:
        """List all entities with mention counts."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT me.entity_name, me.entity_type, COUNT(*) as mention_count,
                          (SELECT COUNT(*) FROM memory_entities me2
                           JOIN memories m ON me2.memory_id = m.id
                           WHERE me2.entity_name = me.entity_name AND m.is_current = 1
                          ) as current_mentions
                   FROM memory_entities me
                   GROUP BY me.entity_name
                   HAVING COUNT(*) >= ?
                   ORDER BY mention_count DESC""",
                (min_mentions,),
            ).fetchall()
            return [
                {
                    "entity_name": r["entity_name"],
                    "entity_type": r["entity_type"],
                    "mention_count": r["mention_count"],
                    "current_mentions": r["current_mentions"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ── Ingest ───────────────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        session_key: str,
        agent_id: str,
        document_date: str | None = None,
    ) -> list[dict]:
        """
        Extract atomic memories from text, detect relations, embed and store.

        Transaction strategy: LLM calls happen OUTSIDE db transactions.
        DB writes use short, focused transactions to avoid lock contention.
        """
        if document_date is None:
            document_date = datetime.now().isoformat()[:10]

        # ── Phase 1: LLM extraction (no DB lock) ────────────────────────
        extract_response = self._llm_call(EXTRACT_PROMPT.format(text=text))
        try:
            extracted = self._parse_json(extract_response)
        except (json.JSONDecodeError, ValueError) as err:
            raise ValueError(f"LLM extraction returned invalid JSON: {extract_response}") from err

        if not extracted:
            return []

        # Embed all new memories in one batch (no DB lock)
        contents = [m["content"] for m in extracted]
        embeddings = self._embed_batch(contents)

        # ── Phase 2: Short write transaction for inserts ─────────────────
        created_memories = []
        conn = self._conn()
        try:
            # Load existing embeddings for dedup (read-only, WAL allows concurrent reads)
            existing_rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE is_current = 1 AND embedding IS NOT NULL"
            ).fetchall()

            existing_matrix = None
            if existing_rows:
                existing_matrix = np.empty(
                    (len(existing_rows), self._embedding_dim), dtype=np.float32
                )
                for ei, er in enumerate(existing_rows):
                    blob = er["embedding"]
                    if blob and len(blob) == self._embedding_dim * 4:
                        existing_matrix[ei] = np.frombuffer(blob, dtype=np.float32)
                    else:
                        existing_matrix[ei] = 0

            # Build insert batch (filter dupes first, then single INSERT batch)
            insert_batch = []
            for i, mem_data in enumerate(extracted):
                # Skip exact content duplicates
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content = ? AND is_current = 1 LIMIT 1",
                    (mem_data["content"],),
                ).fetchone()
                if existing:
                    continue

                embedding = embeddings[i]

                # Skip semantic near-duplicates
                if existing_matrix is not None and len(existing_rows) > 0:
                    sims = existing_matrix @ embedding
                    if float(np.max(sims)) > self._dedup_threshold:
                        continue

                mem_id = str(uuid.uuid4())
                insert_batch.append((mem_id, mem_data, embedding))

            # Single write transaction for all inserts + entity links
            if insert_batch:
                conn.execute("BEGIN IMMEDIATE")
                for mem_id, mem_data, embedding in insert_batch:
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
                    # Store entity-memory links
                    entities = mem_data.get("entities", [])
                    self._store_entities(conn, mem_id, entities)

                    created_memories.append(
                        {
                            "id": mem_id,
                            "content": mem_data["content"],
                            "category": mem_data.get("category"),
                            "confidence": mem_data.get("confidence", 1.0),
                            "entities": entities,
                            "embedding": embedding,
                        }
                    )
                conn.commit()
        finally:
            conn.close()

        if not created_memories:
            return []

        # ── Phase 3: Relation detection via LLM (no DB lock) ────────────
        # Read existing memories for similarity comparison
        conn = self._conn()
        try:
            existing_rows = conn.execute(
                "SELECT id, content, embedding FROM memories WHERE is_current = 1"
            ).fetchall()
        finally:
            conn.close()

        new_ids = {m["id"] for m in created_memories}
        existing = [
            (r["id"], r["content"], self._blob_to_vec(r["embedding"]))
            for r in existing_rows
            if r["id"] not in new_ids and r["embedding"] is not None
        ]

        relations = []
        if existing:
            all_relation_items = []
            for mem in created_memories:
                similarities = [
                    (eid, econtent, self._cosine_similarity(mem["embedding"], evec))
                    for eid, econtent, evec in existing
                ]
                similarities.sort(key=lambda x: x[2], reverse=True)
                top_5 = similarities[:5]

                if top_5 and top_5[0][2] > 0.3:
                    all_relation_items.append((mem, top_5))

            if all_relation_items:
                relation_blocks = []
                for mem, top_5 in all_relation_items:
                    existing_desc = "\n".join(
                        f'  - id: {eid}, content: "{econtent}" (similarity: {sim:.2f})'
                        for eid, econtent, sim in top_5
                    )
                    relation_blocks.append(
                        f'NEW MEMORY (id: {mem["id"]}):\n"{mem["content"]}"\n\n'
                        f"CANDIDATE EXISTING MEMORIES:\n{existing_desc}"
                    )

                combined_prompt = (
                    "For each new memory below, determine if it has relationships "
                    "to the candidate existing memories.\n\n"
                    + "\n---\n".join(relation_blocks)
                    + '\n\nReturn a JSON array of objects, each with: "new_id" (the new memory id), '
                    '"existing_id", "relation" (updates/extends/contradicts/supports/derives), "context".\n'
                    "Return ONLY a JSON array (empty [] if no relationships), no other text."
                )

                # LLM call happens outside any transaction
                relate_response = self._llm_call(combined_prompt)
                try:
                    relations = self._parse_json(relate_response)
                except (json.JSONDecodeError, ValueError):
                    relations = []

        # ── Phase 4: Short write transaction for relations ───────────────
        if relations:
            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
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

                    if relation_type == "updates":
                        old_row = conn.execute(
                            "SELECT version FROM memories WHERE id = ?",
                            (existing_id,),
                        ).fetchone()
                        old_version = old_row["version"] if old_row else 1

                        conn.execute(
                            "UPDATE memories SET is_current = 0, superseded_by = ?, "
                            "updated_at = datetime('now') WHERE id = ?",
                            (new_id, existing_id),
                        )
                        conn.execute(
                            "UPDATE memories SET version = ?, "
                            "updated_at = datetime('now') WHERE id = ?",
                            (old_version + 1, new_id),
                        )
                conn.commit()
            finally:
                conn.close()

        # ── Phase 5: Profile updates via LLM (no DB lock during LLM) ────
        all_entities = set()
        for mem in created_memories:
            for entity in mem.get("entities", []):
                all_entities.add(entity)

        if all_entities:
            for entity_name in all_entities:
                self._update_profile_safe(entity_name)

        return [{k: v for k, v in m.items() if k != "embedding"} for m in created_memories]

    def _update_profile_safe(self, entity_name: str):
        """Update profile with LLM call outside the DB transaction."""
        # Phase A: Read memories via indexed join (no LIKE scan)
        conn = self._conn()
        try:
            canonical = self._resolve_entity(conn, entity_name)
            rows = conn.execute(
                """SELECT m.content, m.category, m.confidence, m.document_date
                   FROM memories m
                   JOIN memory_entities me ON m.id = me.memory_id
                   WHERE me.entity_name = ? AND m.is_current = 1""",
                (canonical,),
            ).fetchall()
            # Fallback to LIKE scan if join table not yet populated for this entity
            if not rows:
                rows = conn.execute(
                    "SELECT content, category, confidence, document_date "
                    "FROM memories WHERE is_current = 1 AND content LIKE ?",
                    (f"%{canonical}%",),
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            return

        memories_text = "\n".join(
            f"- [{r['category']}] {r['content']} "
            f"(date: {r['document_date']}, confidence: {r['confidence']})"
            for r in rows
        )

        # Phase B: LLM call (no DB lock)
        profile_response = self._llm_call(
            PROFILE_PROMPT.format(entity_name=entity_name, memories=memories_text)
        )
        try:
            profile_data = self._parse_json(profile_response)
        except (json.JSONDecodeError, ValueError):
            return

        static = json.dumps(profile_data.get("static_profile", {}))
        dynamic = json.dumps(profile_data.get("dynamic_profile", {}))

        # Phase C: Short write transaction
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT id FROM profiles WHERE entity_name = ?", (entity_name,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE profiles SET static_profile = ?, dynamic_profile = ?, "
                    "updated_at = datetime('now') WHERE entity_name = ?",
                    (static, dynamic, entity_name),
                )
            else:
                conn.execute(
                    "INSERT INTO profiles (id, entity_name, static_profile, dynamic_profile) "
                    "VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), entity_name, static, dynamic),
                )
            conn.commit()
        finally:
            conn.close()

    # _update_profile removed — replaced by _update_profile_safe above

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        current_only: bool = True,
        as_of_date: str | None = None,
    ) -> list[dict]:
        """
        Semantic search: embed query, rank by cosine similarity, hydrate top-k only.

        Optimized: Phase 1 loads only id+embedding for ranking (minimal I/O).
        Phase 2 hydrates full metadata for just the top-k results.
        """
        query_vec = self._embed(query)

        conn = self._conn()
        try:
            # ── Phase 1: Load only id + embedding for ranking ────────────
            if as_of_date:
                # For as_of_date, load candidates and filter in Python
                id_rows = conn.execute(
                    """SELECT id, embedding, is_current, superseded_by, document_date
                       FROM memories
                       WHERE document_date <= ? AND embedding IS NOT NULL""",
                    (as_of_date,),
                ).fetchall()

                # Filter: keep memories that were current as of that date
                filtered = []
                for r in id_rows:
                    if r["is_current"]:
                        filtered.append(r)
                    elif r["superseded_by"]:
                        sup = conn.execute(
                            "SELECT document_date FROM memories WHERE id = ?",
                            (r["superseded_by"],),
                        ).fetchone()
                        if sup and sup["document_date"] > as_of_date:
                            filtered.append(r)
                    else:
                        filtered.append(r)
                id_rows = filtered
            elif current_only:
                id_rows = conn.execute(
                    "SELECT id, embedding FROM memories "
                    "WHERE is_current = 1 AND embedding IS NOT NULL"
                ).fetchall()
            else:
                id_rows = conn.execute(
                    "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
                ).fetchall()

            if not id_rows:
                return []

            # Build embedding matrix and rank
            embed_dim = len(query_vec)
            byte_len = embed_dim * 4
            ids = []
            matrix = np.empty((len(id_rows), embed_dim), dtype=np.float32)
            valid_count = 0

            for r in id_rows:
                blob = r["embedding"]
                if blob and len(blob) == byte_len:
                    matrix[valid_count] = np.frombuffer(blob, dtype=np.float32)
                    ids.append(r["id"])
                    valid_count += 1

            if valid_count == 0:
                return []

            matrix = matrix[:valid_count]
            similarities = matrix @ query_vec

            # Use argpartition for O(n) top-k instead of O(n log n) full sort
            if valid_count > top_k:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1]

            top_ids = [ids[i] for i in top_indices]
            top_sims = {ids[i]: float(similarities[i]) for i in top_indices}

            # ── Phase 2: Hydrate full metadata for top-k only ────────────
            placeholders = ",".join("?" for _ in top_ids)
            full_rows = conn.execute(
                f"""SELECT id, content, category, confidence, document_date, event_date,
                          source_session, source_agent, source_chunk, version,
                          is_current, superseded_by
                   FROM memories WHERE id IN ({placeholders})""",
                top_ids,
            ).fetchall()

            # Preserve ranking order
            row_map = {r["id"]: r for r in full_rows}
            results = []
            for mid in top_ids:
                r = row_map.get(mid)
                if not r:
                    continue
                results.append(
                    {
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
                        "similarity": top_sims[mid],
                    }
                )

            # ── Phase 3: Expand relations for results ────────────────────
            for result in results:
                rels = conn.execute(
                    """SELECT mr.relation, mr.context, m.content as related_content,
                              m.id as related_id
                       FROM memory_relations mr
                       JOIN memories m ON (
                           CASE WHEN mr.from_memory = ? THEN mr.to_memory
                                ELSE mr.from_memory END
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
                    for rel in rels
                ]

            return results
        finally:
            conn.close()

    # ── History ──────────────────────────────────────────────────────────

    def get_history(self, entity_name: str) -> list[dict]:
        """Return version chain for all memories mentioning an entity."""
        with self._conn() as conn:
            canonical = self._resolve_entity(conn, entity_name)
            # Use indexed join table first
            rows = conn.execute(
                """SELECT m.id, m.content, m.category, m.confidence, m.document_date,
                          m.event_date, m.version, m.is_current, m.superseded_by, m.created_at
                   FROM memories m
                   JOIN memory_entities me ON m.id = me.memory_id
                   WHERE me.entity_name = ?
                   ORDER BY m.document_date ASC, m.created_at ASC""",
                (canonical,),
            ).fetchall()
            # Fallback to LIKE if join table not populated
            if not rows:
                rows = conn.execute(
                    """SELECT id, content, category, confidence, document_date, event_date,
                              version, is_current, superseded_by, created_at
                       FROM memories
                       WHERE content LIKE ?
                       ORDER BY document_date ASC, created_at ASC""",
                    (f"%{canonical}%",),
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

    def get_profile(self, entity_name: str) -> dict | None:
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
                "static_profile": json.loads(row["static_profile"])
                if row["static_profile"]
                else {},
                "dynamic_profile": json.loads(row["dynamic_profile"])
                if row["dynamic_profile"]
                else {},
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
            current = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE is_current = 1"
            ).fetchone()["c"]
            superseded = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE is_current = 0"
            ).fetchone()["c"]
            relations = conn.execute("SELECT COUNT(*) as c FROM memory_relations").fetchone()["c"]
            profiles = conn.execute("SELECT COUNT(*) as c FROM profiles").fetchone()["c"]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT source_session) as c FROM memories"
            ).fetchone()["c"]

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
