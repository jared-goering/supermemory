"""
OpenClaw Memory Engine — Phase 1
Local-first structured memory with temporal versioning, relations, and hybrid search.
"""

import json
import os
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
        try:
            from sentence_transformers import SentenceTransformer as ST
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install openclaw-ultramemory[local]\n"
                "Or switch to API embeddings: set ULTRAMEMORY_EMBEDDING_PROVIDER=litellm"
            ) from None
        SentenceTransformer = ST
    return SentenceTransformer


from ultramemory.config import get_config

# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    session_key TEXT,
    agent_id TEXT,
    document_date TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT,
    confidence REAL DEFAULT 1.0,
    document_date TEXT NOT NULL,
    event_date TEXT,
    source_session TEXT,
    source_agent TEXT,
    source_chunk_id TEXT,
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT 1,
    superseded_by TEXT,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_chunk_id) REFERENCES source_chunks(id)
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
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(source_agent);
CREATE INDEX IF NOT EXISTS idx_memories_chunk ON memories(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_relations_from ON memory_relations(from_memory);
CREATE INDEX IF NOT EXISTS idx_relations_to ON memory_relations(to_memory);
CREATE INDEX IF NOT EXISTS idx_profiles_entity ON profiles(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical);
CREATE INDEX IF NOT EXISTS idx_source_chunks_session ON source_chunks(session_key);

-- Event extraction + clustering layer
CREATE TABLE IF NOT EXISTS event_mentions (
    id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    source_chunk_id TEXT,
    event_type TEXT NOT NULL,
    subtype TEXT,
    summary TEXT NOT NULL,
    participants TEXT,          -- JSON array
    time_text TEXT,
    normalized_date TEXT,
    duration_minutes REAL,
    user_involvement TEXT,
    confidence REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_chunk_id) REFERENCES source_chunks(id)
);

CREATE TABLE IF NOT EXISTS event_clusters (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    subtype TEXT,
    canonical_label TEXT,
    distinct_key TEXT NOT NULL,
    participants TEXT,          -- JSON array
    normalized_date TEXT,
    duration_minutes REAL,
    user_involvement TEXT,
    confidence REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS event_cluster_members (
    cluster_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    PRIMARY KEY (cluster_id, event_id),
    FOREIGN KEY (cluster_id) REFERENCES event_clusters(id),
    FOREIGN KEY (event_id) REFERENCES event_mentions(id)
);

CREATE TABLE IF NOT EXISTS event_mention_memories (
    event_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (event_id, memory_id),
    FOREIGN KEY (event_id) REFERENCES event_mentions(id),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_event_mentions_type ON event_mentions(event_type, subtype);
CREATE INDEX IF NOT EXISTS idx_event_mentions_session ON event_mentions(session_key);
CREATE INDEX IF NOT EXISTS idx_event_mentions_chunk ON event_mentions(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_event_clusters_type ON event_clusters(event_type, subtype);
CREATE INDEX IF NOT EXISTS idx_event_clusters_distinct ON event_clusters(distinct_key);

-- Structured facts layer for aggregate queries
CREATE TABLE IF NOT EXISTS structured_facts (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    source_chunk_id TEXT,
    session_key TEXT,
    fact_type TEXT NOT NULL,
    category TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value REAL,
    unit TEXT,
    date TEXT,
    confidence REAL DEFAULT 1.0,
    is_user_action BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    FOREIGN KEY (source_chunk_id) REFERENCES source_chunks(id)
);

CREATE INDEX IF NOT EXISTS idx_structured_facts_memory ON structured_facts(memory_id);
CREATE INDEX IF NOT EXISTS idx_structured_facts_type ON structured_facts(fact_type, category);
CREATE INDEX IF NOT EXISTS idx_structured_facts_category ON structured_facts(category);
CREATE INDEX IF NOT EXISTS idx_structured_facts_session ON structured_facts(session_key);
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

EVENT_EXTRACT_PROMPT = """Extract distinct events from this conversation text.
An event is something that happened or will happen — a meeting, trip, exercise session, wedding, dinner, etc.
Do NOT extract general facts, preferences, or background information — only specific episodes.

For each event, return a JSON object with:
- "event_type": broad category (e.g. "wedding", "exercise", "meeting", "art_event", "travel", "meal", "medical")
- "subtype": more specific (e.g. "yoga", "jogging", "gallery_opening", "museum_visit") or null
- "summary": one-sentence description of the event
- "participants": JSON array of participant names (e.g. ["Rachel", "Mike"]), empty array if none mentioned
- "time_text": raw temporal phrase from the text (e.g. "last week", "on Tuesday", "in June") or null
- "normalized_date": ISO date string if a specific date is mentioned or clearly inferable, otherwise null
- "duration_minutes": numeric duration in minutes if mentioned (e.g. 30 for "30 minutes of yoga"), otherwise null
- "user_involvement": one of "attended", "did", "planned", "discussed", "observed" — how the user relates to this event
- "confidence": float 0-1 indicating certainty this is a real episodic event (1.0 = explicitly stated, 0.5 = implied)

Only extract episodic events (things that actually happened at a specific time).
Do NOT extract habitual activities described as routines (e.g. "I usually jog every morning").
If a habitual activity has a specific instance mentioned (e.g. "I jogged for 30 minutes on Tuesday"), extract that instance.

Document date for temporal reference: {document_date}

Conversation text:
---
{text}
---

Return ONLY a JSON array of event objects, no other text. Return empty array [] if no events found."""

FACT_EXTRACT_PROMPT = """Extract quantifiable, structured facts from this conversation text.
A fact is a measurable assertion: a count, duration, cost, attendance, distance, etc.

For each fact, return a JSON object with:
- "fact_type": one of "quantity", "attendance", "duration", "cost", "count", "distance", "frequency"
- "category": broad topic (e.g. "gaming", "wedding", "exercise", "travel", "social", "cooking", "work", "shopping", "health", "education")
- "subject": what or who — the specific thing (e.g. "Assassin's Creed Odyssey", "cousin Rachel's wedding", "marathon training")
- "predicate": the action verb (e.g. "played", "attended", "spent", "ran", "bought", "cooked")
- "value": numeric value (e.g. 70 for 70 hours, 1 for a single attendance, 270 for $270). Use null if no number is stated.
- "unit": measurement unit (e.g. "hours", "count", "dollars", "minutes", "miles", "sessions")
- "date": ISO date string if known, otherwise null
- "confidence": float 0-1 (1.0 = explicitly stated number, 0.7 = clearly implied, 0.4 = rough estimate)
- "is_user_action": true if the USER did/experienced this, false if it's metadata or about someone else

CRITICAL RULES:
- ONLY extract facts where the USER personally did something (is_user_action=true) unless it's clearly about someone else (set is_user_action=false).
- For gaming: "I played Assassin's Creed for 70 hours" → is_user_action=true, value=70, unit="hours"
  But "Assassin's Creed typically takes 60-100 hours to complete" → is_user_action=false (game metadata)
- For weddings: "I attended my cousin Rachel's wedding" → fact_type="attendance", category="wedding", value=1, unit="count", is_user_action=true
  But "Rachel is planning her wedding" → do NOT extract (no user action yet)
- For duration/cost: only extract if a specific number is mentioned or strongly implied.
- Each distinct user action = one fact. Do NOT merge multiple actions into one.
- If the user mentions attending an event, that's fact_type="attendance" with value=1, unit="count".

Document date for temporal reference: {document_date}

Conversation text:
---
{text}
---

Return ONLY a JSON array of fact objects, no other text. Return empty array [] if no quantifiable facts found."""

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
        # Ensure parent directories exist (fresh installs won't have ~/.ultramemory/)
        import os

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
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
            # Migration: add source_chunk_id column if only old source_chunk exists
            cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
            if "source_chunk_id" not in cols:
                conn.execute("ALTER TABLE memories ADD COLUMN source_chunk_id TEXT")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_chunk ON memories(source_chunk_id)"
                )

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

    # ── Multimodal embedding helpers ────────────────────────────────

    SUPPORTED_MEDIA_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
    }

    def _get_genai_client(self):
        """Lazy-initialize and cache the Google GenAI client."""
        if not hasattr(self, "_genai_client") or self._genai_client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required for multimodal media embedding. "
                    "Install with: pip install ultramemory[gemini]"
                ) from None

            import os

            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required "
                    "for multimodal media embedding."
                )
            self._genai_client = genai.Client(api_key=api_key)
        return self._genai_client

    def _embed_media(self, file_path: str) -> np.ndarray:
        """Embed a media file using Gemini's multimodal embedding API.

        Requires: pip install ultramemory[gemini] and GOOGLE_API_KEY env var.
        Only works when embedding_model contains 'gemini'.
        """
        import os

        from google.genai import types

        ext = os.path.splitext(file_path)[1].lower()
        mime_type = self.SUPPORTED_MEDIA_TYPES.get(ext)
        if not mime_type:
            supported = ", ".join(sorted(self.SUPPORTED_MEDIA_TYPES.keys()))
            raise ValueError(f"Unsupported media format '{ext}'. Supported: {supported}")

        with open(file_path, "rb") as f:
            data = f.read()

        client = self._get_genai_client()
        response = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=[types.Part.from_bytes(data=data, mime_type=mime_type)],
        )

        embedding = np.array(response.embeddings[0].values, dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _describe_media(self, file_path: str, user_description: str | None = None) -> str:
        """Generate a text description of a media file using the LLM."""
        import base64
        import os

        if user_description:
            return user_description

        ext = os.path.splitext(file_path)[1].lower()
        basename = os.path.basename(file_path)

        # For images: use vision-capable LLM via litellm
        if ext in (".png", ".jpg", ".jpeg"):
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            mime = self.SUPPORTED_MEDIA_TYPES[ext]
            data_url = f"data:{mime};base64,{b64}"

            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image in 1-3 sentences for use as a memory. "
                                    "Focus on the key subjects, setting, and any notable details. "
                                    "Be specific and factual."
                                ),
                            },
                        ],
                    }
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        # For audio/video: generate description from filename (LLM vision doesn't support these)
        media_type = "audio" if ext in (".mp3", ".wav") else "video"
        return f"{media_type.title()} file: {basename}"

    def ingest_media(
        self,
        file_path: str,
        session_key: str,
        agent_id: str,
        description: str | None = None,
        category: str | None = None,
        document_date: str | None = None,
    ) -> dict:
        """Ingest a media file (image, audio, video) as a memory.

        The media is embedded using Gemini's multimodal API directly,
        mapping it into the same vector space as text memories.
        A text description is generated for the content field.

        Returns dict with memory id and details.
        """
        import os

        if document_date is None:
            document_date = datetime.now().isoformat()[:10]

        # Validate file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Media file not found: {file_path}")

        # Validate extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_MEDIA_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_MEDIA_TYPES.keys()))
            raise ValueError(f"Unsupported media format '{ext}'. Supported: {supported}")

        # Validate embedding model is Gemini
        if "gemini" not in self._embedding_model.lower():
            raise ValueError(
                "Multimodal media ingestion requires a Gemini embedding model. "
                "Set embedding_model to 'gemini/gemini-embedding-2-preview' in your config."
            )

        # Get or generate text description
        desc = self._describe_media(file_path, description)

        # Embed the actual media bytes (NOT the text description)
        embedding = self._embed_media(file_path)

        # Determine media type label
        if ext in (".png", ".jpg", ".jpeg"):
            media_type = "image"
        elif ext in (".mp3", ".wav"):
            media_type = "audio"
        else:
            media_type = "video"

        category = category or "media"
        mem_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        abs_path = os.path.abspath(file_path)

        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Store source chunk with file path reference
            conn.execute(
                "INSERT INTO source_chunks (id, content, session_key, agent_id, document_date) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk_id, f"[media:{ext}] {abs_path}", session_key, agent_id, document_date),
            )

            # Store memory with multimodal embedding
            conn.execute(
                """INSERT INTO memories
                   (id, content, category, confidence, document_date,
                    source_session, source_agent, source_chunk_id, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem_id,
                    desc,
                    category,
                    1.0,
                    document_date,
                    session_key,
                    agent_id,
                    chunk_id,
                    self._vec_to_blob(embedding),
                ),
            )

            # Extract entities from the description text
            entities = []
            try:
                entity_response = self._llm_call(
                    f"Extract entity names (people, places, organizations) from this text. "
                    f"Return a JSON array of strings.\n\nText: {desc}\n\n"
                    f"Return ONLY a JSON array, no other text."
                )
                entities = self._parse_json(entity_response)
                if isinstance(entities, list):
                    self._store_entities(conn, mem_id, entities)
            except (json.JSONDecodeError, ValueError):
                pass

            conn.commit()
        finally:
            conn.close()

        return {
            "id": mem_id,
            "content": desc,
            "category": category,
            "media_type": media_type,
            "file_path": abs_path,
            "embedding_dim": len(embedding),
            "entities": entities,
        }

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
            # Safe merge: INSERT OR IGNORE to handle PK (memory_id, entity_name) collisions,
            # then DELETE the old rows. Avoids unique constraint violations when a memory
            # already has both old_name and new_name.
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_name, entity_type) "
                "SELECT memory_id, ?, entity_type FROM memory_entities WHERE entity_name = ?",
                (new_name.strip(), old_name.strip()),
            )
            conn.execute(
                "DELETE FROM memory_entities WHERE entity_name = ?",
                (old_name.strip(),),
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
            _fast = os.environ.get("ULTRAMEMORY_FAST_INGEST")

            # Load existing embeddings for dedup (skip in fast mode for performance)
            existing_rows = []
            existing_matrix = None
            if not _fast:
                existing_rows = conn.execute(
                    "SELECT id, embedding FROM memories WHERE is_current = 1 AND embedding IS NOT NULL"
                ).fetchall()

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
                # Skip exact content duplicates (cheap text check, keep even in fast mode)
                existing = conn.execute(
                    "SELECT id FROM memories WHERE content = ? AND is_current = 1 LIMIT 1",
                    (mem_data["content"],),
                ).fetchone()
                if existing:
                    continue

                embedding = embeddings[i]

                # Skip semantic near-duplicates (skip in fast mode)
                if not _fast and existing_matrix is not None and len(existing_rows) > 0:
                    sims = existing_matrix @ embedding
                    if float(np.max(sims)) > self._dedup_threshold:
                        continue

                mem_id = str(uuid.uuid4())
                insert_batch.append((mem_id, mem_data, embedding))

            # Single write transaction for all inserts + entity links
            if insert_batch:
                conn.execute("BEGIN IMMEDIATE")

                # Store source chunk once in normalized table
                chunk_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO source_chunks (id, content, session_key, agent_id, document_date) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, text, session_key, agent_id, document_date),
                )

                for mem_id, mem_data, embedding in insert_batch:
                    conn.execute(
                        """INSERT INTO memories
                           (id, content, category, confidence, document_date, event_date,
                            source_session, source_agent, source_chunk_id, embedding)
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
                            chunk_id,
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
        # Skip expensive phases in fast/batch mode
        _fast_mode = os.environ.get("ULTRAMEMORY_FAST_INGEST")
        if _fast_mode:
            return [{k: v for k, v in m.items() if k != "embedding"} for m in created_memories]

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
        if not os.environ.get("ULTRAMEMORY_SKIP_PROFILES"):
            all_entities = set()
            for mem in created_memories:
                for entity in mem.get("entities", []):
                    all_entities.add(entity)

            if all_entities:
                for entity_name in all_entities:
                    self._update_profile_safe(entity_name)

        # ── Phase 6: Structured fact extraction (no DB lock during LLM) ──
        # Skip if ULTRAMEMORY_SKIP_FACTS=1 (for batch ingestion; backfill later)
        if created_memories and not os.environ.get("ULTRAMEMORY_SKIP_FACTS"):
            sample_id = created_memories[0]["id"]
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT source_chunk_id FROM memories WHERE id = ?", (sample_id,)
                ).fetchone()
                chunk_id = row["source_chunk_id"] if row else None
            finally:
                conn.close()

            if chunk_id:
                try:
                    self.extract_facts(
                        text,
                        session_key=session_key,
                        chunk_id=chunk_id,
                        document_date=document_date,
                    )
                except Exception:
                    pass  # Non-critical: don't fail ingest if fact extraction fails

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
            PROFILE_PROMPT.format(entity_name=canonical, memories=memories_text)
        )
        try:
            profile_data = self._parse_json(profile_response)
        except (json.JSONDecodeError, ValueError):
            return

        static = json.dumps(profile_data.get("static_profile", {}))
        dynamic = json.dumps(profile_data.get("dynamic_profile", {}))

        # Phase C: Short write transaction (use canonical name consistently)
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT id FROM profiles WHERE entity_name = ?", (canonical,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE profiles SET static_profile = ?, dynamic_profile = ?, "
                    "updated_at = datetime('now') WHERE entity_name = ?",
                    (static, dynamic, canonical),
                )
            else:
                conn.execute(
                    "INSERT INTO profiles (id, entity_name, static_profile, dynamic_profile) "
                    "VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), canonical, static, dynamic),
                )
            conn.commit()
        finally:
            conn.close()

    # _update_profile removed — replaced by _update_profile_safe above

    # ── Event Extraction ─────────────────────────────────────────────────

    def extract_events(
        self,
        text: str,
        session_key: str,
        chunk_id: str | None = None,
        document_date: str | None = None,
    ) -> list[dict]:
        """
        Extract structured events from text, cluster them, and link to memories.

        Returns list of created event_mention dicts.
        """
        if document_date is None:
            document_date = datetime.now().isoformat()[:10]

        # LLM extraction (no DB lock)
        prompt = EVENT_EXTRACT_PROMPT.format(text=text, document_date=document_date)
        response = self._llm_call(prompt)
        try:
            events = self._parse_json(response)
        except (json.JSONDecodeError, ValueError):
            return []

        if not events or not isinstance(events, list):
            return []

        created = []
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")

            for ev in events:
                if not isinstance(ev, dict):
                    continue
                event_type = ev.get("event_type", "").strip()
                if not event_type:
                    continue

                mention_id = str(uuid.uuid4())
                participants = ev.get("participants", [])
                if not isinstance(participants, list):
                    participants = []
                participants_json = json.dumps(
                    sorted([str(p).strip() for p in participants if str(p).strip()])
                )

                summary = ev.get("summary", "").strip()
                if not summary:
                    continue

                subtype = ev.get("subtype")
                if subtype:
                    subtype = subtype.strip()
                time_text = ev.get("time_text")
                normalized_date = ev.get("normalized_date")
                duration_minutes = ev.get("duration_minutes")
                user_involvement = ev.get("user_involvement", "attended")
                confidence = ev.get("confidence", 0.5)

                # Insert event mention
                conn.execute(
                    """INSERT INTO event_mentions
                       (id, session_key, source_chunk_id, event_type, subtype,
                        summary, participants, time_text, normalized_date,
                        duration_minutes, user_involvement, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        mention_id,
                        session_key,
                        chunk_id,
                        event_type,
                        subtype,
                        summary,
                        participants_json,
                        time_text,
                        normalized_date,
                        duration_minutes,
                        user_involvement,
                        confidence,
                    ),
                )

                # Link to memories sharing the same chunk_id
                if chunk_id:
                    memory_rows = conn.execute(
                        "SELECT id FROM memories WHERE source_chunk_id = ?",
                        (chunk_id,),
                    ).fetchall()
                    for row in memory_rows:
                        conn.execute(
                            "INSERT OR IGNORE INTO event_mention_memories (event_id, memory_id) "
                            "VALUES (?, ?)",
                            (mention_id, row["id"]),
                        )

                # Deterministic clustering by distinct_key
                sorted_participants = json.loads(participants_json)
                distinct_key = self._compute_event_distinct_key(
                    event_type, sorted_participants, normalized_date
                )

                existing_cluster = conn.execute(
                    "SELECT id FROM event_clusters WHERE distinct_key = ?",
                    (distinct_key,),
                ).fetchone()

                if existing_cluster:
                    cluster_id = existing_cluster["id"]
                    conn.execute(
                        "INSERT OR IGNORE INTO event_cluster_members (cluster_id, event_id) "
                        "VALUES (?, ?)",
                        (cluster_id, mention_id),
                    )
                    # Update cluster confidence if this mention is higher
                    conn.execute(
                        "UPDATE event_clusters SET confidence = MAX(confidence, ?) WHERE id = ?",
                        (confidence, cluster_id),
                    )
                else:
                    cluster_id = str(uuid.uuid4())
                    conn.execute(
                        """INSERT INTO event_clusters
                           (id, event_type, subtype, canonical_label, distinct_key,
                            participants, normalized_date, duration_minutes,
                            user_involvement, confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            cluster_id,
                            event_type,
                            subtype,
                            summary,
                            distinct_key,
                            participants_json,
                            normalized_date,
                            duration_minutes,
                            user_involvement,
                            confidence,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO event_cluster_members (cluster_id, event_id) VALUES (?, ?)",
                        (cluster_id, mention_id),
                    )

                created.append(
                    {
                        "id": mention_id,
                        "event_type": event_type,
                        "subtype": subtype,
                        "summary": summary,
                        "participants": sorted_participants,
                        "cluster_id": cluster_id,
                        "distinct_key": distinct_key,
                    }
                )

            conn.commit()
        finally:
            conn.close()

        return created

    @staticmethod
    def _compute_event_distinct_key(
        event_type: str,
        participants: list[str],
        normalized_date: str | None,
    ) -> str:
        """Compute a deterministic key for event clustering.

        Key format: event_type|sorted_participants|normalized_date
        """
        parts = [event_type.lower().strip()]
        if participants:
            parts.append(",".join(p.lower().strip() for p in sorted(participants)))
        else:
            parts.append("")
        parts.append(normalized_date or "")
        return "|".join(parts)

    # ── Structured Fact Extraction ──────────────────────────────────────

    def extract_facts(
        self,
        text: str,
        session_key: str,
        chunk_id: str | None = None,
        document_date: str | None = None,
    ) -> list[dict]:
        """
        Extract structured, quantifiable facts from text and store in structured_facts.

        Links facts to memories sharing the same chunk_id.
        Returns list of created fact dicts.
        """
        if document_date is None:
            document_date = datetime.now().isoformat()[:10]

        # LLM extraction (no DB lock)
        prompt = FACT_EXTRACT_PROMPT.format(text=text, document_date=document_date)
        response = self._llm_call(prompt)
        try:
            facts = self._parse_json(response)
        except (json.JSONDecodeError, ValueError):
            return []

        if not facts or not isinstance(facts, list):
            return []

        created = []
        conn = self._conn()
        try:
            # Find memory_ids linked to this chunk
            memory_ids = []
            if chunk_id:
                memory_rows = conn.execute(
                    "SELECT id FROM memories WHERE source_chunk_id = ?",
                    (chunk_id,),
                ).fetchall()
                memory_ids = [r["id"] for r in memory_rows]

            if not memory_ids:
                # No memories to link — skip storing facts without a parent
                return []

            conn.execute("BEGIN IMMEDIATE")

            for fact in facts:
                if not isinstance(fact, dict):
                    continue

                fact_type = (fact.get("fact_type") or "").strip()
                category = (fact.get("category") or "").strip()
                subject = (fact.get("subject") or "").strip()
                predicate = (fact.get("predicate") or "").strip()

                if not fact_type or not category or not subject or not predicate:
                    continue

                value = fact.get("value")
                if value is not None:
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = None

                unit = fact.get("unit")
                date = fact.get("date")
                confidence = fact.get("confidence", 1.0)
                is_user_action = fact.get("is_user_action", True)

                # Create one fact row per linked memory
                for memory_id in memory_ids:
                    fact_id = str(uuid.uuid4())
                    conn.execute(
                        """INSERT INTO structured_facts
                           (id, memory_id, source_chunk_id, session_key,
                            fact_type, category, subject, predicate,
                            value, unit, date, confidence, is_user_action)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            fact_id,
                            memory_id,
                            chunk_id,
                            session_key,
                            fact_type,
                            category,
                            subject,
                            predicate,
                            value,
                            unit,
                            date,
                            confidence,
                            1 if is_user_action else 0,
                        ),
                    )

                created.append(
                    {
                        "fact_type": fact_type,
                        "category": category,
                        "subject": subject,
                        "predicate": predicate,
                        "value": value,
                        "unit": unit,
                        "date": date,
                        "confidence": confidence,
                        "is_user_action": is_user_action,
                        "linked_memories": len(memory_ids),
                    }
                )

            conn.commit()
        finally:
            conn.close()

        return created

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
                f"""SELECT m.id, m.content, m.category, m.confidence, m.document_date,
                          m.event_date, m.source_session, m.source_agent,
                          m.source_chunk_id, m.version, m.is_current, m.superseded_by,
                          sc.content as source_chunk
                   FROM memories m
                   LEFT JOIN source_chunks sc ON m.source_chunk_id = sc.id
                   WHERE m.id IN ({placeholders})""",
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
        """Return static + dynamic profile for an entity (resolves aliases)."""
        with self._conn() as conn:
            canonical = self._resolve_entity(conn, entity_name)
            row = conn.execute(
                "SELECT * FROM profiles WHERE entity_name = ?",
                (canonical,),
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

    # ── Re-embedding ──────────────────────────────────────────────────────

    def reembed_all(
        self,
        batch_size: int = 100,
        dry_run: bool = False,
        progress_callback=None,
    ) -> dict:
        """Re-embed all current memories using the currently configured provider/model.

        Args:
            batch_size: Number of memories to embed per batch.
            dry_run: If True, only count tokens and estimate cost without embedding.
            progress_callback: Optional callable(reembedded, total) for progress updates.

        Returns:
            Dict with total, reembedded, estimated_tokens, estimated_cost_usd, dry_run.
        """
        conn = self._conn()
        try:
            rows = conn.execute("SELECT id, content FROM memories WHERE is_current = 1").fetchall()
        finally:
            conn.close()

        total = len(rows)
        if total == 0:
            return {
                "total": 0,
                "reembedded": 0,
                "estimated_tokens": 0,
                "estimated_cost_usd": 0.0,
                "dry_run": dry_run,
            }

        # Estimate tokens: ~4 chars per token
        estimated_tokens = sum(len(r["content"]) for r in rows) // 4
        estimated_cost = estimated_tokens * 0.20 / 1_000_000

        if dry_run:
            return {
                "total": total,
                "reembedded": 0,
                "estimated_tokens": estimated_tokens,
                "estimated_cost_usd": estimated_cost,
                "dry_run": True,
            }

        reembedded = 0
        for i in range(0, total, batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["content"] for r in batch]
            ids = [r["id"] for r in batch]

            embeddings = self._embed_batch(texts)

            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for j, mem_id in enumerate(ids):
                    blob = self._vec_to_blob(embeddings[j])
                    conn.execute(
                        "UPDATE memories SET embedding = ?, updated_at = datetime('now') WHERE id = ?",
                        (blob, mem_id),
                    )
                conn.commit()
                reembedded += len(batch)
            except Exception:
                conn.rollback()
                conn.close()
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            if progress_callback:
                progress_callback(reembedded, total)

        return {
            "total": total,
            "reembedded": reembedded,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost,
            "dry_run": False,
        }

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
