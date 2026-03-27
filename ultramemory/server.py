"""FastAPI wrapper around MemoryEngine for the visualization UI."""

import asyncio
import json
from datetime import datetime
from functools import partial

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from ultramemory.config import get_config
from ultramemory.engine import MemoryEngine

cfg = get_config()

app = FastAPI(title="Memory Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.get("cors_origins", ["http://localhost:3333", "http://127.0.0.1:3333"]),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ── Auth middleware ──────────────────────────────────────────────────────────

_api_key = cfg.get("api_key")


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    """Require X-API-Key header if api_key is configured."""
    if _api_key:
        # Skip auth for health endpoint
        if request.url.path == "/api/health":
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if key != _api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


DB_PATH = cfg["db_path"]
engine = MemoryEngine(db_path=DB_PATH)

# Pre-warm: load embedding matrix into memory on startup for fast search
import sqlite3

import numpy as np


def _build_embedding_cache():
    """Load all current embeddings into a numpy matrix for fast batch search."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, content, category, confidence, document_date, event_date,
                  source_session, source_chunk_id, version, is_current, embedding
           FROM memories WHERE is_current = 1 AND embedding IS NOT NULL"""
    ).fetchall()
    conn.close()

    if not rows:
        return None, []

    embed_dim = cfg["embedding_dim"]
    matrix = np.empty((len(rows), embed_dim), dtype=np.float32)
    metadata = []
    for i, r in enumerate(rows):
        blob = r["embedding"]
        if blob and len(blob) == embed_dim * 4:
            matrix[i] = np.frombuffer(blob, dtype=np.float32)
            metadata.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "category": r["category"],
                    "confidence": r["confidence"],
                    "document_date": r["document_date"],
                    "event_date": r["event_date"],
                    "source_session": r["source_session"],
                    "source_chunk_id": r["source_chunk_id"],
                    "version": r["version"],
                    "is_current": bool(r["is_current"]),
                }
            )
        else:
            matrix[i] = 0
            metadata.append(None)

    return matrix, metadata


_embed_matrix, _embed_meta = _build_embedding_cache()
_cache_built_at = datetime.now()


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH, timeout=10)
    count = conn.execute("SELECT COUNT(*) FROM memories WHERE is_current = 1").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0]
    conn.close()
    return {"status": "ok", "memories": count, "source_chunks": chunks, "version": "0.2.1"}


class IngestRequest(BaseModel):
    text: str
    session_key: str = "ui"
    agent_id: str = "user"
    document_date: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text_size(cls, v):
        max_bytes = cfg.get("max_ingest_bytes", 51200)
        if len(v.encode("utf-8")) > max_bytes:
            raise ValueError(f"Ingest text exceeds maximum size of {max_bytes} bytes")
        return v


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    current_only: bool = True
    as_of_date: str | None = None
    include_source: bool = False

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v):
        max_len = cfg.get("max_query_length", 1024)
        if len(v) > max_len:
            raise ValueError(f"Query exceeds maximum length of {max_len} characters")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        max_k = cfg.get("max_top_k", 100)
        if v > max_k:
            raise ValueError(f"top_k exceeds maximum of {max_k}")
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    loop = asyncio.get_event_loop()
    memories = await loop.run_in_executor(
        None,
        partial(
            engine.ingest,
            req.text,
            session_key=req.session_key,
            agent_id=req.agent_id,
            document_date=req.document_date,
        ),
    )
    # Auto-refresh cache after ingest so search/recall see new memories immediately
    if memories:
        global _embed_matrix, _embed_meta, _cache_built_at
        _embed_matrix, _embed_meta = _build_embedding_cache()
        _cache_built_at = datetime.now()
    return {"memories": memories, "count": len(memories)}


@app.post("/api/ingest-media")
async def ingest_media(
    file: UploadFile,
    session_key: str = Form("ui"),
    agent_id: str = Form("user"),
    description: str | None = Form(None),
    category: str | None = Form(None),
    document_date: str | None = Form(None),
):
    """Ingest a media file (image, audio, video) via multipart upload."""
    import os
    import tempfile

    # Save uploaded file to a temp location
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not ext:
        return JSONResponse(status_code=400, content={"detail": "File must have an extension"})

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                engine.ingest_media,
                file_path=tmp_path,
                session_key=session_key,
                agent_id=agent_id,
                description=description,
                category=category,
                document_date=document_date,
            ),
        )
        # Auto-refresh cache so search sees the new memory immediately
        global _embed_matrix, _embed_meta, _cache_built_at
        _embed_matrix, _embed_meta = _build_embedding_cache()
        _cache_built_at = datetime.now()
        return result
    except (ImportError, ValueError, FileNotFoundError) as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    finally:
        os.unlink(tmp_path)


class IngestRawRequest(BaseModel):
    """Lightweight ingest: chunk + embed without LLM extraction.
    Good for benchmarks and bulk RAG-style ingestion."""

    text: str
    session_key: str = "raw"
    agent_id: str = "raw"
    document_date: str | None = None
    chunk_size: int = 512  # chars per chunk
    chunk_overlap: int = 64

    @field_validator("text")
    @classmethod
    def validate_text_size(cls, v):
        max_bytes = cfg.get("max_ingest_bytes", 51200)
        if len(v.encode("utf-8")) > max_bytes:
            raise ValueError(f"Ingest text exceeds maximum size of {max_bytes} bytes")
        return v


@app.post("/api/ingest_raw")
async def ingest_raw(req: IngestRawRequest):
    """Chunk text and embed directly into memories, no LLM extraction.
    Much faster than /api/ingest (~10ms per chunk vs ~3s per LLM call)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_ingest_raw_sync, req))


def _ingest_raw_sync(req: IngestRawRequest):
    """Sync implementation of ingest_raw, runs in thread pool."""
    import uuid

    # Simple sentence-aware chunking
    text = req.text.strip()
    if not text:
        return {"count": 0, "chunks": 0}

    # Split into chunks respecting sentence boundaries
    chunks = []
    start = 0
    while start < len(text):
        end = start + req.chunk_size
        if end < len(text):
            # Try to break at a sentence boundary
            for sep in ["\n\n", "\n", ". ", "? ", "! "]:
                boundary = text.rfind(sep, start + req.chunk_size // 2, end + 50)
                if boundary > start:
                    end = boundary + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - req.chunk_overlap if end < len(text) else end

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Store source chunk
    source_id = None
    if len(text) > 200:
        chunk_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO source_chunks (id, content, session_key, agent_id, document_date) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, text[:5000], req.session_key, req.agent_id, req.document_date),
        )
        source_id = chunk_id

    inserted = 0
    for chunk in chunks:
        embedding = engine._embed(chunk)
        mem_id = str(uuid.uuid4())
        blob = embedding.tobytes() if hasattr(embedding, "tobytes") else None

        doc_date = req.document_date or datetime.now().strftime("%Y-%m-%d")
        conn.execute(
            """INSERT INTO memories (id, content, category, confidence, source_session, source_agent,
                       source_chunk_id, document_date, embedding, is_current, version, created_at)
               VALUES (?, ?, 'fact', 0.7, ?, ?, ?, ?, ?, 1, 1, datetime('now'))""",
            (mem_id, chunk, req.session_key, req.agent_id, source_id, doc_date, blob),
        )
        inserted += 1

    conn.commit()
    conn.close()

    # Don't rebuild cache on every raw ingest - too expensive with many ingests.
    # Call POST /api/refresh_cache after bulk ingest, or search auto-refreshes.

    return {"count": inserted, "chunks": len(chunks)}


@app.post("/api/refresh_cache")
async def refresh_cache():
    """Rebuild the in-memory embedding cache from the database."""
    global _embed_matrix, _embed_meta, _cache_built_at
    _embed_matrix, _embed_meta = _build_embedding_cache()
    _cache_built_at = datetime.now()
    count = len(_embed_meta) if _embed_meta else 0
    return {"status": "ok", "cached_memories": count}


@app.post("/api/search")
async def search(req: SearchRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_search_sync, req))


def _search_sync(req: SearchRequest):
    global _embed_matrix, _embed_meta

    # Use in-memory cache for current_only searches without as_of_date (fast path)
    if (
        req.current_only
        and not req.as_of_date
        and _embed_matrix is not None
        and len(_embed_meta) > 0
    ):
        query_vec = engine._embed(req.query)
        similarities = _embed_matrix @ query_vec

        # O(n) argpartition for top-k
        n = len(similarities)
        k = min(req.top_k, n)
        if n > k:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            meta = _embed_meta[idx]
            if meta is None:
                continue
            result = {
                **meta,
                "similarity": float(similarities[idx]),
                "relations": [],
            }
            # Remove source_chunk_id from output (internal ref only)
            result.pop("source_chunk_id", None)
            results.append(result)

        # Hydrate relations (and optionally source_chunks) from DB for top results
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row

        # Hydrate source_chunk text if requested
        if req.include_source:
            chunk_ids = [
                _embed_meta[idx].get("source_chunk_id")
                for idx in top_indices
                if _embed_meta[idx] is not None and _embed_meta[idx].get("source_chunk_id")
            ]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                chunk_rows = conn.execute(
                    f"SELECT id, content FROM source_chunks WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                chunk_map = {r["id"]: r["content"] for r in chunk_rows}
                for r_idx, idx in enumerate(top_indices):
                    meta = _embed_meta[idx]
                    if meta and meta.get("source_chunk_id"):
                        results[r_idx]["source_chunk"] = chunk_map.get(meta["source_chunk_id"])

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
        conn.close()
        return {"results": results, "count": len(results)}

    # Fallback to engine.search for as_of_date or non-current queries
    results = engine.search(
        req.query,
        top_k=req.top_k,
        current_only=req.current_only,
        as_of_date=req.as_of_date,
    )
    if not req.include_source:
        for r in results:
            r.pop("source_chunk", None)
    return {"results": results, "count": len(results)}


# ── Entity-Aware Search ─────────────────────────────────────────────────────


class EntitySearchRequest(BaseModel):
    query: str
    top_k: int = 30
    entity_expand_k: int = 50
    include_source: bool = False
    current_only: bool = True

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v):
        max_len = cfg.get("max_query_length", 1024)
        if len(v) > max_len:
            raise ValueError(f"Query exceeds maximum length of {max_len} characters")
        return v


@app.post("/api/search_entities")
async def search_entities(req: EntitySearchRequest):
    """Entity-aware search: vector search + entity expansion for cross-session retrieval."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_entity_search_sync, req))


def _entity_search_sync(req: EntitySearchRequest):
    """
    Three-phase entity-aware search:
    1. Standard vector search for initial results
    2. Extract named entities from content of top results using simple NLP
    3. Do secondary vector searches for each extracted entity to find cross-session memories
    4. Also expand via memory_entities join table where available
    5. Merge, deduplicate, group by entity
    """
    import re
    import sqlite3 as _sqlite3

    # Phase 1: Standard vector search
    vector_results = _search_sync(
        SearchRequest(
            query=req.query,
            top_k=req.top_k,
            include_source=req.include_source,
            current_only=req.current_only,
        )
    )
    vector_memories = vector_results.get("results", [])

    conn = _sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = _sqlite3.Row

    vector_ids = set(m["id"] for m in vector_memories)

    # Phase 2: Extract entities from memory_entities table AND from content via regex
    entity_counts = {}
    if vector_ids:
        placeholders = ",".join("?" for _ in vector_ids)
        entity_rows = conn.execute(
            f"SELECT entity_name, COUNT(*) as cnt FROM memory_entities WHERE memory_id IN ({placeholders}) GROUP BY entity_name",
            list(vector_ids),
        ).fetchall()
        entity_counts = {r["entity_name"]: r["cnt"] for r in entity_rows}

    # Extract proper nouns from memory content
    name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    stop_words = {
        "the",
        "this",
        "that",
        "they",
        "their",
        "there",
        "when",
        "what",
        "where",
        "which",
        "while",
        "with",
        "would",
        "will",
        "some",
        "about",
        "after",
        "before",
        "between",
        "each",
        "every",
        "first",
        "last",
        "most",
        "next",
        "other",
        "user",
        "memory",
        "date",
        "recently",
        "approximately",
        "several",
        "many",
        "few",
        "new",
        "old",
        "also",
        "however",
        "been",
        "has",
        "had",
        "was",
        "were",
        "are",
        "not",
    }
    content_entities = {}
    for m in vector_memories:
        content = m.get("content", "")
        for name in name_pattern.findall(content):
            if name.lower() not in stop_words and len(name) >= 2:
                content_entities[name] = content_entities.get(name, 0) + 1

    # Direct entity match in query
    query_lower = req.query.lower()
    direct_matches = set()
    for r in conn.execute("SELECT DISTINCT entity_name FROM memory_entities").fetchall():
        ename = r["entity_name"]
        if len(ename) > 2 and ename.lower() in query_lower:
            direct_matches.add(ename)

    # Combine: any entity from join table or content extraction that appears 1+ times
    # (being less strict - we'll filter noise at the context building stage)
    found_entities = set(direct_matches)
    for ename, _cnt in entity_counts.items():
        found_entities.add(ename)
    for ename, cnt in content_entities.items():
        if cnt >= 2:
            found_entities.add(ename)

    generic_entities = {
        "user",
        "AI",
        "United States",
        "US",
        "the user",
        "they",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
    found_entities -= generic_entities

    # Phase 3: Multi-strategy entity expansion
    entity_memory_ids = {}  # entity -> set of memory_ids

    # 3a: Expand via memory_entities join table
    for entity_name in found_entities:
        rows = conn.execute(
            """SELECT me.memory_id FROM memory_entities me
               JOIN memories m ON me.memory_id = m.id
               WHERE me.entity_name = ? AND m.is_current = 1""",
            (entity_name,),
        ).fetchall()
        if rows:
            entity_memory_ids[entity_name] = {r["memory_id"] for r in rows}

    # 3b: Content-based expansion: find memories containing entity name in text
    for entity_name in found_entities:
        rows = conn.execute(
            """SELECT id FROM memories
               WHERE content LIKE ? AND is_current = 1
               LIMIT 20""",
            (f"%{entity_name}%",),
        ).fetchall()
        if rows:
            if entity_name not in entity_memory_ids:
                entity_memory_ids[entity_name] = set()
            entity_memory_ids[entity_name].update(r["id"] for r in rows)

    # 3c: Secondary vector searches for entities not well covered by text search
    for entity_name in found_entities:
        current_count = len(entity_memory_ids.get(entity_name, set()))
        if current_count < 3:  # Only do vector search if we found few matches
            sec_results = _search_sync(
                SearchRequest(
                    query=entity_name,
                    top_k=5,
                    include_source=False,
                    current_only=True,
                )
            )
            sec_mems = [m for m in sec_results.get("results", []) if m.get("similarity", 0) >= 0.4]
            if sec_mems:
                if entity_name not in entity_memory_ids:
                    entity_memory_ids[entity_name] = set()
                entity_memory_ids[entity_name].update(m["id"] for m in sec_mems)

    # Collect all expanded memory IDs
    all_expanded_ids = set()
    for mids in entity_memory_ids.values():
        all_expanded_ids.update(mids)

    # Hydrate new memories not in vector results
    new_ids = all_expanded_ids - vector_ids
    entity_expanded = []
    if new_ids:
        new_ids_list = list(new_ids)[: req.entity_expand_k]
        placeholders = ",".join("?" for _ in new_ids_list)
        rows = conn.execute(
            f"""SELECT m.id, m.content, m.category, m.confidence,
                       m.document_date, m.event_date, m.source_session,
                       m.source_agent, m.version, m.is_current
                FROM memories m
                WHERE m.id IN ({placeholders})""",
            new_ids_list,
        ).fetchall()

        for r in rows:
            entity_expanded.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "category": r["category"],
                    "confidence": r["confidence"],
                    "document_date": r["document_date"],
                    "event_date": r["event_date"],
                    "source_session": r["source_session"],
                    "version": r["version"],
                    "is_current": bool(r["is_current"]),
                    "similarity": 0.0,
                    "source": "entity_expansion",
                    "relations": [],
                }
            )

    # Tag all results with their entities
    all_memories_map = {}
    for m in vector_memories:
        m["source"] = "vector"
        m["entities"] = []
        all_memories_map[m["id"]] = m
    for m in entity_expanded:
        m["entities"] = []
        all_memories_map[m["id"]] = m

    for ename, mids in entity_memory_ids.items():
        for mid in mids:
            if mid in all_memories_map:
                all_memories_map[mid]["entities"].append(ename)

    # Merge: vector first, then entity-expanded
    all_results = vector_memories + entity_expanded

    conn.close()

    return {
        "results": all_results,
        "count": len(all_results),
        "vector_count": len(vector_memories),
        "entity_expanded_count": len(entity_expanded),
        "entities_found": sorted(found_entities),
        "entity_groups": {k: len(v) for k, v in entity_memory_ids.items()},
    }


@app.get("/api/graph")
async def graph():
    """Return all memories and relations as nodes + edges for visualization."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get all memories
    rows = conn.execute(
        """SELECT id, content, category, confidence,
                  document_date, event_date, is_current, version,
                  source_session, source_agent, created_at
           FROM memories ORDER BY created_at"""
    ).fetchall()

    nodes = []
    for r in rows:
        nodes.append(
            {
                "id": r["id"],
                "content": r["content"],
                "category": r["category"],
                "confidence": r["confidence"],
                "documentDate": r["document_date"],
                "eventDate": r["event_date"],
                "isCurrent": bool(r["is_current"]),
                "version": r["version"],
                "session": r["source_session"],
                "agent": r["source_agent"],
                "createdAt": r["created_at"],
            }
        )

    # Get all relations
    rel_rows = conn.execute(
        """SELECT mr.from_memory, mr.to_memory, mr.relation, mr.context,
                  ms.content as source_content, mt.content as target_content
           FROM memory_relations mr
           JOIN memories ms ON mr.from_memory = ms.id
           JOIN memories mt ON mr.to_memory = mt.id"""
    ).fetchall()

    edges = []
    for r in rel_rows:
        edges.append(
            {
                "source": r["from_memory"],
                "target": r["to_memory"],
                "type": r["relation"],
                "context": r["context"],
                "sourceContent": r["source_content"],
                "targetContent": r["target_content"],
            }
        )

    conn.close()
    return {"nodes": nodes, "edges": edges}


@app.get("/api/stats")
async def stats():
    return engine.get_stats()


@app.get("/api/history/{entity_name}")
async def history(entity_name: str):
    return {"history": engine.get_history(entity_name)}


@app.get("/api/profile/{entity_name}")
async def profile(entity_name: str):
    return engine.get_profile(entity_name)


@app.get("/api/entities")
async def entities(min_mentions: int = 1):
    """Return entities with mention counts from the join table, with profile status."""
    entity_list = engine.list_entities(min_mentions=min_mentions)

    # Enrich with profile existence check
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    profile_names = {r[0] for r in conn.execute("SELECT entity_name FROM profiles").fetchall()}
    conn.close()

    for e in entity_list:
        e["has_profile"] = e["entity_name"] in profile_names

    return {"entities": entity_list, "count": len(entity_list)}


class MergeRequest(BaseModel):
    old_name: str
    new_name: str


@app.post("/api/entities/merge")
async def merge_entities(req: MergeRequest):
    """Merge old_name into new_name: updates join table + adds alias."""
    engine.merge_entities(req.old_name, req.new_name)
    return {"status": "ok", "merged": req.old_name, "into": req.new_name}


class AliasRequest(BaseModel):
    alias: str
    canonical: str


@app.post("/api/entities/alias")
async def add_alias(req: AliasRequest):
    """Register an alias for an entity name."""
    engine.add_entity_alias(req.alias, req.canonical)
    return {"status": "ok", "alias": req.alias, "canonical": req.canonical}


class RecallRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/recall")
async def recall(req: RecallRequest):
    """
    Lightweight recall endpoint for agent startup.
    Uses in-memory embedding cache for sub-100ms search.
    Returns compact text blocks suitable for prompt injection.
    """
    global _embed_matrix, _embed_meta

    if _embed_matrix is None or len(_embed_meta) == 0:
        return {"text": "(no memories loaded)", "count": 0}

    # Embed query using engine's embedder (already warm in server process)
    query_vec = engine._embed(req.query)

    # Single matrix-vector multiply: all similarities at once
    similarities = _embed_matrix @ query_vec

    # Get top_k indices
    top_indices = np.argsort(similarities)[-req.top_k :][::-1]

    lines = []
    for idx in top_indices:
        meta = _embed_meta[idx]
        if meta is None:
            continue
        sim = float(similarities[idx])
        cat = meta["category"]
        content = meta["content"]
        version = meta["version"]
        lines.append(f"[{cat}] {content} (v{version}, {sim:.0%} match)")

    return {
        "text": "\n".join(lines),
        "count": len(lines),
    }


class StartupContextRequest(BaseModel):
    agent_id: str | None = None  # reserved for future agent-scoped filtering
    queries: list[str] = [
        "current projects and priorities",
        "recent decisions",
        "known issues and blockers",
    ]
    top_k_per_query: int = 3


@app.post("/api/startup-context")
async def startup_context(req: StartupContextRequest):
    """
    Generate a compact context block for agent session startup.
    Uses in-memory cache for fast multi-query recall. Deduplicates across queries.
    """
    global _embed_matrix, _embed_meta

    if _embed_matrix is None:
        return {"context": "(no memories)", "memory_count": 0, "queries": req.queries}

    seen_ids = set()
    sections = []

    for query in req.queries:
        query_vec = engine._embed(query)
        similarities = _embed_matrix @ query_vec
        top_indices = np.argsort(similarities)[-req.top_k_per_query * 2 :][::-1]

        lines = []
        for idx in top_indices:
            meta = _embed_meta[idx]
            if meta is None or meta["id"] in seen_ids:
                continue
            seen_ids.add(meta["id"])
            lines.append(f"- [{meta['category']}] {meta['content']}")
            if len(lines) >= req.top_k_per_query:
                break
        if lines:
            sections.append(f"## {query.title()}\n" + "\n".join(lines))

    context_block = "\n\n".join(sections)
    return {
        "context": context_block,
        "memory_count": len(seen_ids),
        "queries": req.queries,
    }


# ── Event Aggregation + Search ─────────────────────────────────────────────

AGGREGATE_PARSE_PROMPT = """Parse this question into a structured aggregation intent.

Question: {question}

Return a JSON object with:
- "operation": one of "count_distinct" (how many different events) or "sum_duration" (total hours/minutes)
- "event_types": array of event type strings to match (e.g. ["wedding"], ["exercise"], ["art_event"])
- "subtypes": array of subtype strings if specific (e.g. ["yoga", "jogging"]) or empty array
- "time_scope": time scope string like "this_year", "last_week", "past_month", or null if no time constraint
- "user_involvement": filter like "attended", "did", or null for any involvement

Return ONLY a JSON object, no other text."""


class AggregateRequest(BaseModel):
    question: str
    session_prefix: str | None = None  # Filter events to sessions matching this prefix


class SearchEventsRequest(BaseModel):
    event_type: str | None = None
    subtype: str | None = None
    time_range: str | None = None
    participants: list[str] | None = None
    session_prefix: str | None = None  # Filter events to sessions matching this prefix
    limit: int = 20


@app.post("/api/aggregate")
async def aggregate(req: AggregateRequest):
    """Detect aggregate intent, query event_clusters, return deterministic answer."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_aggregate_sync, req))


def _aggregate_sync(req: AggregateRequest):
    # Parse question into structured intent via LLM
    prompt = AGGREGATE_PARSE_PROMPT.format(question=req.question)
    response = engine._llm_call(prompt)
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        intent = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"answer": None, "error": "Could not parse question intent"}

    operation = intent.get("operation", "count_distinct")
    event_types = intent.get("event_types", [])
    subtypes = intent.get("subtypes", [])
    user_involvement = intent.get("user_involvement")

    # Build SQL query against event_clusters
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    conditions = []
    params = []

    if event_types:
        placeholders = ",".join("?" for _ in event_types)
        conditions.append(f"event_type IN ({placeholders})")
        params.extend([t.lower() for t in event_types])

    if subtypes:
        placeholders = ",".join("?" for _ in subtypes)
        conditions.append(f"subtype IN ({placeholders})")
        params.extend([s.lower() for s in subtypes])

    if user_involvement:
        conditions.append("user_involvement = ?")
        params.append(user_involvement)

    # If session_prefix is set, filter clusters to those with mentions in matching sessions
    if req.session_prefix:
        conditions.append("""id IN (
            SELECT ecm.cluster_id FROM event_cluster_members ecm
            JOIN event_mentions em ON em.id = ecm.event_id
            WHERE em.session_key LIKE ?
        )""")
        params.append(f"bench_{req.session_prefix}%")

    where = " AND ".join(conditions) if conditions else "1=1"

    rows = conn.execute(
        f"""SELECT id, event_type, subtype, canonical_label, distinct_key,
                   participants, normalized_date, duration_minutes,
                   user_involvement, confidence
            FROM event_clusters WHERE {where}
            ORDER BY normalized_date DESC""",
        params,
    ).fetchall()

    events = [dict(r) for r in rows]
    conn.close()

    if operation == "count_distinct":
        answer = len(events)
    elif operation == "sum_duration":
        total = sum(e["duration_minutes"] or 0 for e in events)
        answer = round(total / 60, 2)  # Return in hours
    else:
        answer = len(events)

    return {
        "answer": answer,
        "operation": operation,
        "intent": intent,
        "events": events[:50],
    }


@app.post("/api/search_events")
async def search_events(req: SearchEventsRequest):
    """Search event_clusters with optional filters."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_search_events_sync, req))


def _search_events_sync(req: SearchEventsRequest):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    conditions = []
    params = []

    if req.event_type:
        conditions.append("ec.event_type = ?")
        params.append(req.event_type.lower())

    if req.subtype:
        conditions.append("ec.subtype = ?")
        params.append(req.subtype.lower())

    if req.participants:
        for p in req.participants:
            conditions.append("ec.participants LIKE ?")
            params.append(f"%{p}%")

    where = " AND ".join(conditions) if conditions else "1=1"

    clusters = conn.execute(
        f"""SELECT ec.id, ec.event_type, ec.subtype, ec.canonical_label,
                   ec.distinct_key, ec.participants, ec.normalized_date,
                   ec.duration_minutes, ec.user_involvement, ec.confidence
            FROM event_clusters ec
            WHERE {where}
            ORDER BY ec.normalized_date DESC
            LIMIT ?""",
        params + [req.limit],
    ).fetchall()

    results = []
    for cluster in clusters:
        c = dict(cluster)
        # Get mentions for this cluster
        mentions = conn.execute(
            """SELECT em.id, em.summary, em.session_key, em.time_text,
                      em.normalized_date, em.confidence
               FROM event_mentions em
               JOIN event_cluster_members ecm ON em.id = ecm.event_id
               WHERE ecm.cluster_id = ?""",
            (c["id"],),
        ).fetchall()
        c["mentions"] = [dict(m) for m in mentions]

        # Get linked memories
        if mentions:
            mention_ids = [m["id"] for m in mentions]
            placeholders = ",".join("?" for _ in mention_ids)
            memories = conn.execute(
                f"""SELECT DISTINCT mem.id, mem.content, mem.category
                    FROM memories mem
                    JOIN event_mention_memories emm ON mem.id = emm.memory_id
                    WHERE emm.event_id IN ({placeholders})""",
                mention_ids,
            ).fetchall()
            c["linked_memories"] = [dict(m) for m in memories]
        else:
            c["linked_memories"] = []

        results.append(c)

    conn.close()
    return {"results": results, "count": len(results)}


class AggregateSearchRequest(BaseModel):
    question: str
    session_prefix: str | None = None
    top_k: int = 50
    include_source: bool = True


@app.post("/api/aggregate_search")
async def aggregate_search(req: AggregateSearchRequest):
    """Multi-session aggregate search: broad memory retrieval + event clusters for counting/aggregation questions."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_aggregate_search_sync, req))


def _aggregate_search_sync(req: AggregateSearchRequest):
    """
    Combines three retrieval strategies for multi-session aggregate questions:
    1. Vector search with high top_k for broad coverage
    2. Event clusters matching the question topic
    3. Keyword-based memory scan for completeness

    Returns all evidence organized for LLM synthesis.
    """
    import re
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = _sqlite3.Row

    session_filter = f"bench_{req.session_prefix}%" if req.session_prefix else None

    # Phase 1: Vector search with high top_k
    vector_results = _search_sync(
        SearchRequest(
            query=req.question,
            top_k=req.top_k,
            include_source=req.include_source,
            current_only=True,
        )
    )
    vector_memories = vector_results.get("results", [])

    # Filter to session prefix if specified
    if session_filter:
        vector_memories = [
            m
            for m in vector_memories
            if m.get("source_session", "").startswith(session_filter.rstrip("%"))
        ]

    # Phase 2: Get event clusters for this question's sessions, filtered by topic relevance
    cluster_conditions = []
    cluster_params = []
    if session_filter:
        cluster_conditions.append("""ec.id IN (
            SELECT ecm.cluster_id FROM event_cluster_members ecm
            JOIN event_mentions em ON em.id = ecm.event_id
            WHERE em.session_key LIKE ?
        )""")
        cluster_params.append(session_filter)

    # Filter clusters by question keywords matching event_type, subtype, or canonical_label
    question_lower = req.question.lower()
    topic_stop = {
        "how",
        "many",
        "much",
        "did",
        "do",
        "have",
        "has",
        "had",
        "i",
        "my",
        "me",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "is",
        "was",
        "were",
        "what",
        "which",
        "when",
        "where",
        "who",
        "all",
        "each",
        "every",
        "any",
        "this",
        "that",
        "it",
        "we",
        "last",
        "past",
        "total",
        "different",
        "spend",
        "spent",
        "hours",
        "go",
        "went",
        "gone",
        "get",
        "got",
        "more",
        "than",
        "week",
        "month",
        "year",
        "time",
        "times",
        "about",
    }
    topic_words = re.findall(r"\b[a-z]+\b", question_lower)
    topic_keywords = [w for w in topic_words if w not in topic_stop and len(w) > 2]

    # Simple stemming: expand each keyword to include stem variants
    def _stem_variants(word):
        """Generate simple stem variants for LIKE matching."""
        variants = {word}
        # Strip common suffixes
        for suffix in ("ings", "ing", "tion", "tions", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                variants.add(word[: -len(suffix)])
        # Also add the word itself as a stem prefix
        if len(word) >= 4:
            variants.add(word[: max(3, len(word) - 2)])
        return variants

    if topic_keywords:
        kw_conditions = []
        for kw in topic_keywords:
            stems = _stem_variants(kw)
            for stem in stems:
                kw_conditions.append(
                    "(ec.event_type LIKE ? OR ec.subtype LIKE ? OR ec.canonical_label LIKE ?)"
                )
                cluster_params.extend([f"%{stem}%", f"%{stem}%", f"%{stem}%"])
        cluster_conditions.append("(" + " OR ".join(kw_conditions) + ")")

    cluster_where = " AND ".join(cluster_conditions) if cluster_conditions else "1=1"
    clusters = conn.execute(
        f"""SELECT ec.id, ec.event_type, ec.subtype, ec.canonical_label,
                   ec.distinct_key, ec.participants, ec.normalized_date,
                   ec.duration_minutes, ec.user_involvement, ec.confidence
            FROM event_clusters ec
            WHERE {cluster_where}
            ORDER BY ec.normalized_date""",
        cluster_params,
    ).fetchall()
    event_clusters = [dict(c) for c in clusters]

    # Phase 3: Keyword scan — search memories by topic keywords for completeness
    keyword_memories = []
    seen_ids = {m["id"] for m in vector_memories}
    keywords = topic_keywords  # reuse from Phase 2

    if keywords and session_filter:
        for kw in keywords[:5]:
            rows = conn.execute(
                """SELECT id, content, category, confidence, document_date,
                          event_date, source_session, source_chunk_id, version
                   FROM memories
                   WHERE content LIKE ? AND is_current = 1 AND source_session LIKE ?
                   LIMIT 30""",
                (f"%{kw}%", session_filter),
            ).fetchall()
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    keyword_memories.append(
                        {
                            "id": r["id"],
                            "content": r["content"],
                            "category": r["category"],
                            "confidence": r["confidence"],
                            "document_date": r["document_date"],
                            "event_date": r["event_date"],
                            "source_session": r["source_session"],
                            "source_chunk_id": r["source_chunk_id"],
                            "version": r["version"],
                            "is_current": True,
                            "similarity": 0.0,
                            "source": "keyword_scan",
                            "relations": [],
                        }
                    )

    # Source chunks for vector memories are already hydrated by _search_sync when include_source=True.
    # Hydrate source chunks for keyword memories.
    if req.include_source:
        for m in keyword_memories:
            chunk_id = m.get("source_chunk_id")
            if chunk_id and not m.get("source_chunk"):
                row = conn.execute(
                    "SELECT content FROM source_chunks WHERE id = ?", (chunk_id,)
                ).fetchone()
                if row:
                    m["source_chunk"] = row["content"]

    conn.close()

    all_memories = vector_memories + keyword_memories

    return {
        "memories": all_memories,
        "memory_count": len(all_memories),
        "vector_count": len(vector_memories),
        "keyword_count": len(keyword_memories),
        "event_clusters": event_clusters,
        "cluster_count": len(event_clusters),
    }


class ReembedRequest(BaseModel):
    batch_size: int = 100
    dry_run: bool = False


@app.post("/api/reembed")
async def reembed(req: ReembedRequest):
    """Re-embed all current memories with the configured embedding model."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(engine.reembed_all, batch_size=req.batch_size, dry_run=req.dry_run),
    )
    # Rebuild in-memory cache after re-embedding
    if not req.dry_run and result.get("reembedded", 0) > 0:
        global _embed_matrix, _embed_meta, _cache_built_at
        _embed_matrix, _embed_meta = _build_embedding_cache()
        _cache_built_at = datetime.now()
    return result


@app.post("/api/cache/refresh")
async def refresh_cache_v2():
    """Rebuild the in-memory embedding cache after new ingestions."""
    global _embed_matrix, _embed_meta, _cache_built_at
    _embed_matrix, _embed_meta = _build_embedding_cache()
    _cache_built_at = datetime.now()
    return {
        "status": "ok",
        "memories_cached": len([m for m in _embed_meta if m is not None]),
        "built_at": str(_cache_built_at),
    }


def main():
    """Entry point for ultramemory serve command."""
    import uvicorn

    uvicorn.run(app, host=cfg["api_host"], port=cfg["api_port"])


if __name__ == "__main__":
    main()
