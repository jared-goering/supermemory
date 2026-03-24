"""FastAPI wrapper around MemoryEngine for the visualization UI."""

from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from supermemory.config import get_config
from supermemory.engine import MemoryEngine

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
                  source_session, version, is_current, embedding
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
    return {"status": "ok", "memories": count, "source_chunks": chunks, "version": "0.2.0"}


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
    memories = engine.ingest(
        req.text,
        session_key=req.session_key,
        agent_id=req.agent_id,
        document_date=req.document_date,
    )
    # Auto-refresh cache after ingest so search/recall see new memories immediately
    if memories:
        global _embed_matrix, _embed_meta, _cache_built_at
        _embed_matrix, _embed_meta = _build_embedding_cache()
        _cache_built_at = datetime.now()
    return {"memories": memories, "count": len(memories)}


@app.post("/api/search")
async def search(req: SearchRequest):
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
            if not req.include_source:
                result.pop("source_chunk", None)
            results.append(result)

        # Hydrate relations from DB for top results
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
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
    agent_id: str | None = None


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
    agent_id: str
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


@app.post("/api/cache/refresh")
async def refresh_cache():
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
    """Entry point for supermemory serve command."""
    import uvicorn

    uvicorn.run(app, host=cfg["api_host"], port=cfg["api_port"])


if __name__ == "__main__":
    main()
