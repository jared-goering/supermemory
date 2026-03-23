"""FastAPI wrapper around MemoryEngine for the visualization UI."""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supermemory.config import get_config
from supermemory.engine import MemoryEngine

cfg = get_config()

app = FastAPI(title="Memory Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = cfg["db_path"]
engine = MemoryEngine(db_path=DB_PATH)

# Pre-warm: load embedding matrix into memory on startup for fast search
import numpy as np
import sqlite3

def _build_embedding_cache():
    """Load all current embeddings into a numpy matrix for fast batch search."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
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
            metadata.append({
                "id": r["id"], "content": r["content"], "category": r["category"],
                "confidence": r["confidence"], "document_date": r["document_date"],
                "event_date": r["event_date"], "source_session": r["source_session"],
                "version": r["version"], "is_current": bool(r["is_current"]),
            })
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
    conn.close()
    return {"status": "ok", "memories": count, "version": "0.1.0"}


class IngestRequest(BaseModel):
    text: str
    session_key: str = "ui"
    agent_id: str = "user"
    document_date: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    current_only: bool = True
    as_of_date: Optional[str] = None


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    memories = engine.ingest(
        req.text,
        session_key=req.session_key,
        agent_id=req.agent_id,
        document_date=req.document_date,
    )
    return {"memories": memories, "count": len(memories)}


@app.post("/api/search")
async def search(req: SearchRequest):
    results = engine.search(
        req.query,
        top_k=req.top_k,
        current_only=req.current_only,
        as_of_date=req.as_of_date,
    )
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
        nodes.append({
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
        })

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
        edges.append({
            "source": r["from_memory"],
            "target": r["to_memory"],
            "type": r["relation"],
            "context": r["context"],
            "sourceContent": r["source_content"],
            "targetContent": r["target_content"],
        })

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
async def entities():
    """Return unique entity names."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    # entity_name may not exist as column; extract from content
    rows = conn.execute(
        "SELECT DISTINCT entity_name FROM profiles ORDER BY entity_name"
    ).fetchall()
    conn.close()
    return {"entities": [r[0] for r in rows]}


class RecallRequest(BaseModel):
    query: str
    top_k: int = 5
    agent_id: Optional[str] = None


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
    top_indices = np.argsort(similarities)[-req.top_k:][::-1]

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
    queries: list[str] = ["current projects and priorities", "recent decisions", "known issues and blockers"]
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
        top_indices = np.argsort(similarities)[-req.top_k_per_query * 2:][::-1]

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
    return {"status": "ok", "memories_cached": len([m for m in _embed_meta if m is not None]), "built_at": str(_cache_built_at)}


def main():
    """Entry point for supermemory serve command."""
    import uvicorn
    uvicorn.run(app, host=cfg["api_host"], port=cfg["api_port"])


if __name__ == "__main__":
    main()
