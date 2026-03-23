# Supermemory

Local-first AI memory engine for agents. Supermemory extracts atomic facts from conversations, versions them with relational semantics (update, extend, contradict), grounds everything in time, and stores it all in SQLite with local embeddings. No SaaS, no cloud dependency — your memory stays on your machine.

## Why Supermemory?

**Relational versioning.** When you tell an agent you moved from Seattle to Portland, most memory systems just append a new fact. Supermemory detects the relationship: the new memory *updates* the old one. Old facts get marked as superseded, not deleted — so you can query "where did Jared live in January?" and get the right answer. Relations include `updates`, `extends`, `contradicts`, `supports`, and `derives`.

**Temporal grounding.** Every memory carries two timestamps: `documentDate` (when it was recorded) and `eventDate` (when it happened). This lets you do time-travel queries — search memories as-of any date and get the facts that were current at that point in time.

**Local-first.** SQLite for storage, sentence-transformers for embeddings, any LLM via litellm for extraction. Nothing leaves your machine except the LLM calls, and you choose the provider.

## Quickstart

```bash
pip install openclaw-supermemory
export ANTHROPIC_API_KEY=sk-ant-...    # or any litellm-supported provider
supermemory init
supermemory ingest --text "Jared works at OpenClaw. He lives in Portland." --session demo --agent kit
supermemory search "Where does Jared live?"
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `supermemory init` | Create `~/.supermemory/` with default config and empty database |
| `supermemory ingest --text TEXT --session KEY --agent ID` | Extract and store memories from text |
| `supermemory ingest --file PATH --session KEY --agent ID` | Extract and store memories from a file |
| `supermemory search QUERY` | Semantic search over current memories |
| `supermemory search QUERY --all-versions` | Include superseded memories in results |
| `supermemory search QUERY --as-of 2025-06-01` | Time-travel: search as of a specific date |
| `supermemory history ENTITY` | Show version history for an entity |
| `supermemory profile ENTITY` | Show auto-built entity profile |
| `supermemory stats` | Show database statistics |

Options: `--db PATH` overrides the database path for any command.

## API Reference

Start the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8642
# or
python server.py
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check with memory count and version |
| `POST` | `/api/ingest` | Ingest text → extract memories |
| `POST` | `/api/search` | Semantic search with filters |
| `POST` | `/api/recall` | Fast recall using in-memory embedding cache |
| `POST` | `/api/startup-context` | Multi-query context block for agent startup |
| `GET` | `/api/graph` | All memories and relations as nodes + edges |
| `GET` | `/api/stats` | Database statistics |
| `GET` | `/api/history/{entity}` | Version history for an entity |
| `GET` | `/api/profile/{entity}` | Auto-built entity profile |
| `GET` | `/api/entities` | List all known entities |
| `POST` | `/api/cache/refresh` | Rebuild in-memory embedding cache |

### Ingest request

```json
{
  "text": "Jared moved to Portland in March.",
  "session_key": "daily-standup",
  "agent_id": "kit",
  "document_date": "2025-03-15"
}
```

### Search request

```json
{
  "query": "Where does Jared live?",
  "top_k": 10,
  "current_only": true,
  "as_of_date": null
}
```

## Configuration

Supermemory loads config from (highest priority wins):

1. Environment variables (`SUPERMEMORY_*`)
2. `./supermemory.yaml` (project-local)
3. `~/.supermemory/config.yaml` (user-global)
4. Built-in defaults

### supermemory.yaml

```yaml
# Database path (SQLite)
db_path: ~/.supermemory/memory.db

# LLM model (any litellm-compatible model string)
model: anthropic/claude-haiku-4-5

# Embedding model (sentence-transformers model name)
embedding_model: all-MiniLM-L6-v2
embedding_dim: 384

# API server
api_port: 8642
api_host: 0.0.0.0

# Semantic dedup threshold (0.0-1.0, higher = stricter)
dedup_threshold: 0.97

# Live ingest interval in seconds
ingest_interval: 900

# Regex patterns to skip during ingestion
skip_patterns:
  - "HEARTBEAT_OK"

# Directories to scan for session JSONL files
session_scan_dirs:
  - ~/.openclaw/agents
```

### Environment variables

| Variable | Config key | Default |
|----------|-----------|---------|
| `SUPERMEMORY_DB_PATH` | `db_path` | `~/.supermemory/memory.db` |
| `SUPERMEMORY_MODEL` | `model` | `anthropic/claude-haiku-4-5` |
| `SUPERMEMORY_EMBEDDING_MODEL` | `embedding_model` | `all-MiniLM-L6-v2` |
| `SUPERMEMORY_EMBEDDING_DIM` | `embedding_dim` | `384` |
| `SUPERMEMORY_API_PORT` | `api_port` | `8642` |
| `SUPERMEMORY_API_HOST` | `api_host` | `0.0.0.0` |
| `SUPERMEMORY_DEDUP_THRESHOLD` | `dedup_threshold` | `0.97` |

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │              Ingestion Pipeline                 │
                        │                                                 │
  Text/File ──────────► │  Extract (LLM)  ──►  Relate (LLM)  ──►  Store  │
                        │       │                    │               │    │
                        │  atomic facts         detect relations   SQLite │
                        │  + categories         UPDATE/EXTEND/     + BLOB │
                        │  + confidence         CONTRADICT/etc.  embeddings│
                        │                                                 │
                        └─────────────────────────────────────────────────┘

                        ┌─────────────────────────────────────────────────┐
                        │              Search Pipeline (no LLM)          │
                        │                                                 │
  Query ──────────────► │  Embed ──► Cosine Sim ──► Temporal ──► Expand  │
                        │  (local)   (matrix mul)   Filter      Relations │
                        │                                                 │
                        └─────────────────────────────────────────────────┘

  Storage: SQLite (memories, memory_relations, profiles)
  Embeddings: all-MiniLM-L6-v2, 384-dim float32, stored as BLOB
  LLM: litellm (any provider — OpenAI, Anthropic, Ollama, etc.)
```

### Ingestion (2-3 bounded LLM calls)

1. **Extract** — LLM splits text into atomic facts with category, confidence, and event dates
2. **Relate** — new facts compared to existing memories via embedding similarity, then LLM classifies the relationship type
3. **Profile** (optional) — entity profiles rebuilt from accumulated memories

### Search (zero LLM calls)

1. Embed query locally with sentence-transformers
2. Single matrix-vector multiply for all cosine similarities
3. Apply temporal and version filters
4. Expand related memories for context

## License

MIT

## Built by

[OpenClaw](https://github.com/openclaw) — open-source tools for AI agent infrastructure.
