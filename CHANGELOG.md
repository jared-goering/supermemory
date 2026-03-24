# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-03-23

### Fixed
- **Dedup FK violation:** Dedup now deletes `memory_entities` rows before memories (prevents foreign key constraint errors)
- **Fresh install crash:** Engine creates parent directories for DB path on init (`os.makedirs`)
- **Missing sentence-transformers:** Clear `ImportError` with install instructions instead of cryptic `ModuleNotFoundError`
- **Profile alias inconsistency:** `get_profile()` and `_update_profile_safe()` now resolve aliases to canonical names
- **Entity merge PK collision:** Uses `INSERT OR IGNORE` + `DELETE` pattern to avoid unique constraint violations when merging entities that share memories
- **Cached search ignores `include_source`:** In-memory cache path now hydrates source chunks from `source_chunks` table when requested
- **Phantom `agent_id` parameter:** Removed unused `agent_id` from recall endpoint; made optional in startup-context
- **`__version__` mismatch:** `supermemory/__init__.py` now matches `pyproject.toml`

### Changed
- Research background section added to README with benchmark attribution

## [0.2.0] - 2026-03-23

### Added
- **Security hardening:** API key auth (`X-API-Key` header), CORS origin locking, input validation (text ≤10K chars, query ≤100 chars), bind `127.0.0.1` by default
- **Entity system:** `memory_entities` join table, `entity_aliases` table, entity merge API, indexed entity lookups (replaces O(n) LIKE scans)
- **Entity endpoints:** `/api/entities`, `/api/entity/{name}`, `/api/entity/{name}/merge`
- **Source chunk normalization:** Deduplicated `source_chunks` table with FK (98% storage reduction on source text)
- **Migration tooling:** `supermemory.migrate_chunks` for upgrading existing databases
- `include_source` parameter on search (source text hidden by default)
- `supermemory serve` CLI command for starting the API server
- GitHub Actions CI (Python 3.10-3.12) with ruff linting step
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, CHANGELOG.md
- Pre-commit hooks (ruff linting + formatting, pytest)
- Issue and PR templates
- `PRAGMA foreign_keys=ON` enforced on all connections
- Dynamic `embedding_dim` config (replaces hardcoded 384)
- New env vars: `SUPERMEMORY_API_HOST`, `SUPERMEMORY_API_KEY`, `SUPERMEMORY_CORS_ORIGINS`

### Changed
- LLM calls decoupled from write transactions (no DB lock during API calls)
- Search ranks by ID+embedding only, then lazy-hydrates top-k results (no full-record scan)
- Moved all modules into `supermemory/` package (was flat root files)
- `sentence-transformers` is now an optional dependency: `pip install openclaw-supermemory[local]`
- Replaced personal data in tests/docs with generic examples
- Health endpoint now reports source chunk count and version

### Fixed
- Three real bugs: unused variable in dedup.py, exception chaining in engine.py, unused loop variable
- 94 ruff lint issues resolved

### Removed
- Machine-specific files (launchd plists, bulk logs)

### Security
- Default bind address changed from `0.0.0.0` to `127.0.0.1`
- CORS no longer defaults to `*` when origins are configured
- Scrubbed personal information from test fixtures, documentation, and git history
- `pip audit` clean (zero known vulnerabilities)

## [0.1.0] - 2026-03-23

### Added
- Core memory engine with SQLite storage
- Atomic fact extraction via LLM (litellm multi-provider)
- Relational versioning (updates, extends, contradicts, supports, derives)
- Temporal grounding (documentDate, eventDate, as-of-date queries)
- Semantic search with local embeddings (sentence-transformers)
- Entity profiles and version history
- FastAPI server with REST API
- CLI tools (init, ingest, search, stats, history, profile)
- Live ingest pipeline for OpenClaw session files
- Deduplication (exact + semantic at 0.97+ cosine similarity)
- React + Next.js visualization UI with 3D force graph
