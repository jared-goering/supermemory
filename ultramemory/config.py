"""
Ultramemory configuration system.

Loads config from (highest priority wins):
1. Environment variables (ULTRAMEMORY_DB_PATH, ULTRAMEMORY_MODEL, etc.)
2. ./ultramemory.yaml (project-local)
3. ~/.ultramemory/config.yaml (user-global)
4. Built-in defaults
"""

import os
from pathlib import Path

# Try to import yaml, but don't fail if not installed (defaults still work)
try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG = {
    "db_path": os.path.join(str(Path.home()), ".ultramemory", "memory.db"),
    "model": "anthropic/claude-haiku-4-5",
    "embedding_provider": "local",  # "local" (sentence-transformers) or "litellm" (any litellm-supported provider)
    "embedding_model": "all-MiniLM-L6-v2",  # local: model name; litellm: e.g. "text-embedding-3-small", "cohere/embed-english-v3.0"
    "embedding_dim": 384,  # must match the model (e.g. 1536 for text-embedding-3-small)
    "api_port": 8642,
    "api_host": "127.0.0.1",
    "log_level": "info",
    "api_key": None,  # Optional API key for authentication (X-API-Key header)
    "cors_origins": ["http://localhost:3333", "http://127.0.0.1:3333"],  # Allowed CORS origins
    "max_ingest_bytes": 51200,  # 50KB max ingest text size
    "max_query_length": 1024,  # 1KB max search query
    "max_top_k": 100,  # Max results per search
    "dedup_threshold": 0.97,
    "ingest_interval": 900,
    "skip_patterns": [],
    "session_scan_dirs": [],
    "state_file": os.path.join(str(Path.home()), ".ultramemory", "ingest-state.json"),
}

# Map of env var names → config keys
ENV_MAP = {
    "ULTRAMEMORY_DB_PATH": "db_path",
    "ULTRAMEMORY_MODEL": "model",
    "ULTRAMEMORY_EMBEDDING_PROVIDER": "embedding_provider",
    "ULTRAMEMORY_EMBEDDING_MODEL": "embedding_model",
    "ULTRAMEMORY_EMBEDDING_DIM": ("embedding_dim", int),
    "ULTRAMEMORY_API_PORT": ("api_port", int),
    "ULTRAMEMORY_API_HOST": "api_host",
    "ULTRAMEMORY_LOG_LEVEL": "log_level",
    "ULTRAMEMORY_DEDUP_THRESHOLD": ("dedup_threshold", float),
    "ULTRAMEMORY_INGEST_INTERVAL": ("ingest_interval", int),
    "ULTRAMEMORY_API_KEY": "api_key",
    "ULTRAMEMORY_CORS_ORIGINS": "cors_origins",
    "ULTRAMEMORY_MAX_INGEST_BYTES": ("max_ingest_bytes", int),
    "ULTRAMEMORY_MAX_QUERY_LENGTH": ("max_query_length", int),
    "ULTRAMEMORY_MAX_TOP_K": ("max_top_k", int),
    # Legacy env var support
    "MEMORY_DB": "db_path",
}


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning empty dict on failure."""
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _load_env() -> dict:
    """Load config from environment variables."""
    result = {}
    for env_key, mapping in ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if isinstance(mapping, tuple):
            config_key, cast_fn = mapping
            try:
                result[config_key] = cast_fn(val)
            except (ValueError, TypeError):
                pass
        else:
            result[mapping] = val

    # Handle list-type env vars
    cors = os.environ.get("ULTRAMEMORY_CORS_ORIGINS")
    if cors:
        result["cors_origins"] = [o.strip() for o in cors.split(",") if o.strip()]

    skip = os.environ.get("ULTRAMEMORY_SKIP_PATTERNS")
    if skip:
        result["skip_patterns"] = [p.strip() for p in skip.split(",") if p.strip()]

    scan_dirs = os.environ.get("ULTRAMEMORY_SESSION_SCAN_DIRS")
    if scan_dirs:
        result["session_scan_dirs"] = [d.strip() for d in scan_dirs.split(",") if d.strip()]

    return result


def load_config(config_path: str | None = None) -> dict:
    """Load and merge config from all sources.

    Priority (highest wins): env vars > project-local YAML > user-global YAML > defaults.
    If config_path is given, it is loaded instead of the automatic YAML search.
    """
    config = dict(DEFAULT_CONFIG)

    # Layer 1: User-global config
    global_path = os.path.join(str(Path.home()), ".ultramemory", "config.yaml")
    config.update(_load_yaml(global_path))

    # Layer 2: Project-local config
    local_path = os.path.join(os.getcwd(), "ultramemory.yaml")
    config.update(_load_yaml(local_path))

    # Layer 2b: Explicit config path overrides both
    if config_path:
        config.update(_load_yaml(config_path))

    # Layer 3: Environment variables (highest priority)
    config.update(_load_env())

    # Expand ~ in paths
    for key in ("db_path", "state_file"):
        if isinstance(config.get(key), str):
            config[key] = os.path.expanduser(config[key])

    # Expand ~ in session_scan_dirs
    if config.get("session_scan_dirs"):
        config["session_scan_dirs"] = [os.path.expanduser(d) for d in config["session_scan_dirs"]]

    return config


def ensure_dirs(config: dict) -> None:
    """Create necessary directories for the config."""
    db_dir = os.path.dirname(config["db_path"])
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    state_dir = os.path.dirname(config.get("state_file", ""))
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)


def default_config_yaml() -> str:
    """Return the default config as a YAML string for 'ultramemory init'."""
    return """# Ultramemory configuration
# See: https://github.com/openclaw/ultramemory

# Database path (SQLite)
# db_path: ~/.ultramemory/memory.db

# LLM model (any litellm-compatible model string)
# model: anthropic/claude-haiku-4-5

# Embedding provider: "local" (sentence-transformers, free, no API key) or "litellm" (API-based)
# embedding_provider: local

# Embedding model
# For local: any sentence-transformers model (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2)
# For litellm: any litellm embedding model (e.g. text-embedding-3-small, cohere/embed-english-v3.0, voyage/voyage-3)
# Gemini Embedding 2 (Preview): gemini/gemini-embedding-2 (768 dim, $0.20/1M tokens)
#   Requires GOOGLE_API_KEY env var (or GEMINI_API_KEY)
# embedding_model: all-MiniLM-L6-v2

# Embedding dimensions (must match the model)
# Local all-MiniLM-L6-v2: 384, all-mpnet-base-v2: 768
# OpenAI text-embedding-3-small: 1536, text-embedding-3-large: 3072
# Cohere embed-english-v3.0: 1024
# Gemini gemini-embedding-2: 768
# embedding_dim: 384

# API server settings
# api_port: 8642
# api_host: 127.0.0.1  # Use 0.0.0.0 to expose on network (requires api_key)

# API authentication (recommended if exposing on network)
# api_key: null  # Set to require X-API-Key header on all requests

# CORS allowed origins
# cors_origins:
#   - http://localhost:3333
#   - http://127.0.0.1:3333

# Input limits
# max_ingest_bytes: 51200   # 50KB max ingest text
# max_query_length: 1024    # 1KB max search query
# max_top_k: 100            # Max search results

# Logging level: debug, info, warning, error
# log_level: info

# Semantic dedup threshold (0.0-1.0, higher = stricter)
# dedup_threshold: 0.97

# Live ingest interval in seconds
# ingest_interval: 900

# Regex patterns to filter noise during ingestion
# skip_patterns:
#   - "HEARTBEAT_OK"
#   - "NO_REPLY"

# Directories to scan for session JSONL files (for live ingest)
# session_scan_dirs:
#   - ~/.openclaw/agents
"""


# Module-level singleton for convenience
_config: dict | None = None


def get_config() -> dict:
    """Get the global config singleton, loading it on first access."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global config singleton (useful for testing)."""
    global _config
    _config = None
