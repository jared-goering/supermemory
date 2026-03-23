"""
Supermemory configuration system.

Loads config from (highest priority wins):
1. Environment variables (SUPERMEMORY_DB_PATH, SUPERMEMORY_MODEL, etc.)
2. ./supermemory.yaml (project-local)
3. ~/.supermemory/config.yaml (user-global)
4. Built-in defaults
"""

import os
from pathlib import Path
from typing import Any

# Try to import yaml, but don't fail if not installed (defaults still work)
try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG = {
    "db_path": os.path.join(str(Path.home()), ".supermemory", "memory.db"),
    "model": "anthropic/claude-haiku-4-5",
    "embedding_provider": "local",  # "local" (sentence-transformers) or "litellm" (any litellm-supported provider)
    "embedding_model": "all-MiniLM-L6-v2",  # local: model name; litellm: e.g. "text-embedding-3-small", "cohere/embed-english-v3.0"
    "embedding_dim": 384,  # must match the model (e.g. 1536 for text-embedding-3-small)
    "api_port": 8642,
    "api_host": "0.0.0.0",
    "log_level": "info",
    "dedup_threshold": 0.97,
    "ingest_interval": 900,
    "skip_patterns": [],
    "session_scan_dirs": [],
    "state_file": os.path.join(str(Path.home()), ".supermemory", "ingest-state.json"),
}

# Map of env var names → config keys
ENV_MAP = {
    "SUPERMEMORY_DB_PATH": "db_path",
    "SUPERMEMORY_MODEL": "model",
    "SUPERMEMORY_EMBEDDING_PROVIDER": "embedding_provider",
    "SUPERMEMORY_EMBEDDING_MODEL": "embedding_model",
    "SUPERMEMORY_EMBEDDING_DIM": ("embedding_dim", int),
    "SUPERMEMORY_API_PORT": ("api_port", int),
    "SUPERMEMORY_API_HOST": "api_host",
    "SUPERMEMORY_LOG_LEVEL": "log_level",
    "SUPERMEMORY_DEDUP_THRESHOLD": ("dedup_threshold", float),
    "SUPERMEMORY_INGEST_INTERVAL": ("ingest_interval", int),
    # Legacy env var support
    "MEMORY_DB": "db_path",
}


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning empty dict on failure."""
    if yaml is None:
        return {}
    try:
        with open(path, "r") as f:
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
    skip = os.environ.get("SUPERMEMORY_SKIP_PATTERNS")
    if skip:
        result["skip_patterns"] = [p.strip() for p in skip.split(",") if p.strip()]

    scan_dirs = os.environ.get("SUPERMEMORY_SESSION_SCAN_DIRS")
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
    global_path = os.path.join(str(Path.home()), ".supermemory", "config.yaml")
    config.update(_load_yaml(global_path))

    # Layer 2: Project-local config
    local_path = os.path.join(os.getcwd(), "supermemory.yaml")
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
        config["session_scan_dirs"] = [
            os.path.expanduser(d) for d in config["session_scan_dirs"]
        ]

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
    """Return the default config as a YAML string for 'supermemory init'."""
    return """# Supermemory configuration
# See: https://github.com/openclaw/supermemory

# Database path (SQLite)
# db_path: ~/.supermemory/memory.db

# LLM model (any litellm-compatible model string)
# model: anthropic/claude-haiku-4-5

# Embedding provider: "local" (sentence-transformers, free, no API key) or "litellm" (API-based)
# embedding_provider: local

# Embedding model
# For local: any sentence-transformers model (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2)
# For litellm: any litellm embedding model (e.g. text-embedding-3-small, cohere/embed-english-v3.0, voyage/voyage-3)
# embedding_model: all-MiniLM-L6-v2

# Embedding dimensions (must match the model)
# Local all-MiniLM-L6-v2: 384, all-mpnet-base-v2: 768
# OpenAI text-embedding-3-small: 1536, text-embedding-3-large: 3072
# Cohere embed-english-v3.0: 1024
# embedding_dim: 384

# API server settings
# api_port: 8642
# api_host: 0.0.0.0

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
