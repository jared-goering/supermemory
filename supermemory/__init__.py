"""
Supermemory - Local-first AI memory engine with relational versioning and temporal grounding.

This package provides a memory engine for AI agents with temporal versioning,
semantic search, and relationship tracking.
"""

from .config import get_config, load_config
from .engine import MemoryEngine

__version__ = "0.2.1"
__all__ = ["MemoryEngine", "load_config", "get_config"]
