"""
Supermemory - Local-first AI memory engine with relational versioning and temporal grounding.

This package provides a memory engine for AI agents with temporal versioning,
semantic search, and relationship tracking.
"""

from .engine import MemoryEngine
from .config import load_config, get_config

__version__ = "0.1.0"
__all__ = ["MemoryEngine", "load_config", "get_config"]