"""
FAISS index lifecycle helpers and shared globals.
"""

from __future__ import annotations
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

try:
    import faiss  # type: ignore
except ImportError:
    if os.getenv("RAG_DEBUG_MODE", "false").lower() != "true":
        raise
    faiss = None

_index: Optional[Any] = None
_index_to_meta: Optional[Dict[int, Tuple[str, str]]] = None
_embed_dim: Optional[int] = None
_REBUILD_LOCK = threading.Lock()


def get_rebuild_lock() -> threading.Lock:
    """Return the global rebuild lock."""
    return _REBUILD_LOCK


def get_faiss_globals(
    embed_dim: Optional[int], debug_mode: bool, logger: logging.Logger
) -> Tuple[Any, Dict[int, Tuple[str, str]], int]:
    """
    Initialize and return FAISS globals. embed_dim is expected from the embedder; falls back to 384.
    """
    # pylint: disable=global-statement
    global _index, _index_to_meta, _embed_dim

    if _index is None:
        if debug_mode:
            logger.info("Debug mode enabled - skipping FAISS and embedding initialization")
            _embed_dim = 384
            _index = None
            _index_to_meta = {}
        else:
            _embed_dim = embed_dim if embed_dim is not None else 384
            if faiss is not None and _embed_dim is not None:
                _index = faiss.IndexFlatIP(_embed_dim)  # type: ignore
                _index_to_meta = {}
            else:
                _index = None  # type: ignore
                _index_to_meta = {}

        logger.debug("Initialized FAISS globals with embedding dimension %s", _embed_dim)

    # Ensure we never return None values to callers
    if _index_to_meta is None:
        _index_to_meta = {}
    if _embed_dim is None:
        _embed_dim = 384

    return _index, _index_to_meta, _embed_dim
