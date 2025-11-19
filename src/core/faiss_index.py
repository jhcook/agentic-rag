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

_INDEX: Optional[Any] = None
_INDEX_TO_META: Optional[Dict[int, Tuple[str, str]]] = None
_EMBED_DIM: Optional[int] = None
_REBUILD_LOCK = threading.Lock()


def get_rebuild_lock() -> threading.Lock:
    return _REBUILD_LOCK


def get_faiss_globals(embed_dim: Optional[int], debug_mode: bool, logger: logging.Logger) -> Tuple[Any, Dict[int, Tuple[str, str]], int]:
    """
    Initialize and return FAISS globals. embed_dim is expected from the embedder; falls back to 384.
    """
    global _INDEX, _INDEX_TO_META, _EMBED_DIM

    if _INDEX is None:
        if debug_mode:
            logger.info("Debug mode enabled - skipping FAISS and embedding initialization")
            _EMBED_DIM = 384
            _INDEX = None
            _INDEX_TO_META = {}
        else:
            _EMBED_DIM = embed_dim if embed_dim is not None else 384
            if faiss is not None and _EMBED_DIM is not None:
                _INDEX = faiss.IndexFlatIP(_EMBED_DIM)  # type: ignore
                _INDEX_TO_META = {}
            else:
                _INDEX = None  # type: ignore
                _INDEX_TO_META = {}

        logger.debug("Initialized FAISS globals with embedding dimension %s", _EMBED_DIM)

    # Ensure we never return None values to callers
    if _INDEX_TO_META is None:
        _INDEX_TO_META = {}
    if _EMBED_DIM is None:
        _EMBED_DIM = 384

    return _INDEX, _INDEX_TO_META, _EMBED_DIM
