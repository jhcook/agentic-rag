"""
Thin wrapper module to centralize store-related helpers.
"""
from src.core import rag_core

Store = rag_core.Store
get_store = rag_core.get_store
save_store = rag_core.save_store
load_store = rag_core.load_store
resolve_input_path = rag_core.resolve_input_path
DB_PATH = rag_core.DB_PATH
_should_skip_uri = rag_core._should_skip_uri  # pylint: disable=protected-access

__all__ = [
    "Store",
    "get_store",
    "save_store",
    "load_store",
    "resolve_input_path",
    "DB_PATH",
    "_should_skip_uri",
]
