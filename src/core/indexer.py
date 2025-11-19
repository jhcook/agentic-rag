"""
Thin wrapper module to centralize indexing helpers.
"""
from src.core import rag_core

upsert_document = rag_core.upsert_document
index_path = rag_core.index_path
_rebuild_faiss_index = rag_core._rebuild_faiss_index
_collect_files = rag_core._collect_files
_process_file = rag_core._process_file

__all__ = [
    "upsert_document",
    "index_path",
    "_rebuild_faiss_index",
    "_collect_files",
    "_process_file",
]
