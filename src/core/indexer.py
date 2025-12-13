"""Thin wrapper module to centralize indexing helpers."""

from src.core import rag_core

upsert_document = rag_core.upsert_document
index_path = rag_core.index_path
rebuild_index = rag_core.rebuild_index
collect_files = rag_core.collect_files
process_file = rag_core.process_file

__all__ = [
    "upsert_document",
    "index_path",
    "rebuild_index",
    "collect_files",
    "process_file",
]
