"""Thin wrapper module to centralize search helpers."""

from src.core import rag_core

vector_search = rag_core.vector_search
search = rag_core.search

__all__ = [
    "vector_search",
    "search",
]
