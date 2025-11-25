"""
Thin wrapper module to centralize search helpers.
"""
# pylint: disable=protected-access
from src.core import rag_core

_vector_search = rag_core._vector_search
search = rag_core.search

__all__ = [
    "_vector_search",
    "search",
]
