"""Tests for pgvector indexing functionality.

These tests require a running PostgreSQL+pgvector instance (via Docker Compose)
and a configured password (PGVECTOR_PASSWORD or secrets/pgvector_config.json).
"""

import pytest

from src.core import pgvector_store
from src.core import rag_core


def _skip_if_pgvector_unavailable() -> None:
    ok, msg = pgvector_store.test_connection()
    if not ok:
        pytest.skip(f"pgvector unavailable: {msg}")


def test_pgvector_indexing_and_search() -> None:
    """Smoke test: upsert a doc and retrieve it via vector search."""

    _skip_if_pgvector_unavailable()
    rag_core.ensure_vector_store_ready()

    uri = "pgvector_test.txt"
    rag_core.upsert_document(uri, "This is a test for pgvector indexing.")

    hits = rag_core.vector_search("pgvector indexing", k=5)
    assert any(h.get("uri") == uri for h in hits)


def test_pgvector_delete_documents() -> None:
    """Deleting a doc removes its chunks from pgvector."""

    _skip_if_pgvector_unavailable()
    rag_core.ensure_vector_store_ready()

    uri = "pgvector_delete_test.txt"
    rag_core.upsert_document(uri, "Delete me.")
    pgvector_store.delete_documents([uri])

    hits = rag_core.vector_search("Delete me", k=5)
    assert all(h.get("uri") != uri for h in hits)
