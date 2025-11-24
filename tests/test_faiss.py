"""Tests for FAISS indexing functionality."""
import numpy as np

import pytest

from src.core.rag_core import get_faiss_globals, upsert_document


def test_faiss_indexing():
    """Test FAISS indexing functionality."""
    index, index_to_meta, _ = get_faiss_globals()
    upsert_document("faiss_test.txt", "This is a test for FAISS indexing.")
    assert index.ntotal > 0
    assert len(index_to_meta) > 0

    # Check that the indexed text is retrievable
    found = any("faiss_test.txt" in meta[0] for meta in index_to_meta.values())
    assert found

    # Check vector shape
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])  # pylint: disable=no-value-for-parameter
    assert vectors.shape[1] == 384  # Assuming 384D embedding


def test_faiss_reset():
    """Test FAISS index reset."""
    index, _, _ = get_faiss_globals()
    index.reset()
    assert index.ntotal == 0

