import pytest
import numpy as np
from rag_core import get_faiss_globals, upsert_document, get_store

def test_faiss_indexing():
    index, index_to_meta, _ = get_faiss_globals()
    upsert_document("faiss_test.txt", "This is a test for FAISS indexing.")
    assert index.ntotal > 0
    assert len(index_to_meta) > 0

    # Check that the indexed text is retrievable
    found = any("faiss_test.txt" in meta[0] for meta in index_to_meta.values())
    assert found

    # Check vector shape
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
    assert vectors.shape[1] == 384  # Assuming 384D embedding


def test_faiss_reset():
    index, _, _ = get_faiss_globals()
    index.reset()
    assert index.ntotal == 0
