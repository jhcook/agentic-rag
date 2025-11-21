import numpy as np
import pytest

from src.core.rag_core import Store, get_faiss_globals, reset_store, upsert_document


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Reset the global store/FAISS metadata between tests."""
    db_file = tmp_path / "faiss-store.jsonl"
    reset_store(Store(), db_path=str(db_file))
    yield
    if db_file.exists():
        db_file.unlink()


def test_faiss_indexing():
    index, index_to_meta, embed_dim = get_faiss_globals()
    upsert_document("faiss_test.txt", "This is a test for FAISS indexing.")
    assert index is not None
    assert index.ntotal > 0
    assert len(index_to_meta) > 0

    found = any("faiss_test.txt" in meta[0] for meta in index_to_meta.values())
    assert found

    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
    assert vectors.shape[1] == embed_dim


def test_faiss_reset():
    index, _, _ = get_faiss_globals()
    if index is None:
        pytest.skip("FAISS not available in this environment")
    index.reset()
    assert index.ntotal == 0
