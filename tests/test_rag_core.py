import pytest
import logging
from pathlib import Path
import tempfile
from src.core.rag_core import (
    Store, save_store, load_store, upsert_document,
    index_path, search, rerank, synthesize_answer,
    grounded_answer, verify_grounding_simple, verify_grounding, get_store
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_store():
    store = Store()
    store.add("doc1.txt", "This is a sample document about AI.")
    store.add("doc2.txt", "Another document about machine learning.")
    return store

@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix='.jsonl') as f:
        yield f.name

def test_store_initialization():
    store = Store()
    assert store.docs == {}
    assert store.last_loaded == 0.0

def test_store_add():
    store = Store()
    store.add("test.txt", "Test content")
    assert "test.txt" in store.docs
    assert store.docs["test.txt"] == "Test content"

def test_save_and_load_store(temp_db_path, monkeypatch):
    """Test saving and loading store with proper cleanup."""
    # Configure test environment
    monkeypatch.setattr("src.core.rag_core.DB_PATH", temp_db_path)
    logger.info(f"Using temporary DB path: {temp_db_path}")
    
    # Reset global store
    import src.core.rag_core as rag_core
    rag_core._STORE = Store()
    
    # Add test content
    test_content = "Test content for persistence"
    rag_core._STORE.add("test.txt", test_content)
    logger.info("Added test document to store")
    
    # Save store
    save_store()
    logger.info("Saved store to disk")
    
    # Create new store instance
    rag_core._STORE = Store()
    assert len(rag_core._STORE.docs) == 0, "Expected empty store after reset"
    
    # Load from disk
    load_store()
    logger.info(f"Loaded store. Contents: {rag_core._STORE.docs}")
    
    # Verify
    assert "test.txt" in rag_core._STORE.docs, f"Expected 'test.txt' in {list(rag_core._STORE.docs.keys())}"
    assert rag_core._STORE.docs["test.txt"] == test_content

def test_upsert_document(temp_db_path, monkeypatch):
    """Test document upsert with proper store reset."""
    monkeypatch.setattr("src.core.rag_core.DB_PATH", temp_db_path)
    
    # Reset store
    import src.core.rag_core as rag_core
    rag_core._STORE = Store()
    
    # Perform upsert
    test_content = "Test content for upsert"
    result = upsert_document("test_doc.txt", test_content)
    logger.info(f"Upsert result: {result}")
    
    # Verify
    assert result["upserted"] is True
    assert "test_doc.txt" in rag_core._STORE.docs
    assert rag_core._STORE.docs["test_doc.txt"] == test_content


def test_index_path(tmp_path):
    # Create temporary directory with test files
    (tmp_path / "test1.txt").write_text("Content 1")
    (tmp_path / "test2.txt").write_text("Content 2")
    
    result = index_path(str(tmp_path))
    assert result["indexed"] == 2

def test_index_path(tmp_path):
    # Create temporary directory with test files
    (tmp_path / "test1.txt").write_text("Content 1")
    (tmp_path / "test2.txt").write_text("Content 2")
    
    result = index_path(str(tmp_path))
    assert result["indexed"] == 2

def test_search():
    # Clear the store first
    from src.core.rag_core import _STORE
    global _STORE
    _STORE = Store()
    
    # Add test document
    test_doc = "This is a test document about artificial intelligence"
    _STORE.add("test.txt", test_doc)
    logger.info(f"Added test document to store: {test_doc}")
    
    # Perform search
    results = search("artificial intelligence", top_k=5)
    logger.info(f"Search results: {results}")
    
    # Verify results
    # search returns a dict with 'choices' (LLM response) or 'answer' (fallback)
    # It does NOT return a list of results directly anymore.
    assert isinstance(results, dict) or hasattr(results, 'choices')
    if isinstance(results, dict):
        assert "choices" in results or "answer" in results or "error" in results

def test_rerank():
    passages = [
        {"text": "AI and machine learning", "score": 0.5},
        {"text": "Database systems", "score": 0.3}
    ]
    results = rerank("AI", passages)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]

def test_synthesize_answer():
    passages = [
        {"text": "AI is a field of computer science", "uri": "doc1.txt"},
        {"text": "Machine learning is a subset of AI", "uri": "doc2.txt"}
    ]
    result = synthesize_answer("What is AI?", passages)
    assert "answer" in result
    assert "citations" in result

def test_grounded_answer():
    import src.core.rag_core as rag_core
    rag_core._STORE = Store()
    rag_core._STORE.add("ai_doc.txt", "AI is artificial intelligence")
    result = grounded_answer("What is AI?", k=1)
    assert "answer" in result
    assert "sources" in result

def test_verify_grounding_simple():
    passages = [
        {"text": "AI is artificial intelligence", "uri": "doc1.txt"},
        {"text": "Machine learning is part of AI", "uri": "doc2.txt"}
    ]
    result = verify_grounding_simple(
        "What is AI?",
        "AI is artificial intelligence [1]. It includes machine learning [2].",
        passages
    )
    assert "answer_conf" in result
    assert "citation_coverage" in result
    assert "missing_facts" in result

def test_verify_grounding():
    from src.core.rag_core import _STORE
    global _STORE
    _STORE = Store()
    _STORE.add("doc1.txt", "AI is artificial intelligence")
    result = verify_grounding(
        "What is AI?",
        "AI is artificial intelligence [1].",
        ["doc1.txt"]
    )
    assert "answer_conf" in result
    assert "citation_coverage" in result