"""Test deletion queue functionality."""
import time
from src.core.factory import OllamaBackend


def test_deletion_queue(require_pgvector):
    """Test that deletions are queued and processed atomically."""
    backend = OllamaBackend()
    
    # Clear any existing documents to ensure clean test state
    backend.flush_cache()
    
    # Add some test documents
    backend.upsert_document("test1.txt", "Content 1")
    backend.upsert_document("test2.txt", "Content 2")
    backend.upsert_document("test3.txt", "Content 3")
    
    # Get initial count
    docs = backend.list_documents()
    assert len(docs) == 3
    
    # Queue multiple deletions
    result1 = backend.delete_documents(["test1.txt"])
    assert result1["status"] == "queued"
    assert result1["queue_size"] >= 1
    
    result2 = backend.delete_documents(["test2.txt", "test3.txt"])
    assert result2["status"] == "queued"
    
    # Check status
    status = backend.get_deletion_status()
    assert status["queue_size"] >= 0
    
    # Wait for processing to complete
    max_wait = 10  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = backend.get_deletion_status()
        if not status["processing"] and status["queue_size"] == 0:
            break
        time.sleep(0.5)
    
    # Verify all were processed
    assert status["total_processed"] == 3
    assert not status["processing"]
    assert status["queue_size"] == 0
    
    # Verify documents are deleted
    final_docs = backend.list_documents()
    assert len(final_docs) == 0


def test_deletion_status_empty(require_pgvector):
    """Test deletion status when queue is empty."""
    backend = OllamaBackend()
    status = backend.get_deletion_status()
    
    assert status["queue_size"] == 0
    assert status["processing"] is False
    assert status["total_processed"] >= 0


if __name__ == "__main__":
    test_deletion_queue()
    test_deletion_status_empty()
    print("✓ All deletion queue tests passed!")
