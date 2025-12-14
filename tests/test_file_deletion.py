"""Tests for file deletion logic."""
import pathlib
import tempfile
import time

from src.core.factory import OllamaBackend

def test_file_deletion_and_reindexing(monkeypatch, require_pgvector):
    """Verify that deleted files are not re-indexed on restart."""
    with tempfile.TemporaryDirectory() as temp_dir:
        indexed_dir = pathlib.Path(temp_dir) / "indexed"
        monkeypatch.setattr("src.core.document_repo.INDEXED_DIR", str(indexed_dir))
        indexed_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup a dummy file and an OllamaBackend
        backend = OllamaBackend()
        backend._check_core()

        # 2. Create and index a dummy file
        dummy_file = pathlib.Path(temp_dir) / "test_file.txt"
        dummy_file.write_text("This is a test file.")
        
        backend.index_path(str(dummy_file))
        
        # 3. Verify the file is indexed
        docs = backend.list_documents()
        assert len(docs) == 1
        # Use resolved path for comparison
        assert docs[0]["uri"] == str(dummy_file.resolve())
        
        # 4. Delete the file
        backend.delete_documents([str(dummy_file.resolve())])

        # Wait for asynchronous deletion worker to complete
        max_wait = 10  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = backend.get_deletion_status()
            if not status.get("processing") and status.get("queue_size") == 0:
                break
            time.sleep(0.2)
        
        # 5. Verify the file is deleted
        docs = backend.list_documents()
        assert len(docs) == 0
        
        # 6. Simulate a restart by creating a new OllamaBackend instance
        backend2 = OllamaBackend()

        # 7. Verify the file is NOT re-indexed
        docs2 = backend2.list_documents()
        assert len(docs2) == 0, "Deleted file was re-indexed after restart!"
