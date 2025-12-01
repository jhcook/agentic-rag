"""Tests for file deletion logic."""
import os
import pathlib
import tempfile

from src.core.factory import OllamaBackend

def test_file_deletion_and_reindexing():
    """Verify that deleted files are not re-indexed on restart."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a temporary file for the database
        db_path = pathlib.Path(temp_dir) / "rag_store.jsonl"
        
        # We need to monkeypatch the DB_PATH in the local_core module
        from src.core import rag_core
        rag_core.DB_PATH = str(db_path)
        # Reset the global store to ensure clean state
        rag_core._STORE = None
        
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
        
        # 5. Verify the file is deleted
        docs = backend.list_documents()
        assert len(docs) == 0
        
        # 6. Simulate a restart by creating a new OllamaBackend instance
        backend2 = OllamaBackend()
        from src.core import rag_core as rag_core2
        rag_core2.DB_PATH = str(db_path)
        backend2.load_store() # This will load from the persisted file

        # 7. Verify the file is NOT re-indexed
        docs2 = backend2.list_documents()
        assert len(docs2) == 0, "Deleted file was re-indexed after restart!"
