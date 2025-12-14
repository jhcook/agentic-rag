
import sys
import os
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.DEBUG)

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import rag_core
from src.core import pgvector_store
from src.core import document_repo

def test_startup_indexing():
    print("--- Testing Startup Indexing ---")
    
    # 1. Check indexed artifacts directory
    indexed_dir = document_repo.INDEXED_DIR
    print(f"INDEXED_DIR: {indexed_dir}")

    if os.path.isdir(indexed_dir):
        file_count = len([p for p in os.listdir(indexed_dir) if not p.startswith('.')])
        print(f"Indexed dir exists. Files: {file_count}")
    else:
        print("Indexed dir does NOT exist. Creating dummy document...")
        rag_core.upsert_document("test_doc", "This is a test document.")
        print("Created dummy document + artifact.")

    # 3. Ensure vector store and check stats before rebuild
    try:
        rag_core.ensure_vector_store_ready()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"ERROR: pgvector not available: {exc}")
        return

    before = pgvector_store.stats(embedding_model=rag_core.EMBED_MODEL_NAME)
    print(f"pgvector chunks before rebuild: {before.get('chunks')}")

    # 4. Rebuild Index
    print("Rebuilding index...")
    rag_core.rebuild_index()

    # 5. Check stats after rebuild
    after = pgvector_store.stats(embedding_model=rag_core.EMBED_MODEL_NAME)
    print(f"pgvector chunks after rebuild: {after.get('chunks')}")

    docs = rag_core.list_documents()
    print(f"Documents listed: {len(docs)}")

    if int(after.get("chunks", 0) or 0) <= 0:
        print("ERROR: pgvector index is empty after rebuild!")
    else:
        print("SUCCESS: pgvector index populated.")

if __name__ == "__main__":
    test_startup_indexing()

