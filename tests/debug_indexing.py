
import sys
import os
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.DEBUG)

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import rag_core
from src.core import pgvector_store

def test_startup_indexing():
    print("--- Testing Startup Indexing ---")
    
    # 1. Check Store Path
    db_path = rag_core.DB_PATH
    print(f"DB_PATH: {db_path}")
    
    if os.path.exists(db_path):
        print(f"DB file exists. Size: {os.path.getsize(db_path)} bytes")
    else:
        print("DB file does NOT exist.")
        # Create dummy store
        rag_core.upsert_document("test_doc", "This is a test document.")
        print("Created dummy document.")

    # 2. Load Store
    print("Loading store...")
    rag_core.load_store()
    
    store = rag_core.get_store()
    print(f"Store loaded. Docs: {len(store.docs)}")
    
    if len(store.docs) == 0:
        print("ERROR: Store is empty after load!")
        return

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

    if int(after.get("chunks", 0) or 0) <= 0:
        print("ERROR: pgvector index is empty after rebuild!")
    else:
        print("SUCCESS: pgvector index populated.")

if __name__ == "__main__":
    test_startup_indexing()

