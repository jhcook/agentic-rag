import os
import logging
from typing import Dict, Any, Optional
import requests

from src.core.interfaces import RAGBackend
# Import local implementation (lazy import to avoid overhead if not used?)
# For now, we import at top level, but in a real services split, we might not have the dependencies.
try:
    import src.core.rag_core as local_core
    HAS_LOCAL_CORE = True
except ImportError:
    HAS_LOCAL_CORE = False

logger = logging.getLogger(__name__)

class LocalBackend:
    """Direct calls to rag_core functions."""
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return local_core.search(query, top_k=top_k)

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        return local_core.upsert_document(uri, text)

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        # rag_core doesn't have a direct index_path function exposed in the same way, 
        # it usually relies on the caller to walk files. 
        # But let's assume we wrap the logic or add it to rag_core.
        # For now, we'll implement the walk logic here or call a helper.
        # Looking at mcp_server.py, it implements index_documents_tool by walking.
        # We should probably move that logic to rag_core to make it shared.
        # For this POC, I'll leave it as a placeholder or call a helper if it exists.
        from src.utils.simple_indexer import index_directory
        return index_directory(path, glob) # This might need adaptation

    def grounded_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        return local_core.grounded_answer(question, k=k)

    def load_store(self) -> bool:
        local_core.load_store()
        return True

class RemoteBackend:
    """HTTP calls to the REST API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/search", json={"query": query, "k": top_k})
        resp.raise_for_status()
        return resp.json()

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/upsert_document", json={"uri": uri, "text": text})
        resp.raise_for_status()
        return resp.json()

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/index_path", json={"path": path, "glob": glob})
        resp.raise_for_status()
        return resp.json()

    def grounded_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/grounded_answer", json={"question": question, "k": k})
        resp.raise_for_status()
        return resp.json()

    def load_store(self) -> bool:
        # Remote store management might be different
        resp = requests.post(f"{self.base_url}/load_store", json={})
        return resp.status_code == 200

def get_rag_backend() -> RAGBackend:
    """Factory to get the configured backend."""
    mode = os.getenv("RAG_MODE", "local").lower()
    
    if mode == "remote":
        url = os.getenv("RAG_REMOTE_URL", "http://127.0.0.1:8001/api")
        logger.info(f"Initializing RemoteBackend at {url}")
        return RemoteBackend(url)
    
    if not HAS_LOCAL_CORE:
        raise ImportError("RAG_MODE=local but src.core.rag_core could not be imported.")
        
    logger.info("Initializing LocalBackend")
    return LocalBackend()
