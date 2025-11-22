import os
import logging
import pathlib
import psutil
from typing import Dict, Any, Optional, List
import requests

from src.core.interfaces import RAGBackend
# Import local implementation (lazy import to avoid overhead if not used?)
# For now, we import at top level, but in a real services split, we might not have the dependencies.
try:
    import src.core.rag_core as local_core
    from src.core.store import DB_PATH
    from src.core.extractors import _extract_text_from_file
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
        try:
            base = local_core.resolve_input_path(path)
        except FileNotFoundError as exc:
            return {"error": str(exc), "indexed": 0, "uris": []}

        # Normalize glob
        if not glob or not glob.strip():
            glob = "**/*"
        
        if base.is_file():
            files = [base]
        else:
            files = list(base.rglob(glob))
            
        indexed = 0
        indexed_uris = []
        for file_path in files:
            try:
                content = _extract_text_from_file(file_path)
                if not content:
                    continue
                local_core.upsert_document(str(file_path), content)
                indexed += 1
                indexed_uris.append(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")
                
        return {"indexed": indexed, "uris": indexed_uris}

    def grounded_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        return local_core.grounded_answer(question, k=k)

    def load_store(self) -> bool:
        local_core.load_store()
        return True

    def save_store(self) -> bool:
        local_core.save_store()
        return True

    def list_documents(self) -> List[str]:
        store = local_core.get_store()
        return list(store.docs.keys())

    def rebuild_index(self) -> None:
        local_core._rebuild_faiss_index()

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return local_core.rerank(query, passages)

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        return local_core.verify_grounding(question, answer, citations)

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        store = local_core.get_store()
        deleted = 0
        for uri in uris:
            if uri in store.docs:
                del store.docs[uri]
                deleted += 1
        local_core.save_store()
        local_core._rebuild_faiss_index()
        return {"deleted": deleted}

    def flush_cache(self) -> Dict[str, Any]:
        store = local_core.get_store()
        store.docs.clear()
        removed = False
        if DB_PATH and pathlib.Path(DB_PATH).exists():
            try:
                pathlib.Path(DB_PATH).unlink()
                removed = True
            except OSError:
                removed = False
        local_core.save_store()
        local_core._rebuild_faiss_index()
        return {"status": "flushed", "db_removed": removed, "documents": 0}

    def get_stats(self) -> Dict[str, Any]:
        store = local_core.get_store()
        index, _, _ = local_core.get_faiss_globals()
        docs = len(getattr(store, "docs", {}))
        vectors = index.ntotal if index is not None else 0
        
        return {
            "status": "ok",
            "documents": docs,
            "vectors": vectors,
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "memory_limit_mb": local_core.MAX_MEMORY_MB,
        }

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

    def save_store(self) -> bool:
        # Remote save might not be exposed or needed if server handles it
        # But for interface compliance:
        # Assuming there is no explicit save endpoint, or we use flush_cache?
        # Let's assume no-op or log warning for now, or add endpoint if needed.
        # Actually, mcp_server calls save_store on shutdown.
        # If we are a client, we probably don't need to tell the server to save on OUR shutdown.
        return True

    def list_documents(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/documents")
        resp.raise_for_status()
        data = resp.json()
        return [doc["uri"] for doc in data.get("documents", [])]

    def rebuild_index(self) -> None:
        # Triggering rebuild remotely?
        # Maybe /flush_cache triggers it?
        # Or we just assume server manages it.
        pass

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        resp = requests.post(f"{self.base_url}/rerank", json={"query": query, "passages": passages})
        resp.raise_for_status()
        return resp.json().get("results", [])

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/verify_grounding", json={"question": question, "draft_answer": answer, "citations": citations})
        resp.raise_for_status()
        return resp.json()

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/documents/delete", json={"uris": uris})
        resp.raise_for_status()
        return resp.json()

    def flush_cache(self) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/flush_cache", json={})
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

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
