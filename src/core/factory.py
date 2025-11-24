import os
import logging
import pathlib
import psutil
from typing import Dict, Any, Optional, List
import requests

from src.core.interfaces import RAGBackend

# Initialize optional modules to None
local_core = None
_extract_text_from_file = None
DB_PATH = None
GoogleGeminiBackend = None

# Import local implementation
try:
    import src.core.rag_core as local_core
    from src.core.store import DB_PATH
    from src.core.extractors import _extract_text_from_file
    HAS_LOCAL_CORE = True
except ImportError:
    HAS_LOCAL_CORE = False

# Import Google implementation
try:
    from src.core.google_backend import GoogleGeminiBackend
    HAS_GOOGLE_BACKEND = True
except ImportError:
    HAS_GOOGLE_BACKEND = False

logger = logging.getLogger(__name__)

class LocalBackend:
    """Direct calls to rag_core functions."""
    
    def _check_core(self):
        if local_core is None:
            raise RuntimeError("Local core dependencies not available")

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        self._check_core()
        return local_core.search(query, top_k=top_k)

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        self._check_core()
        return local_core.upsert_document(uri, text)

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        self._check_core()
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
                if _extract_text_from_file:
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
        self._check_core()
        return local_core.grounded_answer(question, k=k)

    def load_store(self) -> bool:
        self._check_core()
        local_core.load_store()
        return True

    def save_store(self) -> bool:
        self._check_core()
        local_core.save_store()
        return True

    def list_documents(self) -> List[str]:
        self._check_core()
        store = local_core.get_store()
        return list(store.docs.keys())

    def rebuild_index(self) -> None:
        self._check_core()
        local_core._rebuild_faiss_index()

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._check_core()
        return local_core.rerank(query, passages)

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        self._check_core()
        return local_core.verify_grounding(question, answer, citations)

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        self._check_core()
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
        self._check_core()
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
        self._check_core()
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

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        self._check_core()
        return local_core.chat(messages)

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
        resp = requests.post(f"{self.base_url}/load_store", json={})
        return resp.status_code == 200

    def save_store(self) -> bool:
        return True

    def list_documents(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/documents")
        resp.raise_for_status()
        data = resp.json()
        return [doc["uri"] for doc in data.get("documents", [])]

    def rebuild_index(self) -> None:
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

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/chat", json={"messages": messages})
        resp.raise_for_status()
        return resp.json()

class HybridBackend:
    """
    A backend that wraps multiple implementations and allows switching between them.
    """
    def __init__(self):
        self.backends: Dict[str, RAGBackend] = {}
        self.current_mode = "local"
        
        # Initialize Local
        if HAS_LOCAL_CORE:
            try:
                self.backends["local"] = LocalBackend()
                logger.info("HybridBackend: LocalBackend initialized")
            except Exception as e:
                logger.error(f"HybridBackend: Failed to init LocalBackend: {e}")
        
        # Initialize Google
        if HAS_GOOGLE_BACKEND and GoogleGeminiBackend:
            try:
                self.backends["google"] = GoogleGeminiBackend()
                logger.info("HybridBackend: GoogleGeminiBackend initialized")
            except Exception as e:
                logger.error(f"HybridBackend: Failed to init GoogleGeminiBackend: {e}")
                
        # Default to what's available
        if "google" in self.backends and os.getenv("RAG_MODE", "local").lower() == "google":
            self.current_mode = "google"
        elif "local" in self.backends:
            self.current_mode = "local"
        else:
            logger.warning("HybridBackend: No backends available!")

    def set_mode(self, mode: str) -> bool:
        # Handle top-level switching
        if mode == "local":
            if "local" in self.backends:
                self.current_mode = "local"
                logger.info("HybridBackend: Switched to local")
                return True
            return False
            
        # Handle Google sub-modes
        if "google" in self.backends:
            google_backend = self.backends["google"]
            # Check if the requested mode is supported by Google backend
            if hasattr(google_backend, "get_available_modes"):
                if mode in google_backend.get_available_modes():
                    self.current_mode = "google"
                    google_backend.set_mode(mode)
                    logger.info(f"HybridBackend: Switched to google ({mode})")
                    return True
            
            # Fallback for legacy "google" mode request -> manual
            if mode == "google":
                self.current_mode = "google"
                google_backend.set_mode("manual")
                logger.info("HybridBackend: Switched to google (manual)")
                return True

        return False

    def get_mode(self) -> str:
        if self.current_mode == "google":
            # Return the specific google mode (manual or vertex_ai_search)
            if hasattr(self.backends["google"], "get_mode"):
                return self.backends["google"].get_mode()
        return self.current_mode

    def get_available_modes(self) -> List[str]:
        modes = []
        if "local" in self.backends:
            modes.append("local")
        if "google" in self.backends:
            if hasattr(self.backends["google"], "get_available_modes"):
                modes.extend(self.backends["google"].get_available_modes())
            else:
                modes.append("google")
        return modes

    @property
    def _backend(self) -> RAGBackend:
        if self.current_mode not in self.backends:
            raise RuntimeError(f"Current mode {self.current_mode} not available in backends: {list(self.backends.keys())}")
        return self.backends[self.current_mode]

    # Delegate all methods
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self._backend.search(query, top_k)

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        return self._backend.upsert_document(uri, text)

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        return self._backend.index_path(path, glob)

    def grounded_answer(self, question: str, k: int = 5, model: Optional[str] = None) -> Dict[str, Any]:
        if hasattr(self._backend, "grounded_answer"):
             # Check if backend.grounded_answer accepts model
            import inspect
            sig = inspect.signature(self._backend.grounded_answer)
            if "model" in sig.parameters:
                return self._backend.grounded_answer(question, k, model=model)
            return self._backend.grounded_answer(question, k)
        return {"error": "Grounded answer not supported"}

    def load_store(self) -> bool:
        return self._backend.load_store()

    def save_store(self) -> bool:
        return self._backend.save_store()

    def list_documents(self) -> List[str]:
        return self._backend.list_documents()

    def rebuild_index(self) -> None:
        self._backend.rebuild_index()

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._backend.rerank(query, passages)

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        return self._backend.verify_grounding(question, answer, citations)

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        return self._backend.delete_documents(uris)

    def flush_cache(self) -> Dict[str, Any]:
        return self._backend.flush_cache()

    def get_stats(self) -> Dict[str, Any]:
        stats = self._backend.get_stats()
        stats["mode"] = self.current_mode
        stats["available_modes"] = self.get_available_modes()
        return stats

    def list_models(self) -> List[str]:
        if hasattr(self._backend, "list_models"):
            return self._backend.list_models()
        return []

    def chat(self, messages: List[Dict[str, str]], model: str = None) -> Dict[str, Any]:
        if hasattr(self._backend, "chat"):
            # Check if backend.chat accepts model
            import inspect
            sig = inspect.signature(self._backend.chat)
            if "model" in sig.parameters:
                return self._backend.chat(messages, model=model)
            return self._backend.chat(messages)
        return {"error": "Chat not supported by current backend"}

    def list_drive_files(self, folder_id: str = None) -> List[Dict[str, Any]]:
        if hasattr(self._backend, "list_drive_files"):
            return self._backend.list_drive_files(folder_id)
        return []

    def upload_file(self, name: str, content: bytes, mime_type: str) -> Dict[str, Any]:
        if hasattr(self._backend, "upload_file"):
            return self._backend.upload_file(name, content, mime_type)
        return {"error": "Upload not supported by current backend"}

    def reload_auth(self) -> None:
        """Reload authentication for the current backend if supported."""
        if hasattr(self._backend, "reload_auth"):
            self._backend.reload_auth()

def get_rag_backend() -> RAGBackend:
    """Factory to get the configured backend."""
    mode = os.getenv("RAG_MODE", "local").lower()
    
    if mode == "remote":
        url = os.getenv("RAG_REMOTE_URL", "http://127.0.0.1:8001/api")
        logger.info(f"Initializing RemoteBackend at {url}")
        return RemoteBackend(url)
    
    logger.info(f"Initializing HybridBackend (initial mode: {mode})")
    return HybridBackend()
