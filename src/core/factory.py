"""Factory for creating RAG backend instances."""
import logging
import os
import pathlib
import inspect
import json
import threading
import queue
import time
from typing import Dict, Any, List

import psutil
import requests

from src.core.interfaces import RAGBackend

# Initialize optional modules to None
local_core = None  # pylint: disable=invalid-name
_extract_text_from_file = None  # pylint: disable=invalid-name
DB_PATH = None
GoogleGeminiBackend = None  # pylint: disable=invalid-name
OpenAIAssistantsBackend = None  # pylint: disable=invalid-name

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

# Import OpenAI Assistants implementation
try:
    from src.core.openai_assistants_backend import OpenAIAssistantsBackend
    HAS_OPENAI_ASSISTANTS = True
except ImportError:
    HAS_OPENAI_ASSISTANTS = False

logger = logging.getLogger(__name__)

class LocalBackend:
    """Direct calls to rag_core functions."""

    def __init__(self):
        """Initialize backend with deletion queue."""
        self._deletion_queue: queue.Queue = queue.Queue()
        self._deletion_worker_started = False
        self._deletion_lock = threading.Lock()
        self._deletion_status: Dict[str, Any] = {
            "queue_size": 0,
            "processing": False,
            "last_completed": None,
            "total_processed": 0
        }

    def _check_core(self):
        """Check if local core is available."""
        if local_core is None:
            raise RuntimeError("Local core dependencies not available")

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for documents."""
        self._check_core()
        return local_core.search(query, top_k=top_k)

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        """Add or update a document."""
        self._check_core()
        return local_core.upsert_document(uri, text)

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a directory path."""
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
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to index %s: %s", file_path, exc)

        return {"indexed": indexed, "uris": indexed_uris}

    def grounded_answer(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Generate a grounded answer."""
        self._check_core()
        return local_core.grounded_answer(question, k=k, **kwargs)

    def load_store(self) -> bool:
        """Load the document store."""
        self._check_core()
        local_core.load_store()
        return True

    def save_store(self) -> bool:
        """Save the document store."""
        self._check_core()
        local_core.save_store()
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with metadata."""
        self._check_core()
        # Ensure store is synced with disk before listing
        local_core._ensure_store_synced()  # pylint: disable=protected-access
        store = local_core.get_store()
        return [{"uri": uri, "size": len(text)} for uri, text in store.docs.items()]

    def rebuild_index(self) -> None:
        """Rebuild the vector index."""
        self._check_core()
        local_core._rebuild_faiss_index()  # pylint: disable=protected-access

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank passages by relevance."""
        self._check_core()
        return local_core.rerank(query, passages)

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        """Verify answer grounding."""
        self._check_core()
        return local_core.verify_grounding(question, answer, citations)

    def _start_deletion_worker(self):
        """Start the background deletion worker thread."""
        if self._deletion_worker_started:
            return

        def _worker():
            while True:
                try:
                    uris = self._deletion_queue.get(timeout=1)
                    if not uris:
                        continue

                    with self._deletion_lock:
                        self._deletion_status["processing"] = True

                    # Perform deletion
                    store = local_core.get_store()
                    deleted = 0
                    for uri in uris:
                        if uri in store.docs:
                            del store.docs[uri]
                            deleted += 1

                    local_core.save_store()
                    local_core._rebuild_faiss_index()  # pylint: disable=protected-access

                    with self._deletion_lock:
                        self._deletion_status["processing"] = False
                        self._deletion_status["queue_size"] = self._deletion_queue.qsize()
                        self._deletion_status["last_completed"] = time.time()
                        self._deletion_status["total_processed"] += deleted

                    self._deletion_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("Deletion worker error: %s", exc)
                    with self._deletion_lock:
                        self._deletion_status["processing"] = False
                        self._deletion_status["queue_size"] = self._deletion_queue.qsize()

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()
        self._deletion_worker_started = True
        logger.info("Deletion worker thread started")

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        """Delete documents by URI (queued)."""
        self._check_core()
        self._start_deletion_worker()

        # Add to queue
        self._deletion_queue.put(uris)

        with self._deletion_lock:
            self._deletion_status["queue_size"] = self._deletion_queue.qsize()

        return {
            "status": "queued",
            "uris": uris,
            "queue_size": self._deletion_status["queue_size"]
        }

    def get_deletion_status(self) -> Dict[str, Any]:
        """Get current deletion queue status."""
        with self._deletion_lock:
            return self._deletion_status.copy()

    def flush_cache(self) -> Dict[str, Any]:
        """Clear the document cache."""
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
        local_core._rebuild_faiss_index()  # pylint: disable=protected-access
        return {"status": "flushed", "db_removed": removed, "documents": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        self._check_core()
        store = local_core.get_store()
        index, _, _ = local_core.get_faiss_globals()
        docs = len(getattr(store, "docs", {}))
        vectors = index.ntotal if index is not None else 0

        # Calculate total size of indexed text
        total_size_bytes = sum(len(text.encode('utf-8')) for text in store.docs.values())

        # Get store file size
        store_file_bytes = 0
        if local_core.DB_PATH and os.path.exists(local_core.DB_PATH):
            try:
                store_file_bytes = os.path.getsize(local_core.DB_PATH)
            except OSError:
                pass

        return {
            "status": "ok",
            "documents": docs,
            "vectors": vectors,
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "memory_limit_mb": local_core.MAX_MEMORY_MB,
            "total_size_bytes": total_size_bytes,
            "store_file_bytes": store_file_bytes,
        }

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Chat with the backend."""
        self._check_core()
        return local_core.chat(messages, **kwargs)

class RemoteBackend:
    """HTTP calls to the REST API."""

    def __init__(self, base_url: str):
        """Initialize remote backend with base URL."""
        self.base_url = base_url.rstrip("/")
        self.timeout = 30  # Default timeout in seconds

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for documents."""
        resp = requests.post(
            f"{self.base_url}/search",
            json={"query": query, "k": top_k},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        """Add or update a document."""
        resp = requests.post(
            f"{self.base_url}/upsert_document",
            json={"uri": uri, "text": text},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a directory path."""
        resp = requests.post(
            f"{self.base_url}/index_path",
            json={"path": path, "glob": glob},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def grounded_answer(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Generate a grounded answer."""
        payload = {"question": question, "k": k}
        payload.update(kwargs)
        resp = requests.post(
            f"{self.base_url}/grounded_answer",
            json=payload,
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def load_store(self) -> bool:
        """Load the document store."""
        resp = requests.post(
            f"{self.base_url}/load_store",
            json={},
            timeout=self.timeout)
        return resp.status_code == 200

    def save_store(self) -> bool:
        """Save the document store."""
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with metadata."""
        resp = requests.get(f"{self.base_url}/documents", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Map API response (size_bytes) to internal format (size)
        return [{"uri": doc["uri"], "size": doc.get("size_bytes", 0)}
                for doc in data.get("documents", [])]

    def rebuild_index(self) -> None:
        """Rebuild the vector index."""

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank passages by relevance."""
        resp = requests.post(
            f"{self.base_url}/rerank",
            json={"query": query, "passages": passages},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        """Verify answer grounding."""
        resp = requests.post(
            f"{self.base_url}/verify_grounding",
            json={"question": question, "draft_answer": answer,
                  "citations": citations},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        """Delete documents by URI."""
        resp = requests.post(
            f"{self.base_url}/documents/delete",
            json={"uris": uris},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def flush_cache(self) -> Dict[str, Any]:
        """Clear the document cache."""
        resp = requests.post(
            f"{self.base_url}/flush_cache",
            json={},
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Chat with the backend."""
        payload = {"messages": messages}
        payload.update(kwargs)
        resp = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

class HybridBackend:  # pylint: disable=too-many-public-methods
    """
    A backend that wraps multiple implementations and allows switching between them.
    """

    def __init__(self, initial_mode: str = "local"):
        """Initialize hybrid backend with available backends."""
        self.backends: Dict[str, RAGBackend] = {}
        self.current_mode = "local"

        # Initialize Local
        if HAS_LOCAL_CORE:
            try:
                self.backends["local"] = LocalBackend()
                logger.info("HybridBackend: LocalBackend initialized")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("HybridBackend: Failed to init LocalBackend: %s", exc)

        # Initialize Google
        if HAS_GOOGLE_BACKEND and GoogleGeminiBackend:
            try:
                self.backends["google"] = GoogleGeminiBackend()
                logger.info("HybridBackend: GoogleGeminiBackend initialized")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error(
                    "HybridBackend: Failed to init GoogleGeminiBackend: %s", exc)

        # Initialize OpenAI Assistants
        if HAS_OPENAI_ASSISTANTS and OpenAIAssistantsBackend:
            try:
                self.backends["openai_assistants"] = OpenAIAssistantsBackend()
                logger.info("HybridBackend: OpenAIAssistantsBackend initialized")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error(
                    "HybridBackend: Failed to init OpenAIAssistantsBackend: %s", exc)

        # Default to what's available
        if ("google" in self.backends and initial_mode == "google"):
            self.current_mode = "google"
        elif ("openai_assistants" in self.backends and initial_mode == "openai_assistants"):
            self.current_mode = "openai_assistants"
        elif "local" in self.backends:
            self.current_mode = "local"
        else:
            logger.warning("HybridBackend: No backends available!")

    def set_mode(self, mode: str) -> bool:
        """Set the active backend mode."""
        # Handle top-level switching
        if mode == "local":
            if "local" in self.backends:
                self.current_mode = "local"
                logger.info("HybridBackend: Switched to local")
                return True
            return False

        # Handle OpenAI Assistants
        if mode == "openai_assistants":
            if "openai_assistants" in self.backends:
                self.current_mode = "openai_assistants"
                logger.info("HybridBackend: Switched to openai_assistants")
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
                    logger.info("HybridBackend: Switched to google (%s)", mode)
                    return True

            # Fallback for legacy "google" mode request -> manual
            if mode == "google":
                self.current_mode = "google"
                google_backend.set_mode("manual")
                logger.info("HybridBackend: Switched to google (manual)")
                return True

        return False

    def get_mode(self) -> str:
        """Get the current backend mode."""
        if self.current_mode == "google":
            # Return the specific google mode (manual or vertex_ai_search)
            if hasattr(self.backends["google"], "get_mode"):
                return self.backends["google"].get_mode()
        return self.current_mode

    def get_available_modes(self) -> List[str]:
        """Get list of available backend modes."""
        modes = []
        if "local" in self.backends:
            modes.append("local")
        if "openai_assistants" in self.backends:
            modes.append("openai_assistants")
        if "google" in self.backends:
            if hasattr(self.backends["google"], "get_available_modes"):
                modes.extend(self.backends["google"].get_available_modes())
            else:
                modes.append("google")
        return modes

    @property
    def _backend(self) -> RAGBackend:
        """Get the current backend instance."""
        if self.current_mode not in self.backends:
            raise RuntimeError(
                f"Current mode {self.current_mode} not available in backends: "
                f"{list(self.backends.keys())}")
        return self.backends[self.current_mode]

    # Delegate all methods
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for documents."""
        return self._backend.search(query, top_k)

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        """Add or update a document."""
        return self._backend.upsert_document(uri, text)

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a directory path."""
        return self._backend.index_path(path, glob)

    def grounded_answer(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Generate a grounded answer."""
        if hasattr(self._backend, "grounded_answer"):
            # Check if backend.grounded_answer accepts kwargs or specific args
            # For now, we assume updated backends accept kwargs or we inspect
            sig = inspect.signature(self._backend.grounded_answer)

            # If backend accepts kwargs, pass everything
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return self._backend.grounded_answer(question, k, **kwargs)

            # Otherwise, filter kwargs to what is accepted
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return self._backend.grounded_answer(question, k, **filtered_kwargs)

        return {"error": "Grounded answer not supported"}

    def load_store(self) -> bool:
        """Load the document store."""
        return self._backend.load_store()

    def save_store(self) -> bool:
        """Save the document store."""
        return self._backend.save_store()

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with metadata."""
        return self._backend.list_documents()

    def rebuild_index(self) -> None:
        """Rebuild the vector index."""
        self._backend.rebuild_index()

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank passages by relevance."""
        return self._backend.rerank(query, passages)

    def verify_grounding(self, question: str, answer: str,
                         citations: List[str]) -> Dict[str, Any]:
        """Verify answer grounding."""
        return self._backend.verify_grounding(question, answer, citations)

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        """Delete documents by URI."""
        return self._backend.delete_documents(uris)

    def get_deletion_status(self) -> Dict[str, Any]:
        """Get deletion queue status."""
        if hasattr(self._backend, "get_deletion_status"):
            return self._backend.get_deletion_status()
        # For backends without queue support, return empty status
        return {
            "queue_size": 0,
            "processing": False,
            "last_completed": None,
            "total_processed": 0
        }

    def flush_cache(self) -> Dict[str, Any]:
        """Clear the document cache."""
        return self._backend.flush_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self._backend.get_stats()
        stats["mode"] = self.current_mode
        stats["available_modes"] = self.get_available_modes()
        return stats

    def list_models(self) -> List[str]:
        """List available models."""
        if hasattr(self._backend, "list_models"):
            return self._backend.list_models()
        return []
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Chat with the backend."""
        if hasattr(self._backend, "chat"):
            sig = inspect.signature(self._backend.chat)

            # If backend accepts kwargs, pass everything
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return self._backend.chat(messages, **kwargs)

            # Otherwise, filter kwargs to what is accepted
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return self._backend.chat(messages, **filtered_kwargs)

        return {"error": "Chat not supported by current backend"}

    def list_drive_files(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """List Google Drive files."""
        if hasattr(self._backend, "list_drive_files"):
            return self._backend.list_drive_files(folder_id)
        return []

    def upload_file(self, name: str, content: bytes,
                    mime_type: str, folder_id: str = None) -> Dict[str, Any]:
        """Upload a file."""
        if hasattr(self._backend, "upload_file"):
            sig = inspect.signature(self._backend.upload_file)
            if "folder_id" in sig.parameters:
                return self._backend.upload_file(name, content, mime_type, folder_id)
            return self._backend.upload_file(name, content, mime_type)
        return {"error": "Upload not supported by current backend"}

    def delete_drive_file(self, file_id: str) -> Dict[str, Any]:
        """Delete a file or folder from Google Drive."""
        if hasattr(self._backend, "delete_drive_file"):
            return self._backend.delete_drive_file(file_id)
        return {"error": "Delete not supported by current backend"}

    def create_drive_folder(self, name: str, parent_id: str = None) -> Dict[str, Any]:
        """Create a folder in Google Drive."""
        if hasattr(self._backend, "create_drive_folder"):
            return self._backend.create_drive_folder(name, parent_id)
        return {"error": "Folder creation not supported by current backend"}

    def reload_auth(self) -> None:
        """Reload authentication for the current backend if supported."""
        if hasattr(self._backend, "reload_auth"):
            self._backend.reload_auth()


def get_rag_backend() -> RAGBackend:
    """Factory to get the configured backend."""
    # Check environment variable first
    mode = os.getenv("RAG_MODE", "").lower()
    # If not set in env, check settings.json
    if not mode:
        try:
            config_path = (pathlib.Path(__file__).resolve().parent.parent.parent /
                           "config" / "settings.json")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    mode = config.get("ragMode", "").lower()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to read settings.json: %s", exc)

    # Default to local if still not set
    # Default to local if still not set
    if not mode:
        mode = "local"

    if mode == "remote":
        url = os.getenv("RAG_REMOTE_URL", "http://127.0.0.1:8001/api")
        logger.info("Initializing RemoteBackend at %s", url)
        return RemoteBackend(url)

    logger.info("Initializing HybridBackend (initial mode: %s)", mode)
    return HybridBackend(initial_mode=mode)
