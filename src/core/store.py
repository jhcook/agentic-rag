"""
Document store management: load/save, chunking, path resolution.
"""

from __future__ import annotations
import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class Store:
    """Optimized document store - single source of truth for text."""
    docs: Dict[str, str] = field(default_factory=dict)
    last_loaded: float = 0.0

    def add(self, uri: str, text: str) -> None:
        self.docs[uri] = text


class StoreManager:
    """Encapsulates store persistence and chunking helpers."""

    def __init__(self, db_path: str, project_root: pathlib.Path, logger):
        self.db_path = db_path
        self.project_root = project_root
        self.logger = logger
        self._store: Optional[Store] = None

    def get_store(self) -> Store:
        if self._store is None:
            self._store = Store()
            try:
                self.load_store()
            except (OSError, ValueError) as exc:
                self.logger.debug("No existing store to load: %s", exc)
        return self._store

    def reset(self, store: Optional[Store] = None, db_path: Optional[str] = None) -> Store:
        """Replace the current store (primarily for testing)."""
        if db_path:
            self.db_path = db_path
        self._store = store or Store()
        return self._store

    def ensure_synced(self) -> None:
        if not os.path.exists(self.db_path):
            return
        try:
            file_mtime = os.path.getmtime(self.db_path)
            store = self.get_store()

            if file_mtime > store.last_loaded:
                self.logger.info("Detected external changes, reloading store from disk")
                self.load_store()
                store.last_loaded = time.time()
        except OSError as exc:
            self.logger.warning("Error checking store sync: %s", exc)

    def save_store(self) -> None:
        """Persist current store state to disk."""
        try:
            store = self.get_store()
            self.logger.info("Saving store to %s", self.db_path)
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            snapshot = list(store.docs.items())
            with open(self.db_path, "w", encoding="utf-8") as f:
                for uri, text in snapshot:
                    rec: Dict[str, Any] = {
                        "uri": uri,
                        "id": hashlib.sha1(uri.encode()).hexdigest(),
                        "text": text,
                        "ts": int(time.time()),
                    }
                    f.write(json.dumps(rec) + "\n")
            self.logger.info("Successfully saved %d documents", len(store.docs))
            store.last_loaded = time.time()
        except (OSError, ValueError) as exc:
            self.logger.error("Error saving store: %s", str(exc))
            raise

    def load_store(self) -> None:
        """Load store contents from disk."""
        if not os.path.exists(self.db_path):
            self.logger.warning("Store file not found at %s", self.db_path)
            return

        try:
            self.logger.info("Loading store from %s", self.db_path)
            new_store = Store()
            with open(self.db_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if "uri" in rec and "text" in rec:
                            new_store.add(rec["uri"], rec["text"])
                    except json.JSONDecodeError as exc:
                        self.logger.warning("load_store: %s", exc)
                        continue

            store = self.get_store()
            store.docs.clear()
            store.docs.update(new_store.docs)
            store.last_loaded = time.time()
            self.logger.info("Successfully loaded %d documents", len(store.docs))
        except (OSError, ValueError) as exc:
            self.logger.error("Error loading store: %s", str(exc))
            raise

    def chunk_document(self, text: str, uri: str, max_chars: int = 800, overlap: int = 120) -> Tuple[List[str], List[str]]:
        """Chunk a single document and return chunks with metadata."""
        all_chunks: List[str] = []
        chunk_metadata: List[str] = []

        i = 0
        n = len(text)

        while i < n:
            j = min(n, i + max_chars)
            chunk = text[i:j]
            all_chunks.append(chunk)
            chunk_metadata.append(uri)
            i += max_chars - overlap
            if i >= n:
                break

        return all_chunks, chunk_metadata

    def should_skip_uri(self, uri: str) -> bool:
        name = pathlib.Path(uri).name
        if not name:
            return False
        if name.startswith("."):
            return True
        if name.lower() in ("thumbs.db",):
            return True
        return False

    def resolve_input_path(self, path: str) -> pathlib.Path:
        raw = pathlib.Path(path).expanduser()
        candidates = []

        def _add_candidate(p: pathlib.Path):
            candidate = p.resolve()
            if candidate not in candidates:
                candidates.append(candidate)

        _add_candidate(raw)

        if not raw.is_absolute():
            _add_candidate(self.project_root / raw)
            env_base = os.getenv("RAG_WORKDIR")
            if env_base:
                _add_candidate(pathlib.Path(env_base).expanduser() / raw)

        for candidate in candidates:
            if candidate.exists():
                self.logger.debug("Resolved path '%s' to '%s'", path, candidate)
                return candidate

        attempted = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError("Path '%s' not found (tried: %s)" % (path, attempted))
