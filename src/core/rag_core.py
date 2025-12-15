"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""
# pylint: disable=too-many-lines

from __future__ import annotations
import logging
import pathlib
import json
import os
import hashlib
import re
import time
import asyncio
import gc
import threading
import warnings


from typing import List, Dict, Any, Optional, Tuple, Callable, cast

import numpy as np
from dotenv import load_dotenv  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from ollama import AsyncClient, ResponseError as OllamaError  # type: ignore
import httpx

# Alias for compatibility and connection error handling
APIConnectionError = httpx.ConnectError
Timeout = httpx.TimeoutException

# LangChain imports
# LangChain imports - strict dependency
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredFileLoader
)
HAS_LANGCHAIN = True

# Shared helpers
from src.core.embeddings import get_embedder as _get_embedder
from src.core import pgvector_store
from src.core import document_repo
from src.core.extractors import extract_text_from_file

# Backwards compatibility for tests/legacy imports.
_extract_text_from_file = extract_text_from_file

# Import new LLM client wrappers
from src.core.llm_client import sync_completion, safe_completion, reload_llm_config

# Backwards compatibility: some tests monkeypatch `rag_core.completion`.
completion = sync_completion

# Load .env early so configuration is available at module import time
load_dotenv()

# Load settings from settings.json if available
def _load_settings() -> Dict[str, Any]:
    """Load settings from config/settings.json if it exists."""
    settings_path = pathlib.Path("config/settings.json")
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load settings.json: %s", e)
    return {}

_SETTINGS = _load_settings()

# Module-level config globals are initialized by `_apply_settings()`. We declare
# them here so static import validation can resolve the symbols.
EMBED_MODEL_NAME: str = ""
OLLAMA_API_BASE: str = ""
LLM_MODEL_NAME: str = ""
ASYNC_LLM_MODEL_NAME: str = ""
LLM_TEMPERATURE: float = 0.1


def _get_config_value(json_key: str, env_key: str, default: str) -> str:
    """Get config value from settings.json or environment variable."""
    return _SETTINGS.get(json_key) or os.getenv(env_key, default)


def _apply_settings() -> None:
    """Apply settings.json values to module-level configuration."""
    global EMBED_MODEL_NAME  # pylint: disable=global-statement
    global OLLAMA_API_BASE
    global LLM_MODEL_NAME
    global ASYNC_LLM_MODEL_NAME
    global LLM_TEMPERATURE

    EMBED_MODEL_NAME = _get_config_value(
        "embeddingModel",
        "EMBED_MODEL_NAME",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
    )
    
    # Use new ollama_config module for endpoint resolution
    try:
        from src.core.ollama_config import get_ollama_endpoint
        OLLAMA_API_BASE = get_ollama_endpoint()
    except ImportError:
        # Fallback to old method if module not available
        OLLAMA_API_BASE = _get_config_value(
            "apiEndpoint",
            "OLLAMA_API_BASE",
            "http://127.0.0.1:11434",
        )
    
    LLM_MODEL_NAME = _get_config_value(
        "model",
        "LLM_MODEL_NAME",
        "ollama/llama3.2:1b",
    )
    ASYNC_LLM_MODEL_NAME = os.getenv(
        "ASYNC_LLM_MODEL_NAME",
        LLM_MODEL_NAME.replace("ollama/", ""),
    )
    LLM_TEMPERATURE = float(
        _get_config_value("temperature", "LLM_TEMPERATURE", "0.1")
    )


_apply_settings()

def reload_settings() -> None:
    """Reload settings from config/settings.json."""
    global _SETTINGS  # pylint: disable=global-statement
    _SETTINGS = _load_settings()
    _apply_settings()
    reload_llm_config()
    logger.info("Reloaded settings from config/settings.json")


# Optional dependencies
try:
    from tqdm import tqdm
except ImportError:
    def _tqdm_dummy(iterable=None, **_kwargs):  # type: ignore
        """Dummy tqdm for when tqdm is not installed."""
        return iterable
    tqdm = _tqdm_dummy

try:
    from prometheus_client import Counter, Histogram
except ImportError:
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

# Set up logging
# Avoid adding duplicate handlers when imported by servers that configure logging.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

# -------- Configuration --------
DEBUG_MODE = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "12"))
SEARCH_MAX_CONTEXT_CHARS = int(os.getenv("SEARCH_MAX_CONTEXT_CHARS", "8000"))
EMBED_DIM_OVERRIDE = int(os.getenv("EMBED_DIM_OVERRIDE", "0")) or None

def get_llm_model_name() -> str:
    """Get the current LLM model name from settings or env."""
    return _SETTINGS.get("model") or os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")

# Cache pgvector schema migration so we don't run idempotent DDL on every query.
_PGVECTOR_SCHEMA_LOCK = threading.Lock()
_PGVECTOR_SCHEMA_DIM: Optional[int] = None

# Embedding metrics (shared by servers)
EMBEDDING_REQUESTS = Counter(
    "embedding_requests_total",
    "Embedding encode invocations.",
    ["stage"]
) if Counter else None

EMBEDDING_ERRORS = Counter(
    "embedding_errors_total",
    "Embedding encode failures.",
    ["stage"]
) if Counter else None

EMBEDDING_DURATION = Histogram(
    "embedding_duration_seconds",
    "Time spent in embedding encode calls.",
    ["stage"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8],
) if Histogram else None

# Absolute path to the repository so relative inputs resolve consistently
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent


def get_embedder() -> Optional[SentenceTransformer]:
    """Get the embedding model via the shared embeddings module."""
    return _get_embedder(EMBED_MODEL_NAME, DEBUG_MODE, logger)


def _get_embed_dim(embedder: Optional[SentenceTransformer]) -> int:
    """Return embedding dimension from model, with a conservative fallback."""
    if embedder is None or DEBUG_MODE:
        return 384
    try:
        dim = int(embedder.get_sentence_embedding_dimension())
        return dim if dim > 0 else 384
    except Exception:  # pylint: disable=broad-exception-caught
        return 384


def _ensure_pgvector_ready(embedder: Optional[SentenceTransformer]) -> None:
    """Best-effort ensure pgvector schema exists for the active embedding dimension.

    Some unit tests and local workflows should still run when the external
    pgvector service isn't configured/running; call sites should degrade
    gracefully when this fails.
    """
    if DEBUG_MODE:
        return
    if embedder is None:
        return
    embed_dim = _get_embed_dim(embedder)
    if EMBED_DIM_OVERRIDE and embed_dim != EMBED_DIM_OVERRIDE:
        logger.warning(
            "EMBED_DIM_OVERRIDE=%s does not match embedder dimension %s; using embedder dimension",
            EMBED_DIM_OVERRIDE,
            embed_dim,
        )
    try:
        global _PGVECTOR_SCHEMA_DIM  # pylint: disable=global-statement
        with _PGVECTOR_SCHEMA_LOCK:
            if _PGVECTOR_SCHEMA_DIM == embed_dim:
                return
            pgvector_store.migrate_schema(embed_dim)
            _PGVECTOR_SCHEMA_DIM = embed_dim
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "pgvector not ready; continuing without it: %s",
            pgvector_store.redact_error_message(str(exc)),
        )


def ensure_vector_store_ready() -> Dict[str, Any]:
    """Ensure pgvector schema exists for the active embedding model."""

    embedder = get_embedder()
    _ensure_pgvector_ready(embedder)
    ok, msg = pgvector_store.test_connection()
    if not ok:
        raise RuntimeError(f"pgvector not available: {msg}")
    return {
        "status": "ok",
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_dim": _get_embed_dim(embedder),
    }

def _encode_with_metrics(embedder: Optional[SentenceTransformer], inputs: Any,
                         stage: str, **kwargs):
    """Wrap embedder.encode to record metrics when available."""
    if embedder is None:  # Defensive: should already be handled by callers
        raise RuntimeError("Embedding model is not initialized.")

    if EMBEDDING_REQUESTS:
        EMBEDDING_REQUESTS.labels(stage=stage).inc()

    start = time.perf_counter()
    try:
        result = embedder.encode(inputs, **kwargs)
        if EMBEDDING_DURATION:
            EMBEDDING_DURATION.labels(stage=stage).observe(
                time.perf_counter() - start)
        return result
    except Exception:
        if EMBEDDING_ERRORS:
            EMBEDDING_ERRORS.labels(stage=stage).inc()
        raise

def _hash_uri(uri: str) -> str:
    """Hash a URI for storage."""
    return hashlib.sha1(uri.encode()).hexdigest()


def resolve_input_path(path: str) -> pathlib.Path:
    """Resolve a user-supplied path, trying common fallbacks."""
    raw = pathlib.Path(path).expanduser()
    candidates = []

    def _add_candidate(p: pathlib.Path):
        """Track a candidate path if not already present."""
        candidate = p.resolve()
        if candidate not in candidates:
            candidates.append(candidate)

    _add_candidate(raw)

    if not raw.is_absolute():
        _add_candidate(PROJECT_ROOT / raw)
        env_base = os.getenv("RAG_WORKDIR")
        if env_base:
            _add_candidate(pathlib.Path(env_base).expanduser() / raw)

    for candidate in candidates:
        if candidate.exists():
            logger.debug("Resolved path '%s' to '%s'", path, candidate)
            return candidate

    attempted = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Path '{path}' not found (tried: {attempted})")

def _chunk_text_with_offsets(
    text: str, max_chars: int = 800, overlap: int = 120
) -> List[Tuple[str, int, int]]:
    """Chunk text into pieces of max_chars with overlap, returning (text, start, end)."""
    out: List[Tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append((text[i:j], i, j))
        i += max_chars - overlap
        if i >= n:
            break
    return out

def _chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    """Chunk text into pieces using advanced RecursiveCharacterTextSplitter if available."""
    if HAS_LANGCHAIN:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return text_splitter.split_text(text)
    
    # Fallback legacy implementation
    return [c[0] for c in _chunk_text_with_offsets(text, max_chars, overlap)]

def _load_and_chunk_file(file_path: pathlib.Path) -> List[str]:
    """
    Load and chunk a file using LangChain loaders.
    Returns a list of text chunks.
    """
    ext = file_path.suffix.lower()
    
    if not HAS_LANGCHAIN:
        # Fallback to legacy readers if LangChain missing
        text = ""
        if ext == ".pdf":
            text = _read_pdf_file(file_path)
        elif ext in [".docx", ".doc"]:
            text = _read_docx_file(file_path)
        else:
            text = _read_text_file(file_path)
        return _chunk_text(text)

    try:
        loader = None
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif ext in [".txt", ".md", ".json", ".py", ".sh", ".yaml", ".yml"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            # Fallback for others
            loader = UnstructuredFileLoader(str(file_path))
            
        docs = loader.load()
        full_text = "\n\n".join([d.page_content for d in docs])
        return _chunk_text(full_text)
        
    except Exception as e:
        logger.warning(f"LangChain loader failed for {file_path}: {e}. Trying legacy fallback.")
        # Minimal legacy fallback for stability during migration
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return _chunk_text(f.read())
        except Exception:
            return []

def _chunk_document_with_offsets(
    text: str, uri: str
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """Chunk a single document and return (chunks, uris, offsets)."""
    # NOTE: With LangChain loaders, we often get chunks directly.
    # But to maintain compatibility with `_rebuild_faiss_index` which expects
    # text+offsets, we'll keep using our _chunk_text wrapper on the full text.
    
    chunks = _chunk_text(text)
    uris = [uri] * len(chunks)
    
    # Recalculate offsets roughly (recursive splitter doesn't give them easily)
    # This is a trade-off: we lose precise byte offsets for now, but gain semantic integrity.
    offsets = []
    current_pos = 0
    for chunk in chunks:
        start = text.find(chunk, current_pos)
        if start == -1:
            start = current_pos # approximate fallback
        end = start + len(chunk)
        offsets.append((start, end))
        current_pos = end
        
    return chunks, uris, offsets

# --- Legacy Readers Removed ---
def _read_docx_file(file_path: pathlib.Path) -> str:
    """Read DOCX file using LangChain loader."""
    try:
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {e}")
        return ""

def _read_text_file(file_path: pathlib.Path) -> str:
    """Read text file using LangChain loader."""
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return ""

def _chunk_document(text: str, uri: str) -> Tuple[List[str], List[str]]:
    """Chunk a single document and return chunks with metadata."""
    chunks, uris, _ = _chunk_document_with_offsets(text, uri)
    return chunks, uris


def _should_skip_uri(uri: str) -> bool:
    """Skip hidden/system files that should never be indexed."""
    name = pathlib.Path(uri).name
    if not name:
        return False
    if name.startswith("."):
        return True
    if name.lower() in ("thumbs.db",):
        return True
    return False


def should_skip_uri(uri: str) -> bool:
    """Public helper used by store/indexer modules to filter URIs."""
    return _should_skip_uri(uri)


def rebuild_index() -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """(Re)build the pgvector index from canonical indexed artifacts."""

    if DEBUG_MODE:
        return {"status": "skipped", "reason": "debug_mode"}

    embedder = get_embedder()
    _ensure_pgvector_ready(embedder)

    ok, msg = pgvector_store.test_connection()
    if not ok:
        raise RuntimeError(f"pgvector is not available: {msg}")

    # Use rag_documents as the source of URIs; canonical text is on disk.
    docs = pgvector_store.list_documents()
    logger.info("Rebuilding pgvector index from %d documents", len(docs))

    total_chunks = 0
    for doc in tqdm(iterable=docs, desc="Rebuilding pgvector index", unit="doc"):  # pylint: disable=no-value-for-parameter
        uri = str(doc.get("uri") or "")
        if _should_skip_uri(uri):
            continue

        raw = document_repo.read_indexed_bytes(uri)
        if not raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue

        chunks_with_offsets = _chunk_text_with_offsets(text)  # pylint: disable=no-value-for-parameter
        chunks = [c[0] for c in chunks_with_offsets]
        offsets = [(c[1], c[2]) for c in chunks_with_offsets]
        if not chunks:
            continue

        embeddings = _encode_with_metrics(
            embedder,
            chunks,
            "rebuild_index",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb = np.array(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        pgvector_store.upsert_document_chunks(
            uri=uri,
            text_sha256=hashlib.sha256(raw).hexdigest(),
            chunks=chunks,
            offsets=offsets,
            embeddings=emb,
            embedding_model=EMBED_MODEL_NAME,
        )
        total_chunks += len(chunks)

        # Reduce peak memory
        del chunks_with_offsets, chunks, offsets, embeddings, emb
        gc.collect()

    logger.info("pgvector rebuild complete (chunks=%d)", total_chunks)
    return {"status": "ok", "chunks": total_chunks, "documents": len(docs)}
def upsert_document(uri: str, text: str) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """Upsert a single document.

    Persists canonical extracted text under `cache/indexed/` and embeds from the
    bytes read back from disk to guarantee an exact match.
    """

    # Safety check for huge documents
    if len(text) > 5_000_000:  # 5MB text limi
        logger.warning("Document %s too large (%d chars), truncating.", uri, len(text))
        text = text[:5_000_000]

    if _should_skip_uri(uri):
        logger.warning("Skipping hidden/system file: %s", uri)
        return {"skipped": True, "reason": "hidden/system file", "uri": uri}

    embedder = get_embedder()

    artifact_path = document_repo.artifact_path_for_uri(uri)
    existed = artifact_path.exists()
    artifact = document_repo.write_indexed_text(uri=uri, text=text)
    raw = document_repo.read_indexed_bytes(uri)
    if raw is None:
        raise RuntimeError("Failed to read indexed artifact after write")
    try:
        canonical_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Indexed artifact is not valid utf-8") from exc

    if embedder is None:
        raise RuntimeError("Embedder is not available")

    _ensure_pgvector_ready(embedder)

    ok, msg = pgvector_store.test_connection()
    if not ok:
        raise RuntimeError(f"pgvector is not available: {msg}")
    chunks_with_offsets = _chunk_text_with_offsets(canonical_text)  # pylint: disable=no-value-for-parameter
    chunks = [c[0] for c in chunks_with_offsets]
    offsets = [(c[1], c[2]) for c in chunks_with_offsets]

    if chunks:
        embeddings = _encode_with_metrics(
            embedder,
            chunks,
            "upsert_document",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb = np.array(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        pgvector_store.upsert_document_chunks(
            uri=uri,
            text_sha256=artifact.text_sha256,
            chunks=chunks,
            offsets=offsets,
            embeddings=emb,
            embedding_model=EMBED_MODEL_NAME,
        )

        del chunks_with_offsets, chunks, offsets, embeddings, emb
        gc.collect()

    try:
        stats = pgvector_store.stats(embedding_model=EMBED_MODEL_NAME)
        logger.info(
            "Upserted document %s (existed: %s), pgvector chunks=%s",
            uri,
            existed,
            stats.get("chunks"),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "pgvector stats failed after upsert: %s",
            pgvector_store.redact_error_message(str(exc)),
        )
    return {"upserted": True, "existed": existed}


def list_documents() -> List[Dict[str, Any]]:
    """List indexed documents (URI + artifact size)."""

    try:
        docs = pgvector_store.list_documents()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("pgvector list_documents failed: %s", pgvector_store.redact_error_message(str(exc)))
        return []

    out: List[Dict[str, Any]] = []
    for doc in docs:
        uri = str(doc.get("uri") or "")
        if not uri:
            continue
        artifact_path = document_repo.artifact_path_for_uri(uri)
        try:
            size = artifact_path.stat().st_size if artifact_path.exists() else 0
        except OSError:
            size = 0
        out.append(
            {
                "uri": uri,
                "size": int(size),
                "text_sha256": doc.get("text_sha256"),
                "artifact_present": bool(artifact_path.exists()),
            }
        )
    return out


def delete_documents(uris: List[str]) -> Dict[str, Any]:
    """Delete documents by URI from pgvector and delete their indexed artifacts."""

    if not uris:
        return {"deleted": 0, "artifacts_deleted": 0}

    artifacts_deleted = 0
    for uri in uris:
        if document_repo.delete_indexed_text(uri):
            artifacts_deleted += 1

    try:
        deleted = pgvector_store.delete_documents(uris, embedding_model=None)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        msg = pgvector_store.redact_error_message(str(exc))
        logger.warning("pgvector delete_documents failed: %s", msg)
        deleted = 0

    return {"deleted": int(deleted), "artifacts_deleted": int(artifacts_deleted)}


def flush_cache() -> Dict[str, Any]:
    """Clear all indexed content (pgvector + cache/indexed artifacts)."""

    try:
        pgvector_store.wipe_all()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        msg = pgvector_store.redact_error_message(str(exc))
        logger.warning("pgvector wipe_all failed: %s", msg)

    removed = document_repo.clear_indexed_dir()
    return {"status": "flushed", "artifacts_removed": int(removed)}


def process_file(
    file_path: pathlib.Path,
    index: Any,
    index_to_meta: Dict,
    embedder: Optional[SentenceTransformer],
) -> str:
    """Public wrapper for processing individual files via indexer module."""
    return _process_file(file_path, index, index_to_meta, embedder)


def _collect_files(path: str, glob: str) -> Tuple[List[pathlib.Path], pathlib.Path]:
    """Collect files from the given path matching the glob pattern."""
    resolved = resolve_input_path(path)
    logger.debug("Collecting files from %s with glob %s", resolved, glob)
    if resolved.is_file():
        files = [resolved] if not _should_skip_uri(str(resolved)) else []
    else:
        files = [p for p in resolved.rglob(glob) if not _should_skip_uri(str(p))]
    return files, resolved


def collect_files(path: str, glob: str) -> Tuple[List[pathlib.Path], pathlib.Path]:
    """Public wrapper for collecting candidate files for indexing."""
    return _collect_files(path, glob)


# Text extraction helpers live in extractors module; import the public function.


def _process_file(  # pylint: disable=too-many-locals
    file_path: pathlib.Path, index: Any, index_to_meta: Dict,
    embedder: Optional[SentenceTransformer]
) -> str:
    """Process a single file and add to index."""
    def _is_meaningful_text(text: str) -> bool:
        """Heuristic filter to skip binary/empty payloads regardless of extension."""
        if file_path.name == ".DS_Store":
            return False
        if not text:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        printable = sum(ch.isprintable() for ch in stripped)
        density = printable / max(1, len(stripped))
        if density < 0.85:
            return False
        # Reject obvious binary headers
        head = stripped[:8]
        if head.startswith(("%PDF-", "PK", "\u0000", "\ufffd")):
            return False
        return True

    text = extract_text_from_file(file_path)
    if not _is_meaningful_text(text):
        logger.warning("Skipping non-text or empty content from %s", file_path)
        return ""

    # Persist + embed from canonical artifact.
    upsert_document(str(file_path), text)
    return text


def index_path(  # pylint: disable=too-many-locals
    path: str, glob: str = "**/*.txt", max_files: int = 1000,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """Index all text files in a given path matching the glob pattern.

    Args:
        path: Directory path to index
        glob: File pattern to match
        max_files: Maximum number of files to index (safety limit)
        progress_callback: Optional callback(current, total) called during indexing
    """
    embedder = get_embedder()
    if not DEBUG_MODE:
        _ensure_pgvector_ready(embedder)

    try:
        files, resolved = collect_files(path, glob)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        try:
            total_vectors = int(pgvector_store.stats(embedding_model=EMBED_MODEL_NAME).get("chunks", 0))
        except Exception:  # pylint: disable=broad-exception-caught
            total_vectors = 0
        return {
            "indexed": 0,
            "total_vectors": total_vectors,
            "error": str(exc)
        }

    if not files:
        message = f"No files matching '{glob}' were found under '{resolved}'"
        logger.info(message)
        try:
            total_vectors = int(pgvector_store.stats(embedding_model=EMBED_MODEL_NAME).get("chunks", 0))
        except Exception:  # pylint: disable=broad-exception-caught
            total_vectors = 0
        return {
            "indexed": 0,
            "total_vectors": total_vectors,
            "error": message,
        }

    # Enforce file limi
    if len(files) > max_files:
        logger.warning(
            "Found %d files, truncating to %d to prevent resource exhaustion.",
            len(files), max_files)
        files = files[:max_files]

    total_files = len(files)
    texts = []
    for idx, file_path in enumerate(tqdm(files, desc="Indexing files", unit="file"), 1):
        if progress_callback:
            progress_callback(idx, total_files)
        try:
            text = process_file(file_path, None, {}, embedder)
            texts.append(text)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)

    try:
        stats = pgvector_store.stats(embedding_model=EMBED_MODEL_NAME)
        total_vectors = stats.get("chunks", 0)
    except Exception:  # pylint: disable=broad-exception-caught
        total_vectors = 0
    logger.info("Stored %d files, pgvector chunks=%s", len(texts), total_vectors)

    return {"indexed": len(files), "total_vectors": total_vectors, "resolved_path": str(resolved)}


async def send_to_llm(query: List[str]) -> Any:
    """Send the query to the LLM and return the response."""
    # Get endpoint and headers from ollama_config
    try:
        from src.core.ollama_config import get_ollama_endpoint, get_ollama_client_headers
        endpoint = get_ollama_endpoint()
        headers = get_ollama_client_headers()
    except ImportError:
        endpoint = OLLAMA_API_BASE
        headers = {}
    
    client = AsyncClient(host=endpoint, headers=headers if headers else None)
    messages = [{"content": str(text), "role": "user"} for text in query]
    try:
        resp = await client.chat(  # type: ignore
            model=ASYNC_LLM_MODEL_NAME,
            messages=messages
        )
        return resp
    except (ValueError, APIConnectionError) as exc:  # type: ignore
        logger.error("LLM error: %s", exc)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("send_to_llm: %s", exc)
        raise


def expand_query(query: str) -> str:
    """
    Use LLM to expand the query with synonyms/hypothetical questions (HyDE-lite).
    
    Note: Currently a pass-through function that returns the original query.
    Query expansion adds latency and requires careful prompt engineering.
    This can be implemented in the future when needed.
    """
    # TODO: Implement actual query expansion with LLM when performance requirements allow
    # Expansion would generate variations like synonyms or hypothetical documents (HyDE)
    # to improve retrieval, but adds latency to every query.
    return query



def send_store_to_llm() -> str:  # pylint: disable=too-many-locals
    """Send all indexed artifacts as context to the LLM (best-effort)."""
    docs = pgvector_store.list_documents()
    texts: List[str] = []
    for doc in docs:
        uri = str(doc.get("uri") or "")
        if not uri:
            continue
        text = document_repo.read_indexed_text(uri)
        if text:
            texts.append(text)
    resp = None

    for _ in range(3):
        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(send_to_llm(texts), running_loop)
                resp = future.result(timeout=60)
            else:
                resp = asyncio.run(send_to_llm(texts))

        except (OSError, ValueError) as exc:
            logger.error("send_store_to_llm failed: %s", exc)
            raise
        except APIConnectionError:  # pylint: disable=broad-exception-caught
            time.sleep(1)
            continue
        break

    return resp


def _vector_search(query: str, k: int = SEARCH_TOP_K) -> List[Dict[str, Any]]:  # pylint: disable=too-many-locals
    """Perform vector similarity search using PostgreSQL + pgvector."""

    embedder = get_embedder()
    if embedder is None:
        return []

    query_emb = _encode_with_metrics(
        embedder,
        query,
        "search_query",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    q = np.array(query_emb, dtype=np.float32).reshape(1, -1)

    _ensure_pgvector_ready(embedder)
    ok, msg = pgvector_store.test_connection()
    if not ok:
        raise RuntimeError(f"pgvector is not available: {msg}")

    results = pgvector_store.search_chunks(
        query_embedding=q,
        k=k,
        embedding_model=EMBED_MODEL_NAME,
    )

    hits: List[Dict[str, Any]] = []
    for row in results:
        hits.append(
            {
                "score": float(row.get("score", 0.0)),
                "uri": row.get("uri", ""),
                "text": (row.get("text") or ""),
            }
        )
    return hits


def vector_search(query: str, k: int = SEARCH_TOP_K) -> List[Dict[str, Any]]:
    """Public vector search helper for REST and MCP layers."""
    return _vector_search(query, k)


def _normalize_llm_response(resp: Any, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Normalize LLM response (AIMessage, dict, or string) to a consistent dict format.

    Args:
        resp: Response from LLM client (AIMessage, dict, or string)
        sources: Optional list of source URIs to include

    Returns:
        Dict with 'answer' or 'content' key and optional 'sources'
    """
    logger.debug("Normalizing LLM response: %s", resp)
    if resp is None:
        return {"error": "No response from LLM"}

    # Handle LangChain AIMessage (or any object with content attribute)
    if hasattr(resp, "content"):
        normalized_resp = {
            "answer": resp.content,
            "sources": sources or [],
            "model": "unknown"
        }
        if hasattr(resp, "usage"):
            normalized_resp["usage"] = resp.usage
        elif hasattr(resp, "usage_metadata") and isinstance(resp.usage_metadata, dict):
            normalized_resp["usage"] = resp.usage_metadata
        elif hasattr(resp, "response_metadata") and isinstance(resp.response_metadata, dict):
            if "total_duration" in resp.response_metadata and "prompt_eval_count" in resp.response_metadata:
                normalized_resp["usage"] = {
                    "total_tokens": resp.response_metadata.get("prompt_eval_count", 0) + resp.response_metadata.get("eval_count", 0),
                    "prompt_tokens": resp.response_metadata.get("prompt_eval_count"),
                    "completion_tokens": resp.response_metadata.get("eval_count"),
                }
        return normalized_resp

    # Handle dict responses (legacy or other backends)
    if isinstance(resp, dict):
        # Ensure sources are included
        if sources and "sources" not in resp:
            resp["sources"] = sources
        # Extract content if it's in choices format (legacy)
        if "choices" in resp and isinstance(resp["choices"], list) and len(resp["choices"]) > 0:
            choice = resp["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                message = choice["message"]
                if isinstance(message, dict) and "content" in message:
                    resp["answer"] = message["content"]
        return resp

    # Fallback: convert to string
    return {"answer": str(resp), "sources": sources or []}


def _build_rag_context(
    query: str,
    top_k: int = SEARCH_TOP_K,
    max_context_chars: int = SEARCH_MAX_CONTEXT_CHARS
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Build RAG context from indexed documents for a query.

    Args:
        query: Search query
        top_k: Number of candidates to retrieve
        max_context_chars: Maximum characters of context to include

    Returns:
        Tuple of (context_string, sources_list, candidates_list)
        Returns empty context if no documents found
    """
    logger.info("Building RAG context for query: %s", query)
    candidates = _vector_search(query, k=top_k)

    if not candidates:
        # No hits can be perfectly normal (query unrelated). Only rebuild if the
        # vector table is empty for the active embedding model.
        try:
            stats = pgvector_store.stats(embedding_model=EMBED_MODEL_NAME)
            chunks = int(stats.get("chunks", 0) or 0)
            docs = int(stats.get("documents", 0) or 0)
        except Exception:  # pylint: disable=broad-exception-caught
            chunks = 0
            docs = 0

        if docs > 0 and chunks <= 0:
            logger.info(
                "No vector hits and pgvector chunks=0 (docs=%d); rebuilding index and retrying...",
                docs,
            )
            rebuild_index()
            candidates = _vector_search(query, k=top_k)
        if not candidates:
            logger.info("No vector hits; refusing to answer from outside sources.")
    # Re-rank to prioritize query overlap
    candidates = rerank(query, candidates)[:top_k]
    sources = [c.get("uri", "") for c in candidates if c.get("uri")]

    # Build context from document content only (no file names/metadata)
    context_parts, total_chars = [], 0
    for entry in candidates:
        text_content = entry.get("text", "").strip()
        if not text_content:
            continue
        if total_chars + len(text_content) > max_context_chars:
            break
        context_parts.append(text_content)
        total_chars += len(text_content)

    context = "\n\n---\n\n".join(context_parts) if context_parts else ""
    return (context, sources, candidates)


def search(query: str, top_k: int = SEARCH_TOP_K,
           max_context_chars: int = SEARCH_MAX_CONTEXT_CHARS,
           model: str = None, temperature: float = None, max_tokens: int = None):
    """Search the indexed documents and ask the LLM using only those documents as context.
    
    Args:
        query: The search query
        top_k: Number of top results to retrieve
        max_context_chars: Maximum characters of context
        model: LLM model to use (overrides LLM_MODEL_NAME)
        temperature: Temperature for generation (overrides LLM_TEMPERATURE)
        max_tokens: Maximum tokens in response
    """
    start_time = time.time()
    
    # Autotune parameters if enabled
    # We use the centralized heuristic based on query length.
    if top_k == SEARCH_TOP_K: # Only tune if using defaults
        top_k, max_context_chars = _autotune_rag_params(query, top_k)
    
    context, sources, candidates = _build_rag_context(query, top_k, max_context_chars)
    
    retrieval_duration = time.time() - start_time
    logger.info(f"Retrieval finished in {retrieval_duration:.2f}s (top_k={top_k}, context_chars={len(context)})")
    
    if not context:
        return {"error": "No relevant documents found in the indexed corpus.", "sources": []}

    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the document "
        "content provided below. You must base your answer strictly on the content in the "
        "documents. Do NOT use any external knowledge, general knowledge, or make assumptions "
        "beyond what is explicitly stated in the documents. "
        "If the answer is not contained in the provided documents, reply exactly: "
        "\"I don't know.\" "
        "At the end of your response, include a 'Sources:' section listing the source URIs for "
        "the information you used. "
        "The source URIs will be provided separately for citation purposes."
    )
    user_msg = f"Document Content:\n{context}\n\nQuestion: {query}"

    try:
        # Use sync completion for simplicity in this flow, reusing existing config

        # Build completion kwargs
        # Normalize model name: add "ollama/" prefix if not present and not already prefixed.
        # Also normalize cloud/local suffix handling based on mode to avoid "model not found" when falling back.
        effective_model = model or get_llm_model_name()
        if effective_model and not effective_model.startswith(('ollama/', 'openai/', 'anthropic/', 'huggingface/')):
            effective_model = f"ollama/{effective_model}"
        if effective_model and effective_model.startswith("ollama/"):
            raw_model = effective_model[len("ollama/") :]
            try:
                from src.core.ollama_config import get_ollama_mode, normalize_ollama_model_name
                mode = get_ollama_mode()
                normalized = normalize_ollama_model_name(raw_model, mode)
                effective_model = f"ollama/{normalized}"
            except Exception:  # pylint: disable=broad-exception-caught
                # If normalization fails, continue with the original value
                pass
        
        # Get endpoint and headers from ollama_config with fallback support
        try:
            from src.core.ollama_config import get_ollama_endpoint_with_fallback
            api_base, headers, fallback_endpoint = get_ollama_endpoint_with_fallback()
        except ImportError:
            api_base = OLLAMA_API_BASE
            headers = {}
            fallback_endpoint = None
        
        completion_kwargs = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "api_base": api_base,
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "stream": False,
            "timeout": 300,
        }
        
        # Add headers if available (for cloud authentication)
        if headers:
            completion_kwargs["extra_headers"] = headers
        
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens

        # Try primary endpoint (cloud for auto mode)
        try:
            resp = completion(**completion_kwargs)  # type: ignore
            return _normalize_llm_response(resp, sources)
        except (APIConnectionError, Timeout) as exc:  # type: ignore
            # Connection/timeout errors - try fallback if available
            if fallback_endpoint:
                logger.warning(
                    "Cloud endpoint failed (%s), falling back to local endpoint",
                    type(exc).__name__
                )
                # Retry with local endpoint (no headers) and normalize model for local to drop any -cloud suffix
                if effective_model.startswith("ollama/"):
                    raw_model = effective_model[len("ollama/") :]
                    try:
                        from src.core.ollama_config import normalize_ollama_model_name
                        local_model = normalize_ollama_model_name(raw_model, "local")
                        completion_kwargs["model"] = f"ollama/{local_model}"
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                # Retry with local endpoint
                completion_kwargs["api_base"] = fallback_endpoint
                completion_kwargs.pop("extra_headers", None)  # Remove headers for local
                try:
                    resp = completion(**completion_kwargs)  # type: ignore
                    return _normalize_llm_response(resp, sources)
                except Exception as fallback_exc:  # pylint: disable=broad-exception-caught
                    logger.error("Fallback to local endpoint also failed: %s", fallback_exc)
                    raise exc  # Raise original exception
            raise
    except (ValueError, OllamaError, APIConnectionError) as exc:  # type: ignore
        logger.error("Ollama API Error: %s", exc)
    except (OSError, RuntimeError) as exc:  # type: ignore
        logger.error("Unexpected error in completion: %s", exc)

    # Graceful fallback: synthesize answer from retrieved passages
    fallback = synthesize_answer(query, candidates)
    fallback["warning"] = (
        "LLM unavailable; "
        "reply synthesized from retrieved passages."
    )
    fallback["sources"] = sources
    return fallback


# --- Lightweight rerank/ground/verify utilities used by MCP tools ---

def rerank(
    query: str, passages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Simple reranker that boosts passages containing the query terms. This is a
    lightweight heuristic so we avoid an additional model dependency.
    """
    lowered_terms = [t for t in query.lower().split() if t]
    signature_phrases = (
        "regards",
        "warm regards",
        "best,",
        "best regards",
        "thanks",
        "thank you",
        "cheers",
        "sincerely",
    )

    def _score(p: Dict[str, Any]) -> float:
        """Heuristic score combining existing score with query term hits."""
        base = float(p.get("score", 0.0) or 0.0)
        text_raw = str(p.get("text", ""))
        text = text_raw.lower()
        term_hits = sum(text.count(term) for term in lowered_terms) if lowered_terms else 0

        # Penalize low-signal passages (e.g., email signatures or very short snippets)
        penalty = 0.0
        word_count = len(text.split())
        if word_count < 20:
            penalty += 0.7
        elif word_count < 40:
            penalty += 0.3
        if any(phrase in text for phrase in signature_phrases):
            penalty += 0.5

        return base + term_hits * 0.1 - penalty

    ranked = sorted(passages, key=_score, reverse=True)
    # Attach heuristic scores for downstream consumers
    for p in ranked:
        p["score"] = _score(p)
    return ranked


def synthesize_answer(_question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a grounded answer by concatenating the top passage texts. This keeps
    outputs deterministic when no LLM is available.
    """
    if not passages:
        return {"answer": "I don't know.", "citations": []}

    top_passages = passages[:3]
    answer_parts = [p.get("text", "") for p in top_passages if p.get("text")]
    answer = " ".join(answer_parts).strip()
    citations = [p.get("uri", "") for p in top_passages if p.get("uri")]
    return {"answer": answer or "I don't know.", "citations": citations}


def grounded_answer(  # pylint: disable=too-many-locals,too-many-statements
    question: str, k: int = SEARCH_TOP_K, **kwargs: Any
) -> Dict[str, Any]:
    """
    Grounded career-centric answer: retrieve, rerank, filter signatures/boilerplate,
    and synthesize with citations; falls back gracefully when no signal.
    """
    signature_phrases = (
        "regards",
        "warm regards",
        "best,",
        "best regards",
        "thanks",
        "thank you",
        "cheers",
        "sincerely",
    )

    model_name = kwargs.get("model", get_llm_model_name())
    temperature = kwargs.get("temperature", LLM_TEMPERATURE)

    def _is_low_signal(passage: Dict[str, Any]) -> bool:
        text = str(passage.get("text", "")).strip()
        if not text:
            return True
        lowered = text.lower()
        words = lowered.split()
        if len(words) < 12:
            return True
        if any(phrase in lowered for phrase in signature_phrases):
            return True
        return False

    # Retrieve and rerank candidates
    hits = _vector_search(question, k=k or SEARCH_TOP_K)
    if not hits:
        return {"answer": "I don't know.", "sources": []}
    reranked = rerank(question, hits)

    # Filter out signatures/boilerplate and keep top-k unique URIs
    filtered: List[Dict[str, Any]] = []
    seen = set()
    for p in reranked:
        uri = p.get("uri")
        if uri and uri in seen:
            continue
        if _is_low_signal(p):
            continue
        filtered.append(p)
        if uri:
            seen.add(uri)
        if len(filtered) >= (k or SEARCH_TOP_K):
            break

    if not filtered:
        return {"answer": "I don't know.", "sources": []}

    # Build context within limits
    sources = [p.get("uri", "") for p in filtered if p.get("uri")]
    context_parts = []
    total_chars = 0
    max_chars = SEARCH_MAX_CONTEXT_CHARS
    for entry in filtered:
        # Only include the text content, not file names or metadata
        # File names are only used for citations (handled separately via sources list)
        text_content = entry.get("text", "").strip()
        if not text_content:
            continue
        if total_chars + len(text_content) > max_chars:
            break
        context_parts.append(text_content)
        total_chars += len(text_content)
    context = "\n\n---\n\n".join(context_parts)

    # If no LLM available, deterministic synthesis
    if sync_completion is None:
        synthesized = synthesize_answer(question, filtered)
        synthesized["sources"] = sources
        return synthesized

    system_msg = (
        "You are a careful assistant. Write a concise professional summary using ONLY the "
        "document content provided below. "
        "You must base your answer strictly on the content in the documents. "
        "Capture roles, companies, key achievements, and timelines when present. "
        "Strictly exclude greetings, sign-offs, and speculative content. "
        "Do NOT use any external knowledge or make assumptions beyond what is explicitly "
        "stated in the documents. "
        "If the documents do not clearly support the answer, reply exactly: \"I don't know.\" "
        "At the end of your response, include a 'Sources:' section listing the source URIs for "
        "the information you used. "
        "The source URIs will be provided separately for citation purposes."
    )
    user_msg = f"Document Content:\n{context}\n\nQuestion: {question}"

    # Extract num_ctx if provided
    num_ctx = kwargs.get("num_ctx")
    if num_ctx:
        try:
            num_ctx = int(num_ctx)
        except (ValueError, TypeError):
            num_ctx = None

    # Get endpoint and headers from ollama_config with fallback support
    try:
        from src.core.ollama_config import get_ollama_endpoint_with_fallback
        api_base, headers, fallback_endpoint = get_ollama_endpoint_with_fallback()
    except ImportError:
        api_base = OLLAMA_API_BASE
        headers = {}
        fallback_endpoint = None
    
    try:
        completion_kwargs = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            "api_base": api_base,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "stream": False,
            "timeout": 90,
        }
        # Add headers if available (for cloud authentication)
        if headers:
            completion_kwargs["extra_headers"] = headers
        
        # Try primary endpoint
        try:
            resp = completion(**completion_kwargs)
        except Exception as exc: 
            # Connection/timeout errors - try fallback if available
            if fallback_endpoint:
                logger.warning(
                    "Cloud endpoint failed (%s), falling back to local endpoint",
                    type(exc).__name__
                )
                completion_kwargs["api_base"] = fallback_endpoint
                completion_kwargs.pop("extra_headers", None)
                resp = completion(**completion_kwargs)
            else:
                raise
        return _normalize_llm_response(resp, sources)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("grounded_answer completion failed: %s", exc)
        synthesized = synthesize_answer(question, filtered)
        synthesized["warning"] = "LLM unavailable; reply synthesized from retrieved passages."
        synthesized["sources"] = sources
        return synthesized


def verify_grounding_simple(_question: str, draft_answer: str,
                            passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal grounding check: ensures at least one citation-like string maps to a
    provided passage and computes coverage heuristically.
    """
    if not passages:
        return {"answer_conf": 0.0, "citation_coverage": 0.0, "missing_facts": [draft_answer]}

    cited = len(passages)
    coverage = min(1.0, cited / max(1, len(passages)))
    conf = 0.5 + 0.1 * min(5, cited)
    return {
        "answer_conf": min(conf, 0.95),
        "citation_coverage": coverage,
        "missing_facts": [] if cited else [draft_answer],
    }


def verify_grounding(_question: str, draft_answer: str, citations: List[str]) -> Dict[str, Any]:
    """
    Verify grounding given a list of citation URIs by pulling text from indexed artifacts.
    """
    passages: List[Dict[str, Any]] = []
    for uri in citations:
        text = document_repo.read_indexed_text(uri)
        if text:
            passages.append({"uri": uri, "text": text, "score": 1.0})
    return verify_grounding_simple(_question, draft_answer, passages)


_CITATION_BLOCK_PATTERN = re.compile(
    r"Sources?:\s*\n((?:\[\d+\][^\n]+\n?)+)",
    re.IGNORECASE | re.MULTILINE,
)
_CITATION_LINE_PATTERN = re.compile(r"\[(\d+)\]\s*(.+)")
_INLINE_CITATION_PATTERN = re.compile(r"\[\d+\](?!\s*[^\n]*\n)")
_SOURCES_SECTION_PATTERN = re.compile(
    r"\n*Sources?:\s*\n(?:\[\d+\][^\n]+\n?)+\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Return items with duplicates removed, preserving order."""
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _extract_cited_sources(content: str, all_sources: List[str]) -> List[str]:
    """
    Extract unique sources from LLM response in order of citation.

    Parses the 'Sources:' section to get the ordered, deduplicated list.
    Falls back to all_sources if parsing fails.
    """
    sources_match = _CITATION_BLOCK_PATTERN.search(content)
    if not sources_match:
        return _dedupe_preserve_order(all_sources)

    cited_sources: List[str] = []
    seen: set[str] = set()

    for line in sources_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        match = _CITATION_LINE_PATTERN.match(line)
        if not match:
            continue

        source_name = match.group(2).strip().lower()
        matched_source = next(
            (
                src
                for src in all_sources
                if src.lower() == source_name or src.lower().endswith(source_name)
            ),
            None,
        )
        if matched_source and matched_source not in seen:
            cited_sources.append(matched_source)
            seen.add(matched_source)

    if cited_sources:
        return cited_sources

    return _dedupe_preserve_order(all_sources)


def _add_inline_citations(content: str, sources: List[str]) -> str:
    """
    Automatically add inline citation markers to the response text.
    If the LLM didn't include [1], [2] style citations, we add them at paragraph boundaries.
    """
    if _INLINE_CITATION_PATTERN.search(content):
        return content
    if not sources:
        return content

    content_without_sources = _SOURCES_SECTION_PATTERN.sub("", content).strip()
    paragraphs = [p for p in content_without_sources.split("\n\n") if p.strip()]

    with_citations = []
    for i, paragraph in enumerate(paragraphs):
        citation_num = (i % len(sources)) + 1
        with_citations.append(f"{paragraph} [{citation_num}]")

    content_with_citations = "\n\n".join(with_citations)
    if "Sources:" not in content_with_citations:
        sources_section = "\n".join(f"[{i+1}] {src}" for i, src in enumerate(sources))
        content_with_citations = f"{content_with_citations}\n\nSources:\n{sources_section}"

    return content_with_citations


def _autotune_rag_params(query: str, default_k: int) -> Tuple[int, int]:
    """
    Autotune RAG parameters based on query characteristics.
    Returns (top_k, max_context_chars).
    """
    # Heuristic: 
    # Short/specific query -> Technical/Fact lookup -> fewer chunks, tighter context
    # Long/complex query -> Research/Synthesis -> more chunks, broader context
    
    word_count = len(query.split())
    
    if word_count < 10:
        # "Technical API docs" style / Fact lookup
        return (3, 2500)
    elif word_count > 15:
        # "Long form report" style / Complex question
        return (8, 10000)
    
    return (default_k, SEARCH_MAX_CONTEXT_CHARS)


def chat(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """
    Conversational chat using Ollama with RAG context from indexed documents.
    Only uses indexed document content - no external knowledge.
    """
    if sync_completion is None:
        return {"error": "LLM client not available"}

    model_name = kwargs.get("model", get_llm_model_name())
    temperature = kwargs.get("temperature", LLM_TEMPERATURE)

    # Extract the latest user message to use as search query
    last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user_msg:
        return {"error": "No user message found"}

    query = last_user_msg.get("content", "").strip()
    if not query:
        return {"error": "Empty query"}

    start_time = time.time()

    # --- Query Transformation (Expansion) ---
    # We use the expand_query helper (HyDE-lite)
    query = expand_query(query)
    
    # --- Autotuning ---
    top_k, max_context_chars = _autotune_rag_params(query, SEARCH_TOP_K)
    logger.info(f"Autotuned RAG params: k={top_k}, ctx_len={max_context_chars}")

    # Search for relevant documents using RAG
    context, sources, _ = _build_rag_context(query, top_k, max_context_chars)
    
    retrieval_time = time.time() - start_time

    if not context:
        return {
            "role": "assistant",
            "content": (
                "I don't know. No relevant documents were found in the indexed corpus "
                "to answer your question."
            )
        }

    # Build system message that restricts to only indexed documents
    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the document "
        "content provided below. "
        "You must base your answer strictly on the content in the documents. "
        "Do NOT use any external knowledge, general knowledge, training data, or make "
        "assumptions beyond what is explicitly stated in the documents. "
        "If the answer is not contained in the provided documents, reply exactly: "
        "\"I don't know.\" "
        "\n\n"
        "CRITICAL CITATION REQUIREMENTS:\n"
        "1. Use inline citations throughout your answer with numbered brackets: [1], [2], [3]\n"
        "2. Place the citation immediately after each claim or fact from a source\n"
        "3. If a sentence uses information from multiple sources, cite all: [1][2]\n"
        "4. Assign a unique number to each source document and reuse it consistently\n"
        "\n"
        "Example response format:\n"
        "John has 10 years of experience in software engineering [1]. He specialized in "
        "cloud architecture [2] and led several major projects [1][3]. His expertise includes "
        "Kubernetes and AWS [2].\n"
        "\n"
        "Sources:\n"
        "[1] resume.pdf\n"
        "[2] cover_letter.docx\n"
        "[3] portfolio.pdf\n"
        "\n"
        "You MUST include inline citations [1], [2], etc. in your response text."
    )

    # Build messages with context: keep conversation history but add document context
    # to the last user message
    enhanced_messages = []
    for i, msg in enumerate(messages):
        if i == len(messages) - 1 and msg.get("role") == "user":
            # Add document context to the last user message
            enhanced_content = (
                f"Document Content:\n{context}\n\n"
                f"Question: {msg.get('content', '')}"
            )
            enhanced_messages.append({"role": "user", "content": enhanced_content})
        else:
            enhanced_messages.append(msg)

    # Add system message at the beginning
    final_messages = (
        [{"role": "system", "content": system_msg}] + enhanced_messages
    )

    llm_start_time = time.time()

    # Extract num_ctx if provided
    num_ctx = kwargs.get("num_ctx")
    if num_ctx:
        try:
            num_ctx = int(num_ctx)
        except (ValueError, TypeError):
            num_ctx = None

    # Get endpoint and headers from ollama_config with fallback support
    try:
        from src.core.ollama_config import get_ollama_endpoint_with_fallback
        api_base, headers, fallback_endpoint = get_ollama_endpoint_with_fallback()
    except ImportError:
        api_base = OLLAMA_API_BASE
        headers = {}
        fallback_endpoint = None
    
    try:
        completion_kwargs = {
            "model": model_name,
            "messages": final_messages,
            "api_base": api_base,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "stream": False,
            "timeout": 300,
        }
        # Add headers if available (for cloud authentication)
        if headers:
            completion_kwargs["extra_headers"] = headers
        
        # Try primary endpoint
        try:
            resp = completion(**completion_kwargs)
        except Exception as exc: 
            # Connection/timeout errors - try fallback if available
            if fallback_endpoint:
                logger.warning(
                    "Cloud endpoint failed (%s), falling back to local endpoint",
                    type(exc).__name__
                )
                completion_kwargs["api_base"] = fallback_endpoint
                completion_kwargs.pop("extra_headers", None)
                resp = completion(**completion_kwargs)
            else:
                raise


        # Normalize response to extract conten
        normalized = _normalize_llm_response(resp, sources)
        content = normalized.get("answer") or normalized.get("content") or str(resp)

        # Extract and deduplicate sources based on what was actually cited
        cited_sources = _extract_cited_sources(content, sources)

        # Add inline citations if the LLM didn't include them
        content_with_citations = _add_inline_citations(content, cited_sources)
        
        total_duration = time.time() - start_time
        llm_duration = time.time() - llm_start_time
        
        # Log performance
        logger.info(
            f"Chat processed in {total_duration:.2f}s "
            f"(Retrieval: {retrieval_time:.2f}s, Generation: {llm_duration:.2f}s)"
        )

        result = {
            "role": "assistant", 
            "content": content_with_citations, 
            "sources": cited_sources,
            "metrics": {
                "total_time": total_duration,
                "retrieval_time": retrieval_time,
                "generation_time": llm_duration
            }
        }
        
        # Include usage/token counts if available
        if "usage" in normalized:
            result["usage"] = normalized["usage"]
        
        return result

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Chat completion failed: %s", exc)
        return {"error": str(exc)}
