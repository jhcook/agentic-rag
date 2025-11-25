"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""

from __future__ import annotations
import logging
import pathlib
import json
import os
import hashlib
import time
import asyncio
import gc

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from ollama import AsyncClient  # type: ignore

# Shared helpers
from src.core.embeddings import get_embedder as _get_embedder
from src.core.faiss_index import (
    get_faiss_globals as _get_faiss_globals,
    get_rebuild_lock,
)
from src.core.extractors import _extract_text_from_file

# Load .env early so configuration is available at module import time
load_dotenv()

# Optional dependencies
try:
    from tqdm import tqdm
except ImportError:
    def _tqdm_dummy(iterable=None, **_kwargs):  # type: ignore
        """Dummy tqdm for when tqdm is not installed."""
        return iterable
    tqdm = _tqdm_dummy

try:
    from litellm import completion  # type: ignore
    from litellm.exceptions import APIConnectionError  # type: ignore
    from litellm.llms.ollama.common_utils import OllamaError  # type: ignore
except ImportError:
    completion = None
    APIConnectionError = Exception
    OllamaError = Exception

# Enable debug logging if available
try:
    import litellm  # pylint: disable=unused-import
    os.environ['LITELLM_LOG'] = 'DEBUG'
except (ImportError, AttributeError):
    pass

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
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
DEBUG_MODE = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")
ASYNC_LLM_MODEL_NAME = os.getenv("ASYNC_LLM_MODEL_NAME", "llama3.2:1b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
DB_PATH = os.getenv("RAG_DB", "./cache/rag_store.jsonl")
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "12"))
SEARCH_MAX_CONTEXT_CHARS = int(os.getenv("SEARCH_MAX_CONTEXT_CHARS", "8000"))
EMBED_DIM_OVERRIDE = int(os.getenv("EMBED_DIM_OVERRIDE", "0")) or None

# Lazy-initialized global state
_STORE: Optional['Store'] = None  # pylint: disable=invalid-name

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


def get_faiss_globals() -> Tuple[Any, Dict[int, Tuple[str, int, int]], int]:
    """Get FAISS-related globals, initializing them lazily if needed."""
    embedder = get_embedder()
    embed_dim = 384
    if embedder is not None and not DEBUG_MODE:
        try:
            embed_dim = embedder.get_sentence_embedding_dimension()
        except Exception:  # pylint: disable=broad-exception-caught
            embed_dim = 384
    if EMBED_DIM_OVERRIDE:
        if embedder is not None and not DEBUG_MODE and embed_dim != EMBED_DIM_OVERRIDE:
            logger.warning(
                "EMBED_DIM_OVERRIDE=%s does not match embedder dimension %s; "
                "using embedder dimension",
                EMBED_DIM_OVERRIDE,
                embed_dim,
            )
        else:
            embed_dim = EMBED_DIM_OVERRIDE
    return _get_faiss_globals(embed_dim, DEBUG_MODE, logger)


# -------- In-memory store --------
@dataclass
class Store:
    """Optimized document store - single source of truth for text."""
    docs: Dict[str, str] = field(default_factory=dict)
    last_loaded: float = 0.0

    def add(self, uri: str, text: str) -> None:
        """Add a document to the store."""
        self.docs[uri] = text


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


def _ensure_store_synced():
    """Check and reload store if modified on disk."""
    if not os.path.exists(DB_PATH):
        return

    try:
        file_mtime = os.path.getmtime(DB_PATH)
        store = get_store()

        if file_mtime > store.last_loaded:
            logger.info("Detected external changes, reloading store from disk")
            load_store()
            store.last_loaded = time.time()
    except OSError as exc:
        logger.warning("Error checking store sync: %s", exc)


def get_store() -> Store:
    """Return the global Store instance, creating it if necessary."""
    global _STORE  # pylint: disable=global-statement
    if _STORE is None:
        _STORE = Store()
        try:
            load_store()
        except (OSError, ValueError) as exc:
            logger.debug("No existing store to load: %s", exc)
    return _STORE


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


def save_store():
    """Save the store to disk."""
    try:
        store = get_store()
        logger.info("Saving store to %s", DB_PATH)
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, "w", encoding="utf-8") as f:
            for uri, text in store.docs.items():
                rec: Dict[str, Any] = {
                    "uri": uri,
                    "id": _hash_uri(uri),
                    "text": text,
                    "ts": int(time.time())
                }
                f.write(json.dumps(rec) + "\n")
        logger.info("Successfully saved %d documents", len(get_store().docs))
        get_store().last_loaded = time.time()
    except (OSError, ValueError) as exc:
        logger.error("Error saving store: %s", str(exc))
        raise


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
    """Chunk text into pieces of max_chars with overlap."""
    return [c[0] for c in _chunk_text_with_offsets(text, max_chars, overlap)]

def _chunk_document_with_offsets(
    text: str, uri: str
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """Chunk a single document and return chunks, uris, and offsets."""
    chunks_with_offsets = _chunk_text_with_offsets(text)
    chunks = [c[0] for c in chunks_with_offsets]
    offsets = [(c[1], c[2]) for c in chunks_with_offsets]
    return chunks, [uri] * len(chunks), offsets

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


def _rebuild_faiss_index():  # pylint: disable=no-value-for-parameter
    """Rebuild the FAISS index from store documents."""
    with get_rebuild_lock():
        logger.info("Beginning FAISS rebuild")
        if DEBUG_MODE:
            logger.debug("Debug mode: skipping FAISS index rebuild")
            return

        index, index_to_meta, _ = get_faiss_globals()
        logger.info("FAISS globals ready (index=%s)", "present" if index is not None else "none")
        embedder = get_embedder()
        logger.info("Embedder ready")

        logger.info("Rebuilding FAISS index from store documents")
        index_to_meta.clear()
        if index is not None:
            index.reset()  # type: ignore

        if _STORE is None:
            logger.warning("No store available to rebuild index from")
            return

        logger.info("Processing %d documents for FAISS indexing", len(_STORE.docs))

        total_vectors = 0
        batch_size = 8

        # Process each document separately to avoid accumulating all chunks in memory
        doc_items = list(_STORE.docs.items())
        for uri, text in tqdm(iterable=doc_items, desc="Rebuilding FAISS index", unit="doc"):  # pylint: disable=no-value-for-parameter
            logger.info("Document %s: %d chars", uri, len(text))
            if not text or not text.strip():
                logger.warning("Document %s is empty, skipping", uri)
                continue
            chunks, uris, offsets = _chunk_document_with_offsets(text, uri)
            logger.info("Created %d chunks from %s", len(chunks), uri)

            if not chunks:
                logger.warning(
                    "No chunks created from document %s (text length: %d)", uri, len(text))
                continue

            # Process chunks from this document in small batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_uris = uris[i:i+batch_size]
                batch_offsets = offsets[i:i+batch_size]

                added = _process_batch_for_index(
                    embedder, index, index_to_meta,
                    batch_chunks, batch_uris, batch_offsets, uri
                )
                total_vectors += added

                # Force garbage collection after each batch
                del batch_chunks, batch_uris, batch_offsets
                gc.collect()

        if index is not None:
            logger.info("Added %d vectors to FAISS index", total_vectors)
        else:
            logger.warning("No FAISS index available")


def _process_batch_for_index(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    embedder: Optional[SentenceTransformer],
    index: Any,
    index_to_meta: Dict,
    batch_chunks: List[str],
    batch_uris: List[str],
    batch_offsets: List[Tuple[int, int]],
    uri: str
) -> int:
    """Process a batch of chunks for indexing."""
    if embedder is None:
        logger.error("Embedder is None, cannot create embeddings")
        return 0

    embeddings = _encode_with_metrics(
        embedder,
        batch_chunks,
        "rebuild_index",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if embeddings is None or len(embeddings) == 0:
        logger.warning("No embeddings generated for batch from %s", uri)
        return 0

    if index is not None:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        index.add(embeddings_array)  # type: ignore

        current_index = index.ntotal - len(batch_chunks)  # type: ignore
        for idx, (chunk_uri, (start, end)) in enumerate(zip(batch_uris, batch_offsets)):
            index_to_meta[current_index + idx] = (chunk_uri, start, end)

    return len(batch_chunks)


def load_store():
    """Load the store from disk and rebuild FAISS index."""
    global _STORE  # pylint: disable=global-statement

    if not os.path.exists(DB_PATH):
        logger.warning("Store file not found at %s", DB_PATH)
        return

    try:
        logger.info("Loading store from %s", DB_PATH)
        new_store = Store()
        with open(DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if "uri" in rec and "text" in rec:
                        new_store.add(rec["uri"], rec["text"])
                except json.JSONDecodeError as exc:
                    logger.warning("load_store: %s", exc)
                    continue

        if _STORE is None:
            _STORE = Store()
        _STORE.docs.clear()
        _STORE.docs.update(new_store.docs)
        logger.info("Successfully loaded %d documents", len(_STORE.docs))

        _STORE.last_loaded = time.time()

    except (OSError, ValueError) as exc:
        logger.error("Error loading store: %s", str(exc))
        raise


def upsert_document(uri: str, text: str) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """Upsert a single document into the store."""
    global _STORE  # pylint: disable=global-statement

    # Safety check for huge documents
    if len(text) > 5_000_000:  # 5MB text limi
        logger.warning("Document %s too large (%d chars), truncating.", uri, len(text))
        text = text[:5_000_000]

    if _should_skip_uri(uri):
        logger.warning("Skipping hidden/system file: %s", uri)
        return {"skipped": True, "reason": "hidden/system file", "uri": uri}

    index, index_to_meta, _ = get_faiss_globals()
    embedder = get_embedder()

    _ensure_store_synced()

    if _STORE is None:
        _STORE = Store()

    existed = uri in _STORE.docs
    _STORE.add(uri, text)

    if DEBUG_MODE:
        logger.debug("Debug mode: skipping embeddings for %s", uri)
        save_store()
        return {"upserted": True, "existed": existed}

    base_index = index.ntotal if index is not None else 0  # type: ignore
    chunks_with_offsets = _chunk_text_with_offsets(text)  # pylint: disable=no-value-for-parameter
    chunks = [c[0] for c in chunks_with_offsets]
    offsets = [(c[1], c[2]) for c in chunks_with_offsets]

    if embedder is not None and index is not None:
        # Process in batches to control memory
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_offsets = offsets[i:i+batch_size]
            embeddings = _encode_with_metrics(
                embedder,
                batch_chunks,
                "upsert_document",
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=batch_size,
            )
            embeddings = np.array(embeddings, dtype=np.float32)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            index.add(embeddings)

            # Update metadata for each chunk in batch
            for j, (start, end) in enumerate(batch_offsets):
                index_to_meta[base_index + i + j] = (uri, start, end)

            # Explicit cleanup
            del embeddings, batch_chunks
            gc.collect()

    save_store()
    total_vectors = index.ntotal if index is not None else 0  # type: ignore
    logger.info("Upserted document %s (existed: %s), index now has %d vectors",
                uri, existed, total_vectors)
    return {"upserted": True, "existed": existed}


def _collect_files(path: str, glob: str) -> Tuple[List[pathlib.Path], pathlib.Path]:
    """Collect files from the given path matching the glob pattern."""
    resolved = resolve_input_path(path)
    logger.debug("Collecting files from %s with glob %s", resolved, glob)
    if resolved.is_file():
        files = [resolved] if not _should_skip_uri(str(resolved)) else []
    else:
        files = [p for p in resolved.rglob(glob) if not _should_skip_uri(str(p))]
    return files, resolved


# _extract_text_from_file is now imported from extractors module
# Removed duplicate implementation (209 lines) - use extractors._extract_text_from_file instead


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
        if len(stripped) < 40:  # too shor
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

    text = _extract_text_from_file(file_path)
    if not _is_meaningful_text(text):
        logger.warning("Skipping non-text or empty content from %s", file_path)
        return ""

    store = get_store()
    store.add(str(file_path), text)

    if not DEBUG_MODE and embedder is not None and index is not None:
        base_index = index.ntotal  # type: ignore
        chunks_with_offsets = _chunk_text_with_offsets(text)  # pylint: disable=no-value-for-parameter
        chunks = [c[0] for c in chunks_with_offsets]
        offsets = [(c[1], c[2]) for c in chunks_with_offsets]

        # Process in batches to control memory
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_offsets = offsets[i:i+batch_size]
            embeddings = _encode_with_metrics(
                embedder,
                batch_chunks,
                "index_path",
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=batch_size,
            )
            embeddings = np.array(embeddings, dtype=np.float32)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            index.add(embeddings)

            # Update metadata for each chunk in batch
            for j, (start, end) in enumerate(batch_offsets):
                index_to_meta[base_index + i + j] = (str(file_path), start, end)

            # Explicit cleanup
            del embeddings, batch_chunks
            gc.collect()
    elif DEBUG_MODE:
        logger.debug("Debug mode: skipping embeddings for %s", file_path)

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
    index, index_to_meta, _ = get_faiss_globals()
    embedder = get_embedder()

    try:
        files, resolved = _collect_files(path, glob)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return {
            "indexed": 0,
            "total_vectors": index.ntotal if index is not None else 0,  # type: ignore
            "error": str(exc)
        }

    if not files:
        message = f"No files matching '{glob}' were found under '{resolved}'"
        logger.info(message)
        return {
            "indexed": 0,
            "total_vectors": index.ntotal if index is not None else 0,  # type: ignore
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
            text = _process_file(file_path, index, index_to_meta, embedder)
            texts.append(text)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)

    save_store()
    index_total = index.ntotal if index is not None else 0  # type: ignore
    logger.info("Stored %d files, index now has %d vectors", len(texts), index_total)

    return {"indexed": len(files), "total_vectors": index_total, "resolved_path": str(resolved)}


async def send_to_llm(query: List[str]) -> Any:
    """Send the query to the LLM and return the response."""
    client = AsyncClient(host=OLLAMA_API_BASE)
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


def send_store_to_llm() -> str:  # pylint: disable=too-many-locals
    """Send the entire store as context to the LLM for processing."""
    _ensure_store_synced()
    texts = list(get_store().docs.values())
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
    """Perform vector similarity search using FAISS."""
    index, index_to_meta, _ = get_faiss_globals()
    embedder = get_embedder()

    if index is None or index.ntotal == 0:  # type: ignore
        logger.debug("No FAISS index available for vector search")
        return []

    query_emb = _encode_with_metrics(
        embedder,
        query,
        "search_query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )  # type: ignore
    query_array = np.array(query_emb, dtype=np.float32).reshape(1, -1)

    scores, indices = index.search(query_array, min(k, index.ntotal))  # type: ignore  # pylint: disable=no-value-for-parameter

    hits = []
    store = get_store()
    for score, idx in zip(scores[0], indices[0]):
        if idx in index_to_meta:
            uri, start, end = index_to_meta[idx]
            full_text = store.docs.get(uri, "")
            if len(full_text) >= end:
                text = full_text[start:end]
            else:
                text = full_text[start:] # Fallback
            hits.append({"score": float(score), "uri": uri, "text": text})

    return hits


def _normalize_llm_response(resp: Any, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Normalize LLM response (ModelResponse or dict) to a consistent dict format.

    Args:
        resp: Response from LiteLLM (can be ModelResponse object or dict)
        sources: Optional list of source URIs to include

    Returns:
        Dict with 'answer' or 'content' key and optional 'sources'
    """
    if resp is None:
        return {"error": "No response from LLM"}

    # Handle ModelResponse objects from LiteLLM
    if hasattr(resp, 'choices') and resp.choices:
        # Extract content from ModelResponse
        message = getattr(resp.choices[0], 'message', None)
        content = getattr(message, 'content', None) if message else None
        result = {
            "answer": content or str(resp),
            "model": getattr(resp, 'model', 'unknown'),
        }
        if hasattr(resp, 'usage'):
            result["usage"] = getattr(resp, 'usage', {})
        if sources:
            result["sources"] = sources
        return result

    # Handle dict responses
    if isinstance(resp, dict):
        # Ensure sources are included
        if sources and "sources" not in resp:
            resp["sources"] = sources
        # Extract content if it's in choices forma
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
        # If we have docs but no vectors, force a rebuild once
        store = get_store()
        if len(getattr(store, "docs", {})) > 0:
            logger.info("No vector hits but store has docs; rebuilding index and retrying...")
            _rebuild_faiss_index()
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
           max_context_chars: int = SEARCH_MAX_CONTEXT_CHARS):
    """Search the indexed documents and ask the LLM using only those documents as context."""
    context, sources, candidates = _build_rag_context(query, top_k, max_context_chars)

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
        if completion is None:
            return {"error": "LiteLLM not available"}

        resp = completion(  # type: ignore
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            api_base=OLLAMA_API_BASE,
            temperature=LLM_TEMPERATURE,
            stream=False,
            timeout=300,
        )
        return _normalize_llm_response(resp, sources)
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

    model_name = kwargs.get("model", LLM_MODEL_NAME)
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
    if completion is None:
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

    try:
        resp = completion(  # type: ignore
            model=model_name,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            api_base=OLLAMA_API_BASE,
            temperature=temperature,
            num_ctx=num_ctx,
            stream=False,
            timeout=90,
        )
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
    Verify grounding given a list of citation URIs by pulling text from the store.
    """
    store = get_store()
    passages: List[Dict[str, Any]] = []
    for uri in citations:
        if uri in store.docs:
            passages.append({"uri": uri, "text": store.docs[uri], "score": 1.0})
    return verify_grounding_simple(_question, draft_answer, passages)


def _extract_cited_sources(content: str, all_sources: List[str]) -> List[str]:
    """
    Extract unique sources from LLM response in order of citation.
    Parses the 'Sources:' section to get the ordered, deduplicated list.
    Falls back to all_sources if parsing fails.
    
    Args:
        content: The LLM response text
        all_sources: All sources retrieved from RAG
        
    Returns:
        Ordered list of unique source URIs cited in the response
    """
    import re
    
    # Try to find the Sources: section
    sources_match = re.search(r'Sources?:\s*\n((?:\[\d+\][^\n]+\n?)+)', content, re.IGNORECASE | re.MULTILINE)
    
    if sources_match:
        sources_section = sources_match.group(1)
        # Extract source names from lines like "[1] filename.pdf"
        cited_sources = []
        seen = set()
        
        for line in sources_section.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Match pattern like "[1] document_name.pdf"
            match = re.match(r'\[(\d+)\]\s*(.+)', line)
            if match:
                source_name = match.group(2).strip()
                # Find this source in all_sources (case-insensitive match)
                for src in all_sources:
                    if src.lower() == source_name.lower() or src.lower().endswith(source_name.lower()):
                        if src not in seen:
                            cited_sources.append(src)
                            seen.add(src)
                        break
        
        if cited_sources:
            return cited_sources
    
    # Fallback: return deduplicated all_sources
    seen = set()
    result = []
    for src in all_sources:
        if src not in seen:
            result.append(src)
            seen.add(src)
    return result


def _add_inline_citations(content: str, sources: List[str]) -> str:
    """
    Automatically add inline citation markers to the response text.
    If the LLM didn't include [1], [2] style citations, we add them at paragraph boundaries.
    
    Args:
        content: The LLM response text
        sources: List of unique sources
        
    Returns:
        Content with inline citations added
    """
    import re
    
    # Check if content already has inline citations
    has_inline = bool(re.search(r'\[\d+\](?!\s*[^\n]*\n)', content))  # [N] not at start of line
    if has_inline:
        return content  # Already has inline citations, leave as is
    
    # Remove the Sources: section if it exists (we'll add our own)
    content_without_sources = re.sub(
        r'\n*Sources?:\s*\n(?:\[\d+\][^\n]+\n?)+\s*$',
        '',
        content,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    
    if not sources:
        return content_without_sources
    
    # Split content into paragraphs
    paragraphs = content_without_sources.split('\n\n')
    
    # Add citation at the end of each substantial paragraph
    # Distribute citations across paragraphs
    modified_paragraphs = []
    source_idx = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            modified_paragraphs.append(para)
            continue
            
        # For substantial paragraphs (more than just a short sentence), add citations
        if len(para) > 50 and source_idx < len(sources):
            # Cycle through sources
            cite_num = (source_idx % len(sources)) + 1
            # Add citation at end of paragraph
            if not para.endswith('.'):
                para += '.'
            para += f" [{cite_num}]"
            source_idx += 1
        
        modified_paragraphs.append(para)
    
    return '\n\n'.join(modified_paragraphs)


def chat(messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """
    Conversational chat using Ollama with RAG context from indexed documents.
    Only uses indexed document content - no external knowledge.
    """
    if completion is None:
        return {"error": "LiteLLM not available"}

    model_name = kwargs.get("model", LLM_MODEL_NAME)
    temperature = kwargs.get("temperature", LLM_TEMPERATURE)

    # Extract the latest user message to use as search query
    last_user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user_msg:
        return {"error": "No user message found"}

    query = last_user_msg.get("content", "").strip()
    if not query:
        return {"error": "Empty query"}

    # Search for relevant documents using RAG
    context, sources, _ = _build_rag_context(query, SEARCH_TOP_K, SEARCH_MAX_CONTEXT_CHARS)

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

    # Extract num_ctx if provided
    num_ctx = kwargs.get("num_ctx")
    if num_ctx:
        try:
            num_ctx = int(num_ctx)
        except (ValueError, TypeError):
            num_ctx = None

    try:
        resp = completion(
            model=model_name,
            messages=final_messages,
            api_base=OLLAMA_API_BASE,
            temperature=temperature,
            num_ctx=num_ctx,
            stream=False,
            timeout=300,
        )

        # Normalize response to extract conten
        normalized = _normalize_llm_response(resp, sources)
        content = normalized.get("answer") or normalized.get("content") or str(resp)
        
        # Extract and deduplicate sources based on what was actually cited
        cited_sources = _extract_cited_sources(content, sources)
        
        # Add inline citations if the LLM didn't include them
        content_with_citations = _add_inline_citations(content, cited_sources)
        
        return {"role": "assistant", "content": content_with_citations, "sources": cited_sources}

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Chat completion failed: %s", exc)
        return {"error": str(exc)}
