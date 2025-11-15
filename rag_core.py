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

from typing import List, Dict, Any, Optional, Tuple
from httpcore import RemoteProtocolError

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

import numpy as np
from dotenv import load_dotenv  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from ollama import AsyncClient  # type: ignore

# Load .env early so configuration is available at module import time
load_dotenv()

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
    import litellm  # type: ignore
    litellm.set_verbose = True  # type: ignore
except (ImportError, AttributeError):
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- Configuration --------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
DEBUG_MODE = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")
ASYNC_LLM_MODEL_NAME = os.getenv("ASYNC_LLM_MODEL_NAME", "llama3.2:1b")
DB_PATH = os.getenv("RAG_DB", "./cache/rag_store.jsonl")

# Lazy-initialized global state
_EMBEDDER: Optional[SentenceTransformer] = None
_INDEX: Optional[Any] = None
_INDEX_TO_META: Optional[Dict[int, Tuple[str, str]]] = None
_EMBED_DIM: Optional[int] = None
_STORE: Optional['Store'] = None

# Absolute path to the repository so relative inputs resolve consistently
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent


def get_embedder() -> Optional[SentenceTransformer]:
    """Get the embedding model, loading it lazily if needed. Returns None in debug mode."""
    global _EMBEDDER

    if DEBUG_MODE:
        logger.info("Debug mode enabled - skipping embedding model loading")
        return None

    if _EMBEDDER is None:
        try:
            logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
            _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
        except OSError as exc:
            message = (
                "Failed to load embedding model '%s'. "
                "Confirm the identifier exists on Hugging Face or run `hf auth login` "
                "to provide credentials."
            ) % EMBED_MODEL_NAME
            logger.critical(message)
            raise SystemExit(message) from exc
    return _EMBEDDER


def get_faiss_globals() -> Tuple[Any, Dict[int, Tuple[str, str]], int]:
    """Get FAISS-related globals, initializing them lazily if needed."""
    global _INDEX, _INDEX_TO_META, _EMBED_DIM

    if _INDEX is None:
        if DEBUG_MODE:
            logger.info("Debug mode enabled - skipping FAISS and embedding initialization")
            _EMBED_DIM = 384  # Dummy dimension
            _INDEX = None
            _INDEX_TO_META = {}
        else:
            # Initialize embedding dimension
            embedder = get_embedder()
            if embedder is not None:
                _EMBED_DIM = embedder.get_sentence_embedding_dimension()

                # Initialize FAISS index
                if faiss is not None:
                    _INDEX = faiss.IndexFlatIP(_EMBED_DIM)  # type: ignore
                    _INDEX_TO_META = {}
                else:
                    _INDEX = None  # type: ignore
                    _INDEX_TO_META = {}
            else:
                _EMBED_DIM = 384  # Default dimension
                _INDEX = None
                _INDEX_TO_META = {}

        logger.debug("Initialized FAISS globals with embedding dimension %s", _EMBED_DIM)

    return _INDEX, _INDEX_TO_META, _EMBED_DIM


# -------- In-memory store --------
class Store:
    """Optimized document store - single source of truth for text."""
    def __init__(self):
        self.docs: Dict[str, str] = {}
        self.last_loaded: float = 0.0
        logger.debug("Initialized new Store instance")

    def add(self, uri: str, text: str):
        """Add a document to the store."""
        logger.debug("Adding document: %s", uri)
        self.docs[uri] = text
        logger.debug("Store now contains %d documents", len(self.docs))


def _ensure_store_synced():
    """Ensure the store is synchronized with disk if modified externally."""
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
    global _STORE
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
    raise FileNotFoundError("Path '%s' not found (tried: %s)" % (path, attempted))


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


def _chunk_document(text: str, uri: str) -> Tuple[List[str], List[str]]:
    """Chunk a single document and return chunks with metadata."""
    all_chunks = []
    chunk_metadata = []

    max_chars = 800
    overlap = 120
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


def _rebuild_faiss_index():
    """Rebuild the FAISS index from store documents."""
    # Prevent recursive calls
    if hasattr(_rebuild_faiss_index, 'in_progress') and _rebuild_faiss_index.in_progress:
        logger.warning("_rebuild_faiss_index already in progress, skipping recursive call")
        return

    _rebuild_faiss_index.in_progress = True

    try:
        if DEBUG_MODE:
            logger.debug("Debug mode: skipping FAISS index rebuild")
            return

        index, index_to_meta, _ = get_faiss_globals()
        embedder = get_embedder()

        logger.info("Rebuilding FAISS index from store documents")
        index_to_meta.clear()
        if index is not None:
            index.reset()  # type: ignore

        if _STORE is None:
            logger.warning("No store available to rebuild index from")
            return

        all_chunks = []
        chunk_metadata = []

        logger.info("Processing %d documents for FAISS indexing", len(_STORE.docs))

        for uri, text in _STORE.docs.items():
            logger.debug("Document %s: %d chars", uri, len(text))
            chunks, metadata = _chunk_document(text, uri)
            all_chunks.extend(chunks)
            chunk_metadata.extend(metadata)
            logger.debug("Created %d chunks from %s", len(chunks), uri)

        if not all_chunks:
            logger.warning("No chunks to index")
            return

        logger.info("Encoding %d chunks in batch", len(all_chunks))

        embeddings = embedder.encode(
            all_chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )  # type: ignore

        if index is not None:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            index.add(embeddings_array)  # type: ignore

            for idx, (uri, chunk) in enumerate(zip(chunk_metadata, all_chunks)):
                index_to_meta[idx] = (uri, chunk)

            logger.info("Added %d vectors to FAISS index", len(all_chunks))
        else:
            logger.warning("No FAISS index available")

    finally:
        _rebuild_faiss_index.in_progress = False


def load_store():
    """Load the store from disk and rebuild FAISS index."""
    global _STORE

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
        _rebuild_faiss_index()

    except (OSError, ValueError) as exc:
        logger.error("Error loading store: %s", str(exc))
        raise


def upsert_document(uri: str, text: str) -> Dict[str, Any]:
    """Upsert a single document into the store."""
    global _STORE

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
    chunk_offset = 0
    for piece in _chunk(text):
        if embedder is not None:
            emb = embedder.encode(piece, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
            emb_array = np.array(emb, dtype=np.float32)
            if index is not None:
                index.add(emb_array.reshape(1, -1))  # type: ignore
                index_to_meta[base_index + chunk_offset] = (uri, piece)
                chunk_offset += 1

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
        files = [resolved]
    else:
        files = list(resolved.rglob(glob))
    return files, resolved


def _process_file(file_path: pathlib.Path, index: Any, index_to_meta: Dict,
                  embedder: Optional[SentenceTransformer]) -> str:
    """Process a single file and add to index."""
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    store = get_store()
    store.add(str(file_path), text)

    if not DEBUG_MODE and embedder is not None:
        chunk_count = 0
        base_index = index.ntotal if index is not None else 0  # type: ignore

        for chunk in _chunk(text):
            emb = embedder.encode(chunk, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
            emb_array = np.array(emb, dtype=np.float32)

            if index is not None:
                index.add(emb_array.reshape(1, -1))  # type: ignore
                index_to_meta[base_index + chunk_count] = (str(file_path), chunk)
                chunk_count += 1
    elif DEBUG_MODE:
        logger.debug("Debug mode: skipping embeddings for %s", file_path)

    return text


def index_path(path: str, glob: str = "**/*.txt") -> Dict[str, Any]:
    """Index all text files in a given path matching the glob pattern."""
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
        message = "No files matching '%s' were found under '%s'" % (glob, resolved)
        logger.info(message)
        return {
            "indexed": 0,
            "total_vectors": index.ntotal if index is not None else 0,  # type: ignore
            "error": message,
        }

    texts = []
    for file_path in files:
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
    messages = [{"content": f"{text}", "role": "user"} for text in query]
    try:
        resp = await client.chat(  # type: ignore
            model=ASYNC_LLM_MODEL_NAME,
            messages=messages
        )
        return resp
    except (ValueError, APIConnectionError, RemoteProtocolError) as exc:  # type: ignore
        logger.debug("send_to_llm: %s", exc)
        raise


def send_store_to_llm() -> str:
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
                resp = future.result(timeout=120)
            else:
                resp = asyncio.run(send_to_llm(texts))

        except APIConnectionError:
            time.sleep(1)
            continue
        except (OSError, ValueError) as exc:
            logger.error("send_store_to_llm failed: %s", exc)
            raise
        break

    return resp


def _chunk(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    """Chunk text into pieces of max_chars with overlap."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j])
        i = j - overlap
        if i >= j:
            break
    return out


def _vector_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Perform vector similarity search using FAISS."""
    index, index_to_meta, _ = get_faiss_globals()
    embedder = get_embedder()

    if index is None or index.ntotal == 0:  # type: ignore
        logger.debug("No FAISS index available for vector search")
        return []

    query_emb = embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
    query_array = np.array(query_emb, dtype=np.float32).reshape(1, -1)

    scores, indices = index.search(query_array, min(k, index.ntotal))  # type: ignore

    hits = []
    seen_uris = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx in index_to_meta:
            uri, text = index_to_meta[idx]
            if uri not in seen_uris:
                hits.append({"score": float(score), "uri": uri, "text": text})
                seen_uris.add(uri)

    return hits


def search(query: str, top_k: int = 5, max_context_chars: int = 4000):
    """Search the indexed documents and ask the LLM using only those documents as context."""
    logger.info("Vector searching for: %s", query)
    candidates = _vector_search(query, k=top_k)

    if not candidates:
        logger.info("No vector hits; refusing to answer from outside sources.")
        return {"error": "No relevant documents found in the indexed corpus."}

    context_parts, total_chars = [], 0
    for i, entry in enumerate(candidates, start=1):
        part = "--- Document %d (uri: %s) ---\n%s\n" % (i, entry['uri'], entry['text'])
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)  # type: ignore
        total_chars += len(part)
    context = "\n".join(context_parts)  # type: ignore

    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the documents provided. "
        "If the answer is not contained in the documents, reply exactly: \"I don't know.\" Do not use "
        "any external knowledge or make assumptions."
    )
    user_msg = "Documents:\n%s\n\nQuestion: %s" % (context, query)

    try:
        if completion is None:
            return {"error": "LiteLLM not available"}

        resp = completion(  # type: ignore
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            api_base=OLLAMA_API_BASE,
            stream=False,
            timeout=120,
        )
        return resp
    except (ValueError, OllamaError) as exc:  # type: ignore
        logger.error("Ollama API Error: %s", exc)
        return {"error": str(exc)}  # type: ignore
    except (OSError, RuntimeError) as exc:  # type: ignore
        logger.error("Unexpected error in completion: %s", exc)
        return {"error": str(exc)}  # type: ignore
