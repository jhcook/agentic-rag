"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""
# type: ignore

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Tuple
from httpcore import RemoteProtocolError
import logging, pathlib, json, os, hashlib, time, asyncio

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

import numpy as np
from dotenv import load_dotenv  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from ollama import AsyncClient  # type: ignore
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

# Load environment variables from an optional .env file
load_dotenv()

# -------- Embedding model --------
# -------- Configuration --------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
DEBUG_MODE = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")
ASYNC_LLM_MODEL_NAME = os.getenv("ASYNC_LLM_MODEL_NAME", "llama3.2:1b")

def get_embedder() -> Optional[SentenceTransformer]:
    """Get the embedding model, loading it lazily if needed. Returns None in debug mode."""
    global _EMBEDDER
    
    if DEBUG_MODE:
        logger.info("Debug mode enabled - skipping embedding model loading")
        return None
        
    if _EMBEDDER is None:
        try:
            logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
            _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
        except OSError as exc:
            message = (
                f"Failed to load embedding model '{EMBED_MODEL_NAME}'. "
                "Confirm the identifier exists on Hugging Face or run `hf auth login` "
                "to provide credentials."
            )
            logger.critical(message)
            raise SystemExit(message) from exc
    return _EMBEDDER

def get_faiss_globals() -> tuple[List[np.ndarray], List[tuple[str, str]], Any, Dict[int, tuple[str, str]], int]:
    """Get FAISS-related globals, initializing them lazily if needed."""
    global _VECTORS, _META, _INDEX, _INDEX_TO_META, _EMBED_DIM
    
    if _VECTORS is None:
        _VECTORS = []
        _META = []
        
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
            
        logger.debug(f"Initialized FAISS globals with embedding dimension {_EMBED_DIM}")
    
    return _VECTORS, _META, _INDEX, _INDEX_TO_META, _EMBED_DIM

# Lazy-initialized global state - will be created when first needed
_EMBEDDER: Optional[SentenceTransformer] = None
_VECTORS: Optional[List[np.ndarray]] = None
_META: Optional[List[tuple[str, str]]] = None
_INDEX: Optional[Any] = None
_INDEX_TO_META: Optional[Dict[int, tuple[str, str]]] = None
_EMBED_DIM: Optional[int] = None

# Absolute path to the repository so relative inputs resolve consistently
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# -------- In-memory store (toy) --------
class Store:
    """Optimized document store - single source of truth for text."""
    def __init__(self):
        self.docs: Dict[str, str] = {}  # URI -> full text (single storage)
        self._last_loaded: float = 0.0  # For synchronization tracking
        logger.debug("Initialized new Store instance")

    def add(self, uri: str, text: str):
        """Add a document to the store."""
        logger.debug(f"Adding document: {uri}")
        self.docs[uri] = text
        logger.debug(f"Added document {uri}")
        logger.debug(f"Store now contains {len(self.docs)} documents")
        logger.debug(f"Store now contains {len(self.docs)} documents")

    def _ensure_last_loaded_exists(self):
        """Ensure _last_loaded attribute exists for compatibility."""
        if not hasattr(self, '_last_loaded'):
            self._last_loaded = 0.0


# Global store instance - lazily initialized
_STORE: Optional[Store] = None

def _ensure_store_synced():
    """Ensure the store is synchronized with disk if modified externally."""
    if not os.path.exists(DB_PATH):
        return
    
    try:
        # Check if the file was modified after our last load
        file_mtime = os.path.getmtime(DB_PATH)
        store = get_store()
        
        # Add a last_loaded timestamp to track when we last synced
        if not hasattr(store, '_last_loaded'):
            store._last_loaded = 0
            
        if file_mtime > store._last_loaded:
            logger.info("Detected external changes, reloading store from disk")
            load_store()
            store._last_loaded = time.time()
    except Exception as e:
        logger.warning(f"Error checking store sync: {e}")

def get_store() -> Store:
    """Return the global Store instance, creating it if necessary."""
    global _STORE
    if _STORE is None:
        _STORE = Store()
        # Try to load existing data
        try:
            load_store()
        except Exception as e:
            logger.debug(f"No existing store to load: {e}")
    return _STORE

# -------- Persist store --------
DB_PATH = os.getenv("RAG_DB", "./cache/rag_store.jsonl")

def _hash_uri(uri: str) -> str:
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
            logger.debug(f"Resolved path '{path}' to '{candidate}'")
            return candidate

    attempted = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Path '{path}' not found (tried: {attempted})")

def save_store():
    """Save the store to disk."""
    try:
        store = get_store()
        logger.info(f"Saving store to {DB_PATH}")
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
        logger.info(f"Successfully saved {len(get_store().docs)} documents")
        
        # Update last loaded timestamp to prevent unnecessary reloads
        get_store()._last_loaded = time.time()
    except Exception as e:
        logger.error(f"Error saving store: {str(e)}")
        raise

def _rebuild_faiss_index():
    """Rebuild the FAISS index from store documents."""
    global _STORE
    
    # Get FAISS globals lazily
    VECTORS, META, INDEX, INDEX_TO_META, EMBED_DIM = get_faiss_globals()
    embedder = get_embedder()
    
    logger.info("Rebuilding FAISS index from store documents")
    # Clear existing data
    VECTORS.clear()
    META.clear()
    INDEX_TO_META.clear()
    if INDEX is not None:
        INDEX.reset()  # type: ignore
    
    # Use _STORE directly to avoid circular calls
    if _STORE is None:
        logger.warning("No store available to rebuild index from")
        return
        
    faiss_index = 0
    
    # Process documents directly from store.docs (no redundant index)
    for uri, text in _STORE.docs.items():
        for chunk in _chunk(text):
            emb = embedder.encode(chunk, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
            emb_array = np.array(emb, dtype=np.float32)
            
            # Temporarily store for batch addition
            VECTORS.append(emb_array)
            META.append((uri, chunk))
            INDEX_TO_META[faiss_index] = (uri, chunk)
            faiss_index += 1
    
    if len(VECTORS) > 0 and INDEX is not None:
        v = np.vstack(VECTORS)
        INDEX.add(v)  # type: ignore
        logger.info(f"Added {len(VECTORS)} vectors to FAISS index")
        # Clear temporary storage to save memory
        VECTORS.clear()
    else:
        logger.warning("No vectors to add to FAISS index or FAISS not available")

def load_store():
    """Load the store from disk and rebuild FAISS index."""
    global _STORE
    
    if not os.path.exists(DB_PATH):
        logger.warning(f"Store file not found at {DB_PATH}")
        return
    
    try:
        logger.info(f"Loading store from {DB_PATH}")
        new_store = Store()
        with open(DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if "uri" in rec and "text" in rec:
                        new_store.add(rec["uri"], rec["text"])
                except json.JSONDecodeError as e:
                    logger.warning(f"load_store: {e}")
                    continue
        
        # Replace store contents (no redundant index to maintain)
        if _STORE is None:
            _STORE = Store()
        _STORE.docs.clear()
        _STORE.docs.update(new_store.docs)
        logger.info(f"Successfully loaded {len(_STORE.docs)} documents")
        
        # Set last loaded timestamp
        _STORE._last_loaded = time.time()
        
        # Rebuild FAISS index once after loading all documents
        _rebuild_faiss_index()
        
    except Exception as e:
        logger.error(f"Error loading store: {str(e)}")
        raise

def upsert_document(uri: str, text: str) -> Dict[str, Any]:
    """Upsert a single document into the store."""
    global _STORE
    
    # Get FAISS globals and embedder lazily
    VECTORS, META, INDEX, INDEX_TO_META, EMBED_DIM = get_faiss_globals()
    embedder = get_embedder()
    
    _ensure_store_synced()
    
    # Ensure store is initialized
    if _STORE is None:
        _STORE = Store()
        
    existed = uri in _STORE.docs
    _STORE.add(uri, text)
    
    if DEBUG_MODE:
        logger.debug(f"Debug mode: skipping embeddings for {uri}")
        save_store()
        return {"upserted": True, "existed": existed}
    
    # Add embeddings for new document
    base_index = INDEX.ntotal if INDEX is not None else 0  # type: ignore
    chunk_offset = 0
    for piece in _chunk(text):
        if embedder is not None:
            emb = embedder.encode(piece, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
            emb_array = np.array(emb, dtype=np.float32)
            VECTORS.append(emb_array)
            META.append((uri, piece))
            if INDEX is not None:
                INDEX.add(emb_array.reshape(1, -1))  # type: ignore
                INDEX_TO_META[base_index + chunk_offset] = (uri, piece)
                chunk_offset += 1
    
    save_store()
    total_vectors = INDEX.ntotal if INDEX is not None else 0  # type: ignore
    logger.info(f"Upserted document {uri} (existed: {existed}), index now has {total_vectors} vectors")
    return {"upserted": True, "existed": existed}

# -------- Retrieval / Index --------
def index_documents(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs."""
    # Get FAISS globals and embedder lazily
    VECTORS, META, INDEX, INDEX_TO_META, EMBED_DIM = get_faiss_globals()
    embedder = get_embedder()
    
    count = 0
    for uri in uris:
        try:
            text = pathlib.Path(uri).read_text(encoding="utf-8", errors="ignore")
            # Store document
            store = get_store()
            store.add(str(pathlib.Path(uri)), text)
            
            if not DEBUG_MODE and embedder is not None:
                # Add embeddings with proper index mapping
                base_index = INDEX.ntotal if INDEX is not None else 0  # type: ignore
                chunk_offset = 0
                for piece in _chunk(text):
                    emb = embedder.encode(piece, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
                    emb_array = np.array(emb, dtype=np.float32)
                    VECTORS.append(emb_array)
                    META.append((str(pathlib.Path(uri)), piece))
                    if INDEX is not None:
                        INDEX.add(emb_array.reshape(1, -1))  # type: ignore
                        INDEX_TO_META[base_index + chunk_offset] = (str(pathlib.Path(uri)), piece)
                        chunk_offset += 1
            elif DEBUG_MODE:
                logger.debug(f"Debug mode: skipping embeddings for {uri}")
            
            count += 1
        except Exception as e:
            logger.warning(f"Failed to index {uri}: {e}")
    
    save_store()
    index_total = INDEX.ntotal if INDEX is not None else 0  # type: ignore
    logger.info(f"Indexed {count} documents, index now has {index_total} vectors")
    return {"indexed": count}

def _collect_files(path: str, glob: str) -> Tuple[List[pathlib.Path], pathlib.Path]:
    """Collect files from the given path matching the glob pattern."""
    resolved = resolve_input_path(path)
    logger.debug(f"Collecting files from {resolved} with glob {glob}")
    if resolved.is_file():
        files = [resolved]
    else:
        files = list(resolved.rglob(glob))
    return files, resolved

def _read_and_store_files(files: List[pathlib.Path]) -> List[str]:
    """Read and store files into the STORE."""
    global _STORE
    
    # Get FAISS globals and embedder lazily
    VECTORS, META, INDEX, INDEX_TO_META, EMBED_DIM = get_faiss_globals()
    embedder = get_embedder()
    
    logger.debug(f"Reading and storing {len(files)} files")
    texts: List[str] = []
    
    # Ensure store is initialized
    if _STORE is None:
        _STORE = Store()
    
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            _STORE.add(str(fp), text)
            
            if not DEBUG_MODE and embedder is not None:
                # Add embeddings efficiently
                chunk_count = 0
                base_index = INDEX.ntotal if INDEX is not None else 0  # type: ignore
                
                for chunk in _chunk(text):
                    emb = embedder.encode(chunk, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
                    emb_array = np.array(emb, dtype=np.float32)
                    
                    if INDEX is not None:
                        INDEX.add(emb_array.reshape(1, -1))  # type: ignore
                        INDEX_TO_META[base_index + chunk_count] = (str(fp), chunk)
                        chunk_count += 1
            elif DEBUG_MODE:
                logger.debug(f"Debug mode: skipping embeddings for {fp}")
            
            texts.append(text)
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")
    
    save_store()
    index_total = INDEX.ntotal if INDEX is not None else 0  # type: ignore
    logger.info(f"Stored {len(texts)} files, index now has {index_total} vectors")
    return texts

def index_path(path: str, glob: str = "**/*.txt") -> Dict[str, Any]:
    """Index all text files in a given path matching the glob pattern."""
    # Get FAISS globals to access INDEX
    VECTORS, META, INDEX, INDEX_TO_META, EMBED_DIM = get_faiss_globals()
    
    try:
        files, resolved = _collect_files(path, glob)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return {"indexed": 0, "total_vectors": INDEX.ntotal if INDEX is not None else 0, "error": str(exc)}  # type: ignore

    if not files:
        message = f"No files matching '{glob}' were found under '{resolved}'"
        logger.info(message)
        return {
            "indexed": 0,
            "total_vectors": INDEX.ntotal if INDEX is not None else 0,  # type: ignore
            "error": message,
        }

    _read_and_store_files(files)
    
    index_total = INDEX.ntotal if INDEX is not None else 0  # type: ignore
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
    except (ValueError, APIConnectionError, RemoteProtocolError) as e:  # type: ignore
        logger.debug(f"send_to_llm: {e}")
        raise

def send_store_to_llm() -> str:
    """Send the entire store as context to the LLM for processing."""
    _ensure_store_synced()
    texts = list(get_store().docs.values())

    for _ in range(3):
        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                # Schedule the coroutine on the running loop and block until it finishes.
                # This is thread-safe and returns a concurrent.futures.Future.
                future = asyncio.run_coroutine_threadsafe(send_to_llm(texts), running_loop)
                resp = future.result(timeout=120)  # wait for completion (or raise)
            else:
                # No running loop: asyncio.run blocks until the coroutine completes.
                resp = asyncio.run(send_to_llm(texts))

        except APIConnectionError:
            time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"send_store_to_llm failed: {e}")
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
        if i >= j:  # prevent infinite loop
            break
    return out

def _vector_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Perform vector similarity search using FAISS."""
    # Get FAISS globals and embedder lazily
    VECTORS, META, INDEX, INDEX_TO_META, EMBED_DIM = get_faiss_globals()
    embedder = get_embedder()
    
    if INDEX is None or INDEX.ntotal == 0:  # type: ignore
        logger.debug("No FAISS index available for vector search")
        return []
    
    # Generate query embedding
    query_emb = embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)  # type: ignore
    query_array = np.array(query_emb, dtype=np.float32).reshape(1, -1)
    
    # Search FAISS index
    scores, indices = INDEX.search(query_array, min(k, INDEX.ntotal))  # type: ignore
    
    hits = []
    seen_uris = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx in INDEX_TO_META:
            uri, text = INDEX_TO_META[idx]
            # Deduplicate by URI - keep highest scoring chunk per document
            if uri not in seen_uris:
                hits.append({"score": float(score), "uri": uri, "text": text})
                seen_uris.add(uri)
    
    return hits

def search(query: str, top_k: int = 5, max_context_chars: int = 4000):
    """Search the indexed documents and ask the LLM using only those documents as context."""
    logger.info(f"Vector searching for: {query}")
    candidates = _vector_search(query, k=top_k)

    if not candidates:
        logger.info("No vector hits; refusing to answer from outside sources.")
        return {"error": "No relevant documents found in the indexed corpus."}

    # Build context, respecting char budget
    context_parts, total_chars = [], 0
    for i, entry in enumerate(candidates, start=1):
        part = f"--- Document {i} (uri: {entry['uri']}) ---\n{entry['text']}\n"
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
    user_msg = f"Documents:\n{context}\n\nQuestion: {query}"

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
    except (ValueError, OllamaError) as e:  # type: ignore
        logger.error(f"Ollama API Error: {e}")
        return {"error": str(e)}  # type: ignore
    except Exception as e:  # type: ignore
        logger.error(f"Unexpected error in completion: {e}")
        return {"error": str(e)}  # type: ignore
