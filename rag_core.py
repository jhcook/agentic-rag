"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from httpcore import RemoteProtocolError
import logging, pathlib, json, os, hashlib, time, asyncio

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import AsyncClient
from litellm import completion, _turn_on_debug
from litellm.exceptions import APIConnectionError
from litellm.llms.ollama.common_utils import OllamaError
_turn_on_debug()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- Embedding model --------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)

# FAISS index (empty for now)
VECTORS: List[np.ndarray] = []
META: List[tuple[str, str]] = []
# determine embedding dimension from the sentence-transformers model and handle empty vectors
EMBED_DIM = EMBEDDER.get_sentence_embedding_dimension()
# cosine if vectors are L2-normalized or explicitly normalized
INDEX = faiss.IndexFlatIP(EMBED_DIM)

# -------- In-memory store (toy) --------
class Store:
    """A simple in-memory document store."""
    def __init__(self):
        self.docs: Dict[str, str] = {}
        self.index: List[Dict[str, Any]] = []
        logger.info("Initialized new Store instance")

    def add(self, uri: str, text: str):
        """Add a document to the store and index."""
        logger.debug(f"Adding document: {uri}")
        self.docs[uri] = text
        
        # Clear existing index entries for this URI
        self.index = [idx for idx in self.index if idx["uri"] != uri]
        
        # Create index entry with preprocessed terms
        terms = {word.lower() for word in text.split() 
                if word.isalpha() and len(word) > 1}
        
        # Add new index entry
        index_entry = {
            "uri": uri,
            "text": text,
            "terms": terms
        }
        self.index.append(index_entry)
        
        logger.debug(f"Added document {uri} with {len(terms)} terms")
        logger.debug(f"Store now contains {len(self.docs)} documents")


STORE = Store()

def get_store() -> Store:
    """Return the global Store instance."""
    return STORE

# -------- Persist store --------
DB_PATH = os.getenv("RAG_DB", "./cache/rag_store.jsonl")

def _hash_uri(uri: str) -> str:
    return hashlib.sha1(uri.encode()).hexdigest()

def save_store():
    """Save the store to disk."""
    try:
        logger.info(f"Saving store to {DB_PATH}")
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, "w", encoding="utf-8") as f:
            for uri, text in STORE.docs.items():
                rec = {
                    "uri": uri,
                    "id": _hash_uri(uri),
                    "text": text,
                    "ts": int(time.time())
                }
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Successfully saved {len(STORE.docs)} documents")
    except Exception as e:
        logger.error(f"Error saving store: {str(e)}")
        raise

def _rebuild_faiss_index():
    """Rebuild the FAISS index from STORE documents."""
    global VECTORS, META, INDEX
    
    logger.info("Rebuilding FAISS index from store")
    VECTORS, META = [], []
    INDEX.reset()
    
    for entry in STORE.index:
        for piece in _chunk(entry["text"]):
            emb = EMBEDDER.encode(piece, normalize_embeddings=True)
            VECTORS.append(emb)
            META.append((entry["uri"], piece))
    
    if len(VECTORS) > 0:
        v = np.vstack(VECTORS).astype("float32")
        INDEX.add(v)
        logger.info(f"Added {len(VECTORS)} vectors to FAISS index")
    else:
        logger.warning("No vectors to add to FAISS index")

def load_store():
    """Load the store from disk and rebuild FAISS index."""
    global STORE
    
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
        
        STORE = new_store
        logger.info(f"Successfully loaded {len(STORE.docs)} documents")
        
        # Rebuild FAISS index once after loading all documents
        _rebuild_faiss_index()
        
    except Exception as e:
        logger.error(f"Error loading store: {str(e)}")
        raise

def upsert_document(uri: str, text: str) -> dict:
    """Upsert a single document into the store."""
    existed = uri in STORE.docs
    STORE.add(uri, text)
    
    # Add embeddings for new document
    for piece in _chunk(text):
        emb = EMBEDDER.encode(piece, normalize_embeddings=True)
        VECTORS.append(emb)
        META.append((uri, piece))
        INDEX.add(emb.reshape(1, -1).astype("float32"))
    
    save_store()
    logger.info(f"Upserted document {uri} (existed: {existed}), index now has {INDEX.ntotal} vectors")
    return {"upserted": True, "existed": existed}

# -------- Retrieval / Index --------
def index_documents(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs."""
    count = 0
    for uri in uris:
        try:
            text = pathlib.Path(uri).read_text(encoding="utf-8", errors="ignore")
            STORE.add(str(pathlib.Path(uri)), text)
            
            # Add embeddings
            for piece in _chunk(text):
                emb = EMBEDDER.encode(piece, normalize_embeddings=True)
                VECTORS.append(emb)
                META.append((str(pathlib.Path(uri)), piece))
                INDEX.add(emb.reshape(1, -1).astype("float32"))
            
            count += 1
        except Exception as e:
            logger.warning(f"Failed to index {uri}: {e}")
    
    save_store()
    logger.info(f"Indexed {count} documents, index now has {INDEX.ntotal} vectors")
    return {"indexed": count}

def _collect_files(path: str, glob: str) -> List[pathlib.Path]:
    """Collect files from the given path matching the glob pattern."""
    logger.debug(f"Collecting files from {path} with glob {glob}")
    p = pathlib.Path(path)
    return list(p.rglob(glob))

def _read_and_store_files(files: List[pathlib.Path]) -> List[str]:
    """Read and store files into the STORE."""
    logger.debug(f"Reading and storing {len(files)} files")
    texts = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            STORE.add(str(fp), text)
            
            # Add embeddings
            for piece in _chunk(text):
                emb = EMBEDDER.encode(piece, normalize_embeddings=True)
                VECTORS.append(emb)
                META.append((str(fp), piece))
                INDEX.add(emb.reshape(1, -1).astype("float32"))
            
            texts.append(text)
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")
    
    save_store()
    logger.info(f"Stored {len(texts)} files, index now has {INDEX.ntotal} vectors")
    return texts

def index_path(path: str, glob: str = "**/*.txt") -> Optional[Any]:
    """Index all text files in a given path matching the glob pattern."""
    files = _collect_files(path, glob)
    texts = _read_and_store_files(files)
    
    return {"indexed": len(files), "total_vectors": INDEX.ntotal}

async def send_to_llm(query: List[str]) -> str:
    """Send the query to the LLM and return the response."""
    client = AsyncClient(host="http://127.0.0.1:11434")
    messages = [{"content": f"{text}", "role": "user"} for text in query]
    try:
        resp = await client.chat(
            model="llama3.2:3b",
            messages=messages
        )
        return resp
    except (ValueError, APIConnectionError, RemoteProtocolError) as e:
        logger.debug(f"send_to_llm: {e}")
        raise

def send_store_to_llm():
    """Send STORE to LLM for processing, waiting for completion whether an
    event loop is running or not.
    """
    logger.debug("Loading STORE")
    resp = None
    texts = list(STORE.docs.values())

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
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j])
        i = j - overlap
        if i >= j:  # prevent infinite loop
            break
    return out

def _vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform vector search using FAISS index."""
    if INDEX.ntotal == 0:
        logger.warning("FAISS index is empty")
        return []
    
    qv = EMBEDDER.encode(query, normalize_embeddings=True).astype("float32")
    # Over-fetch for potential re-ranking/deduplication
    k = min(top_k * 4, INDEX.ntotal)
    D, I = INDEX.search(qv.reshape(1, -1), k)
    
    hits = []
    seen_uris = set()
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        uri, text = META[idx]
        
        # Deduplicate by URI - keep highest scoring chunk per document
        if uri not in seen_uris:
            hits.append({"score": float(score), "uri": uri, "text": text})
            seen_uris.add(uri)
            
        if len(hits) >= top_k:
            break
    
    return hits

def search(query: str, top_k: int = 5, max_context_chars: int = 4000):
    """Search the indexed documents and ask the LLM using only those documents as context."""
    logger.info(f"Vector searching for: {query}")
    candidates = _vector_search(query, top_k=top_k)

    if not candidates:
        logger.info("No vector hits; refusing to answer from outside sources.")
        return {"error": "No relevant documents found in the indexed corpus."}

    # Build context, respecting char budget
    context_parts, total_chars = [], 0
    for i, entry in enumerate(candidates, start=1):
        part = f"--- Document {i} (uri: {entry['uri']}) ---\n{entry['text']}\n"
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        total_chars += len(part)
    context = "\n".join(context_parts)

    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the documents provided. "
        "If the answer is not contained in the documents, reply exactly: \"I don't know.\" Do not use "
        "any external knowledge or make assumptions."
    )
    user_msg = f"Documents:\n{context}\n\nQuestion: {query}"

    try:
        resp = completion(
            model="ollama/llama3.2:3b",
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            api_base="http://localhost:11434",
            stream=False,
            timeout=120,
        )
        return resp
    except (ValueError, OllamaError) as e:
        logger.error(f"Ollama API Error: {e}")
        return {"error": str(e)}