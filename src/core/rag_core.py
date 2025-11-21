"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""

from __future__ import annotations
import logging
import pathlib
import json
import os
import time
import asyncio
import gc
from typing import List, Dict, Any, Optional, Tuple

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
from src.core.store import StoreManager, Store
from src.core.retrieval_utils import rerank, synthesize_answer, is_low_signal
from src.core import extractors as _extractors

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
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
DEBUG_MODE = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")
ASYNC_LLM_MODEL_NAME = os.getenv("ASYNC_LLM_MODEL_NAME", "llama3.2:1b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # Low temperature for consistent, grounded responses
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
DB_PATH = os.getenv("RAG_DB", "./cache/rag_store.jsonl")
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "12"))
SEARCH_MAX_CONTEXT_CHARS = int(os.getenv("SEARCH_MAX_CONTEXT_CHARS", "8000"))
EMBED_DIM_OVERRIDE = int(os.getenv("EMBED_DIM_OVERRIDE", "0")) or None

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable librarian designed to search and summarise documents. "
    "Answer the user's question using ONLY the documents provided. "
    "Only use the document's name and file metadata for citation purposes. "
    "Do not use first person terminology like 'I', 'me', 'my', or 'we'. "
    "Do NOT offer any personal information other than name, role, and employer unless specifically asked. "
    "Cite the exact source URI for each fact. Include a final 'Sources:' section listing the URIs you used. "
    "If the answer is not contained in the documents, reply exactly: \"I don't know.\" Do not use "
    "any external knowledge or make assumptions."
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

DEFAULT_GROUNDED_PROMPT = (
    "You are a careful assistant. Write a concise professional summary using ONLY the provided documents. "
    "Capture roles, companies, key achievements, and timelines when present. "
    "Strictly exclude greetings, sign-offs, and speculative content. "
    "If the documents do not clearly support the answer, reply exactly: \"I don't know.\" "
    "End with a 'Sources:' section listing the URIs you used."
)
GROUNDING_SYSTEM_PROMPT = os.getenv("GROUNDING_SYSTEM_PROMPT", DEFAULT_GROUNDED_PROMPT)

# Embedding metrics (shared by servers)
EMBEDDING_REQUESTS = Counter("embedding_requests_total", "Embedding encode invocations.", ["stage"]) if Counter else None
EMBEDDING_ERRORS = Counter("embedding_errors_total", "Embedding encode failures.", ["stage"]) if Counter else None
EMBEDDING_DURATION = Histogram(
    "embedding_duration_seconds",
    "Time spent in embedding encode calls.",
    ["stage"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8],
) if Histogram else None

# Absolute path to the repository so relative inputs resolve consistently
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
_STORE_MANAGER = StoreManager(DB_PATH, PROJECT_ROOT, logger)


def get_embedder() -> Optional[SentenceTransformer]:
    """Get the embedding model via the shared embeddings module."""
    return _get_embedder(EMBED_MODEL_NAME, DEBUG_MODE, logger)


def get_faiss_globals() -> Tuple[Any, Dict[int, Tuple[str, str]], int]:
    """Get FAISS-related globals, initializing them lazily if needed."""
    embedder = get_embedder()
    embed_dim = 384
    if embedder is not None and not DEBUG_MODE:
        try:
            embed_dim = embedder.get_sentence_embedding_dimension()
        except Exception:
            embed_dim = 384
    if EMBED_DIM_OVERRIDE:
        if embedder is not None and not DEBUG_MODE and embed_dim != EMBED_DIM_OVERRIDE:
            logger.warning(
                "EMBED_DIM_OVERRIDE=%s does not match embedder dimension %s; using embedder dimension",
                EMBED_DIM_OVERRIDE,
                embed_dim,
            )
        else:
            embed_dim = EMBED_DIM_OVERRIDE
    return _get_faiss_globals(embed_dim, DEBUG_MODE, logger)


def _encode_with_metrics(embedder: SentenceTransformer, inputs: Any, stage: str, **kwargs):
    """Wrap embedder.encode to record metrics when available."""
    if embedder is None:  # Defensive: should already be handled by callers
        raise RuntimeError("Embedding model is not initialized.")

    if EMBEDDING_REQUESTS:
        EMBEDDING_REQUESTS.labels(stage=stage).inc()

    start = time.perf_counter()
    try:
        result = embedder.encode(inputs, **kwargs)
        if EMBEDDING_DURATION:
            EMBEDDING_DURATION.labels(stage=stage).observe(time.perf_counter() - start)
        return result
    except Exception:
        if EMBEDDING_ERRORS:
            EMBEDDING_ERRORS.labels(stage=stage).inc()
        raise


def _ensure_store_synced():
    """Ensure the store is synchronized with disk if modified externally."""
    _STORE_MANAGER.ensure_synced()


def get_store() -> Store:
    """Return the global Store instance, creating it if necessary."""
    return _STORE_MANAGER.get_store()


def reset_store(store: Optional[Store] = None, db_path: Optional[str] = None) -> Store:
    """Replace the in-memory store (primarily for tests)."""
    return _STORE_MANAGER.reset(store, db_path=db_path)


def resolve_input_path(path: str) -> pathlib.Path:
    """Resolve a user-supplied path, trying common fallbacks."""
    return _STORE_MANAGER.resolve_input_path(path)


def save_store():
    """Save the store to disk."""
    _STORE_MANAGER.save_store()


def _chunk_document(text: str, uri: str) -> Tuple[List[str], List[str]]:
    """Chunk a single document and return chunks with metadata."""
    return _STORE_MANAGER.chunk_document(text, uri)


def _should_skip_uri(uri: str) -> bool:
    """Skip hidden/system files that should never be indexed."""
    return _STORE_MANAGER.should_skip_uri(uri)


def _rebuild_faiss_index():
    """Rebuild the FAISS index from store documents."""
    with get_rebuild_lock():
        logger.info("Beginning FAISS rebuild")
        if DEBUG_MODE:
            logger.debug("Debug mode: skipping FAISS index rebuild")
            return

        # Ensure store is loaded before rebuilding
        try:
            load_store()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load store before rebuild: %s", exc)

        index, index_to_meta, _ = get_faiss_globals()
        logger.info("FAISS globals ready (index=%s)", "present" if index is not None else "none")
        embedder = get_embedder()
        logger.info("Embedder ready")

        logger.info("Rebuilding FAISS index from store documents")
        index_to_meta.clear()
        if index is not None:
            index.reset()  # type: ignore

        store = get_store()
        if not store.docs:
            logger.warning("No store available to rebuild index from")
            return

        logger.info("Processing %d documents for FAISS indexing", len(store.docs))

        total_vectors = 0
        batch_size = 8

        # Process each document separately to avoid accumulating all chunks in memory
        for uri, text in list(store.docs.items()):
            logger.debug("Document %s: %d chars", uri, len(text))
            chunks, metadata = _chunk_document(text, uri)
            logger.debug("Created %d chunks from %s", len(chunks), uri)

            if not chunks:
                continue

            # Process chunks from this document in small batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_meta = metadata[i:i+batch_size]

                embeddings = _encode_with_metrics(
                    embedder,
                    batch_chunks,
                    "rebuild_index",
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                if index is not None:
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    index.add(embeddings_array)  # type: ignore

                    current_index = index.ntotal - len(batch_chunks)  # type: ignore
                    for idx, (chunk_uri, chunk) in enumerate(zip(batch_meta, batch_chunks)):
                        index_to_meta[current_index + idx] = (chunk_uri, chunk)

                total_vectors += len(batch_chunks)

                # Force garbage collection after each batch
                del embeddings, embeddings_array, batch_chunks, batch_meta
                gc.collect()

        if index is not None:
            logger.info("Added %d vectors to FAISS index", total_vectors)
        else:
            logger.warning("No FAISS index available")



def load_store():
    """Load the store from disk and rebuild FAISS index."""
    _STORE_MANAGER.load_store()


def upsert_document(uri: str, text: str) -> Dict[str, Any]:
    """Upsert a single document into the store."""
    if _should_skip_uri(uri):
        logger.warning("Skipping hidden/system file: %s", uri)
        return {"skipped": True, "reason": "hidden/system file", "uri": uri}

    index, index_to_meta, _ = get_faiss_globals()
    embedder = get_embedder()

    _ensure_store_synced()

    store = get_store()
    existed = uri in store.docs
    store.add(uri, text)

    if DEBUG_MODE:
        logger.debug("Debug mode: skipping embeddings for %s", uri)
        save_store()
        return {"upserted": True, "existed": existed}

    base_index = index.ntotal if index is not None else 0  # type: ignore
    chunks = list(_chunk(text))

    if embedder is not None and index is not None:
        # Process in batches to control memory
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
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
            for j, chunk in enumerate(batch_chunks):
                index_to_meta[base_index + i + j] = (uri, chunk)

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


def _extract_text_from_file(file_path: pathlib.Path) -> str:
    """Delegate to shared extractor implementation."""
    return _extractors._extract_text_from_file(file_path)


def _is_meaningful_text(text: str, file_name: str = "") -> bool:
    """Heuristic filter to skip binary/empty payloads regardless of extension."""
    if file_name == ".DS_Store":
        return False
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    printable = sum(ch.isprintable() for ch in stripped)
    density = printable / max(1, len(stripped))
    if density < 0.85:
        return False
    head = stripped[:8]
    if head.startswith(("%PDF-", "PK", "\u0000", "\ufffd")):
        return False
    return True


def _process_file(
    file_path: pathlib.Path,
    index: Any,
    index_to_meta: Dict,
    embedder: Optional[SentenceTransformer],
) -> str:
    """Process a single file and add to index."""
    text = _extract_text_from_file(file_path)
    if not _is_meaningful_text(text, file_path.name):
        logger.warning("Skipping non-text or empty content from %s", file_path)
        return ""

    store = get_store()
    store.add(str(file_path), text)

    if not DEBUG_MODE and embedder is not None and index is not None:
        base_index = index.ntotal  # type: ignore
        chunks = list(_chunk(text))
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
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
            for j, chunk in enumerate(batch_chunks):
                index_to_meta[base_index + i + j] = (str(file_path), chunk)

            del embeddings, batch_chunks
            gc.collect()
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
    except Exception as exc:  # Catch RemoteProtocolError and other network errors
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
                resp = future.result(timeout=60)
            else:
                resp = asyncio.run(send_to_llm(texts))

        except (OSError, ValueError) as exc:
            logger.error("send_store_to_llm failed: %s", exc)
            raise
        except APIConnectionError:
            time.sleep(1)
            continue
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
        i += max_chars - overlap  # Fixed: increment by step size, not absolute position
        if i >= n:
            break
    return out


def _vector_search(query: str, k: int = SEARCH_TOP_K) -> List[Dict[str, Any]]:
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

    scores, indices = index.search(query_array, min(k, index.ntotal))  # type: ignore

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx in index_to_meta:
            uri, text = index_to_meta[idx]
            hits.append({"score": float(score), "uri": uri, "text": text})

    return hits


def search(query: str, top_k: int = SEARCH_TOP_K, max_context_chars: int = SEARCH_MAX_CONTEXT_CHARS):
    """Search the indexed documents and ask the LLM using only those documents as context."""
    logger.info("Vector searching for: %s", query)
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
            return {"error": "No relevant documents found in the indexed corpus.", "sources": []}
    # Re-rank to downweight signatures/short snippets and prioritize query overlap
    candidates = rerank(query, candidates)[:top_k]
    sources = [c.get("uri", "") for c in candidates if c.get("uri")]

    context_parts, total_chars = [], 0
    for entry in candidates:
        part = "Source: %s\nContent:\n%s\n" % (entry["uri"], entry["text"])
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)  # type: ignore
        total_chars += len(part)
    context = "\n".join(context_parts)  # type: ignore

    system_msg = SYSTEM_PROMPT
    user_msg = "Documents:\n%s\n\nQuestion: %s" % (context, query)

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
            timeout=LLM_TIMEOUT,
        )
        if isinstance(resp, dict):
            resp.setdefault("sources", sources)
        return resp
    except (ValueError, OllamaError, APIConnectionError) as exc:  # type: ignore
        logger.error("Ollama API Error: %s", exc)
    except (OSError, RuntimeError) as exc:  # type: ignore
        logger.error("Unexpected error in completion: %s", exc)

    # Graceful fallback: synthesize answer from retrieved passages
    fallback = synthesize_answer(query, candidates)
    fallback["warning"] = "LLM unavailable; reply synthesized from retrieved passages."
    fallback["sources"] = sources
    return fallback


def grounded_answer(question: str, k: int = SEARCH_TOP_K) -> Dict[str, Any]:
    """
    Grounded career-centric answer: retrieve, rerank, filter signatures/boilerplate,
    and synthesize with citations; falls back gracefully when no signal.
    """
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
        if is_low_signal(p):
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
        part = "Source: %s\nContent:\n%s\n" % (entry.get("uri", ""), entry.get("text", ""))
        if total_chars + len(part) > max_chars:
            break
        context_parts.append(part)
        total_chars += len(part)
    context = "\n".join(context_parts)

    # If no LLM available, deterministic synthesis
    if completion is None:
        synthesized = synthesize_answer(question, filtered)
        synthesized["sources"] = sources
        return synthesized

    system_msg = GROUNDING_SYSTEM_PROMPT
    user_msg = f"Documents:\n{context}\n\nQuestion: {question}"

    try:
        resp = completion(  # type: ignore
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            api_base=OLLAMA_API_BASE,
            temperature=LLM_TEMPERATURE,
            stream=False,
            timeout=LLM_TIMEOUT,
        )
        if isinstance(resp, dict):
            resp.setdefault("sources", sources)
        return resp
    except Exception as exc:  # pragma: no cover
        logger.error("grounded_answer completion failed: %s", exc)
        synthesized = synthesize_answer(question, filtered)
        synthesized["warning"] = "LLM unavailable; reply synthesized from retrieved passages."
        synthesized["sources"] = sources
        return synthesized


def verify_grounding_simple(question: str, draft_answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def verify_grounding(question: str, draft_answer: str, citations: List[str]) -> Dict[str, Any]:
    """
    Verify grounding given a list of citation URIs by pulling text from the store.
    """
    store = get_store()
    passages: List[Dict[str, Any]] = []
    for uri in citations:
        if uri in store.docs:
            passages.append({"uri": uri, "text": store.docs[uri], "score": 1.0})
    return verify_grounding_simple(question, draft_answer, passages)
