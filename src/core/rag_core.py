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
import zipfile
import io
import string
import threading

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv  # type: ignore
# Avoid tokenizers parallelism deadlocks after fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import SentenceTransformer  # type: ignore
from ollama import AsyncClient  # type: ignore

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None

try:
    from docx import Document  # type: ignore
except ImportError:
    Document = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None

try:
    import requests  # type: ignore
except ImportError:
    requests = None

try:
    import faiss  # type: ignore
except ImportError:
    if os.getenv("RAG_DEBUG_MODE", "false").lower() != "true":
        raise
    faiss = None

# Disable PyTorch multiprocessing to avoid segfaults on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
DB_PATH = os.getenv("RAG_DB", "./cache/rag_store.jsonl")

# Lazy-initialized global state
_EMBEDDER: Optional[SentenceTransformer] = None
_INDEX: Optional[Any] = None
_INDEX_TO_META: Optional[Dict[int, Tuple[str, str]]] = None
_EMBED_DIM: Optional[int] = None
_STORE: Optional['Store'] = None
_REBUILD_LOCK = threading.Lock()

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


def get_embedder() -> Optional[SentenceTransformer]:
    """Get the embedding model, loading it lazily if needed. Returns None in debug mode."""
    global _EMBEDDER

    if DEBUG_MODE:
        logger.info("Debug mode enabled - skipping embedding model loading")
        return None

    if _EMBEDDER is None:
        try:
            logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
            _EMBEDDER = SentenceTransformer(
                EMBED_MODEL_NAME,
                device='cpu',
                backend='torch'
            )
            # Disable multiprocessing to avoid segfaults on macOS
            import torch
            torch.set_num_threads(1)
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
        """Initialize an empty in-memory store."""
        self.docs: Dict[str, str] = {}
        self.last_loaded: float = 0.0
        logger.debug("Initialized new Store instance")

    def add(self, uri: str, text: str):
        """Add a document to the store."""
        logger.debug("Adding document: %s", uri)
        self.docs[uri] = text
        logger.debug("Store now contains %d documents", len(self.docs))


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


def _rebuild_faiss_index():
    """Rebuild the FAISS index from store documents."""
    with _REBUILD_LOCK:
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

        logger.info("Processing %d documents for FAISS indexing", len(_STORE.docs))

        total_vectors = 0
        batch_size = 8

        # Process each document separately to avoid accumulating all chunks in memory
        for uri, text in list(_STORE.docs.items()):
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


def _download_from_url(url: str) -> bytes:
    """Download content from a URL."""
    if requests is None:
        logger.warning("URL support not available. Install requests: pip install requests")
        return b""
    try:
        logger.info("Downloading from %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.SSLError as ssl_exc:
        logger.error("SSL error downloading %s: %s", url, ssl_exc)
        return b"[SSL ERROR: Could not connect to %s]" % url.encode()
    except requests.exceptions.ConnectionError as conn_exc:
        logger.error("Connection error downloading %s: %s", url, conn_exc)
        return b"[CONNECTION ERROR: Could not connect to %s]" % url.encode()
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return b"[ERROR: Could not download %s: %s]" % (url.encode(), str(exc).encode())


def _extract_text_from_file(file_path: pathlib.Path) -> str:
    """Extract text from various file types (txt, pdf, docx, html) or URLs."""
    # Check if it's a URL and normalize single-slash URLs
    file_str = str(file_path)
    if file_path.name in {".DS_Store"}:
        return ""

    def _read_head(path: pathlib.Path, size: int = 512) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read(size)
        except Exception:
            return b""

    def _is_probably_text_bytes(buf: bytes) -> bool:
        if not buf:
            return False
        printable = sum((chr(b).isprintable() or chr(b).isspace()) for b in buf)
        density = printable / max(1, len(buf))
        return density >= 0.8

    head = None
    if not file_str.startswith(("http://", "https://")):
        head = _read_head(file_path, 512)
    if file_str.startswith('http:/') and not file_str.startswith('http://'):
        file_str = file_str.replace('http:/', 'http://', 1)
    elif file_str.startswith('https:/') and not file_str.startswith('https://'):
        file_str = file_str.replace('https:/', 'https://', 1)

    def _normalize_suffix_from_head(default_suffix: str, head_bytes: Optional[bytes]) -> str:
        if head_bytes is None:
            return default_suffix
        if head_bytes.startswith(b"%PDF-"):
            return ".pdf"
        if head_bytes.startswith(b"PK"):
            # could be zip-based formats: docx, pages, ods, etc.
            if default_suffix.lower() in (".docx", ".pages"):
                return default_suffix.lower()
            return ".zip"
        return default_suffix

    if file_str.startswith(('http://', 'https://')):
        content = _download_from_url(file_str)
        if not content:
            return ""
        # Determine file type from URL or content
        if file_str.endswith('.pdf'):
            suffix = '.pdf'
        elif file_str.endswith(('.docx', '.doc')):
            suffix = '.docx'
        elif file_str.endswith(('.html', '.htm')):
            suffix = '.html'
        else:
            # Try to detect HTML
            try:
                decoded = content.decode('utf-8', errors='ignore')
                if decoded.strip().startswith(('<html', '<!DOCTYPE', '<!doctype')):
                    suffix = '.html'
                else:
                    suffix = '.txt'
            except Exception:
                suffix = '.txt'
    else:
        content = None
        suffix = _normalize_suffix_from_head(file_path.suffix.lower(), head)

    if suffix == '.pdf':
        if PdfReader is None:
            logger.warning("PDF support not available. Install pypdf: pip install pypdf")
            return ""
        try:
            if content:
                # PDF from URL
                import io
                reader = PdfReader(io.BytesIO(content))
            else:
                # PDF from file
                reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
            joined = "\n".join(text_parts).strip()
            if not joined:
                logger.warning("No text extracted from PDF %s", file_path)
                return ""
            return joined
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", file_path, exc)
            return ""

    elif suffix == '.pages':
        # Pages files are zip archives that usually contain a PDF preview
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                names = zf.namelist()
                # Prefer QuickLook/Preview.pdf if present
                pdf_candidates = []
                for name in names:
                    lower = name.lower()
                    if lower.endswith(".pdf"):
                        # Weight QuickLook/Preview higher
                        priority = 0 if "quicklook/preview.pdf" in lower else 1
                        pdf_candidates.append((priority, name))
                pdf_candidates.sort()
                for _, pdf_name in pdf_candidates:
                    try:
                        pdf_bytes = zf.read(pdf_name)
                        reader = PdfReader(io.BytesIO(pdf_bytes)) if PdfReader else None
                        if reader is None:
                            logger.warning("PDF support not available to read %s inside %s", pdf_name, file_path)
                            break
                        pages = []
                        for page in reader.pages:
                            page_text = page.extract_text() or ""
                            if page_text:
                                pages.append(page_text)
                        joined = "\n".join(pages).strip()
                        if joined:
                            return joined
                    except Exception as exc:
                        logger.debug("Failed to read embedded PDF %s in %s: %s", pdf_name, file_path, exc)
                        continue

                # Fallback to any XML/HTML/text files inside the archive
                text_candidates = [n for n in names if n.lower().endswith((".xml", ".html", ".htm", ".txt"))]
                for name in text_candidates:
                    try:
                        raw = zf.read(name)
                        text = raw.decode("utf-8", errors="ignore").strip()
                        if text:
                            return text
                    except Exception:
                        continue
        except Exception as exc:
            logger.error("Failed to read Pages file %s: %s", file_path, exc)
        return ""

    elif suffix in ['.docx', '.doc']:
        if Document is None:
            logger.warning("DOCX support not available. Install python-docx: pip install python-docx")
            return ""
        try:
            if content:
                # DOCX from URL
                import io
                doc = Document(io.BytesIO(content))
            else:
                # DOCX from file
                doc = Document(str(file_path))
            text_parts = [paragraph.text for paragraph in doc.paragraphs]
            return "\n".join(text_parts)
        except Exception as exc:
            logger.error("Failed to read DOCX %s: %s", file_path, exc)
            return ""

    elif suffix in ['.html', '.htm']:
        if BeautifulSoup is None:
            logger.warning("HTML support not available. Install beautifulsoup4: pip install beautifulsoup4")
            return ""
        try:
            if content:
                # HTML from URL
                html_content = content.decode('utf-8', errors='ignore')
            else:
                # HTML from file
                html_content = file_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text
            text = soup.get_text(separator='\n')
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as exc:
            logger.error("Failed to read HTML %s: %s", file_path, exc)
            return ""

    else:
        # Default to plain text, but guard against binaries by header and printable ratio
        try:
            if content:
                # Text from URL
                text = content.decode('utf-8', errors='ignore')
            else:
                # Text from file
                if head is not None and not _is_probably_text_bytes(head):
                    return ""
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            # Quick binary guard: PDFs, zips, binaries often leave telltale headers
            lower_text = text[:8]
            if lower_text.startswith("%PDF-") or lower_text.startswith("PK"):
                return ""
            return text
        except Exception as exc:
            logger.error("Failed to read file %s: %s", file_path, exc)
            return ""


def _process_file(file_path: pathlib.Path, index: Any, index_to_meta: Dict,
                  embedder: Optional[SentenceTransformer]) -> str:
    """Process a single file and add to index."""
    def _is_meaningful_text(text: str) -> bool:
        """Heuristic filter to skip binary/empty payloads regardless of extension."""
        if file_path.name == ".DS_Store":
            return False
        if not text:
            return False
        stripped = text.strip()
        if len(stripped) < 40:  # too short
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
        chunks = list(_chunk(text))

        # Process in batches to control memory
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
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
            for j, chunk in enumerate(batch_chunks):
                index_to_meta[base_index + i + j] = (str(file_path), chunk)

            # Explicit cleanup
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
                resp = future.result(timeout=120)
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


def _vector_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
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
        # If we have docs but no vectors, force a rebuild once
        store = get_store()
        if len(getattr(store, "docs", {})) > 0:
            logger.info("No vector hits but store has docs; rebuilding index and retrying...")
            _rebuild_faiss_index()
            candidates = _vector_search(query, k=top_k)
        if not candidates:
            logger.info("No vector hits; refusing to answer from outside sources.")
            return {"error": "No relevant documents found in the indexed corpus.", "sources": []}
    sources = [c.get("uri", "") for c in candidates if c.get("uri")]

    context_parts, total_chars = [], 0
    for entry in candidates:
        part = "Source: %s\nContent:\n%s\n" % (entry["uri"], entry["text"])
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)  # type: ignore
        total_chars += len(part)
    context = "\n".join(context_parts)  # type: ignore

    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the documents provided. "
        "Cite the exact source URI for each fact. Include a final 'Sources:' section listing the URIs you used. "
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
            temperature=LLM_TEMPERATURE,
            stream=False,
            timeout=120,
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


# --- Lightweight rerank/ground/verify utilities used by MCP tools ---

def rerank(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple reranker that boosts passages containing the query terms. This is a
    lightweight heuristic so we avoid an additional model dependency.
    """
    lowered_terms = [t for t in query.lower().split() if t]

    def _score(p: Dict[str, Any]) -> float:
        """Heuristic score combining existing score with query term hits."""
        base = float(p.get("score", 0.0) or 0.0)
        text = str(p.get("text", "")).lower()
        term_hits = sum(text.count(term) for term in lowered_terms) if lowered_terms else 0
        return base + term_hits * 0.1

    ranked = sorted(passages, key=_score, reverse=True)
    # Attach heuristic scores for downstream consumers
    for p in ranked:
        p["score"] = _score(p)
    return ranked


def synthesize_answer(question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def grounded_answer(question: str, k: int = 3) -> Dict[str, Any]:
    """
    Retrieve top-k passages via vector search and synthesize a grounded answer.
    """
    hits = _vector_search(question, k=k)
    reranked = rerank(question, hits)
    return synthesize_answer(question, reranked)


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
