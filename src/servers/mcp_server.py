#!/usr/bin/env python3
"""
MCP Retrieval Server — FastMCP Streamable HTTP

Environment Variables:
    MCP_PATH: HTTP path for FastMCP (default: /mcp)
    MCP_HOST: Address to listen     (default: 127.0.0.1)
    MCP_PORT: TCP port to listen    (default: 8000)
    MAX_MEMORY_MB: Memory limit in MB (default: 75% of system memory)

Run (dev):
    uv run python src/servers/mcp_server.py
    # or
    python src/servers/mcp_server.py

Connect URL (client side):
    http://127.0.0.1:8000/mcp
"""

from __future__ import annotations
import atexit
import gc
import os
import json
from pathlib import Path
import signal
import sys
import threading
from typing import Dict, Any, Optional, List
import re
import anyio

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pylint: disable=wrong-import-position
from mcp.server.fastmcp import FastMCP
from starlette.responses import Response
from dotenv import load_dotenv

# Import factory and interfaces
from src.core.factory import get_rag_backend
from src.core.interfaces import RAGBackend

# Import shared logic (only for constants/utils that are safe)
from src.core.rag_core import (
    resolve_input_path,
    OLLAMA_API_BASE
)
from src.core.extractors import _extract_text_from_file

from src.servers.mcp_app.logging_config import (
    configure_logging,
    AccessLogMiddleware,
)
from src.servers.mcp_app.memory import (
    MAX_MEMORY_MB,
    get_memory_usage,
    set_memory_limits,
    start_memory_monitor,
)
from src.servers.mcp_app.metrics import (
    refresh_prometheus_metrics,
    metrics_response,
)
from src.servers.mcp_app.http_app import patch_streamable_http

logger, access_logger = configure_logging()
patch_streamable_http(access_logger=access_logger)

# Load environment variables from .env if present before evaluating settings
load_dotenv()

# Load configuration from file if it exists
CONFIG_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "settings.json"

def load_app_config():
    """Load configuration from settings.json."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Update environment variables for MCP
            if "mcpHost" in config:
                os.environ["MCP_HOST"] = config["mcpHost"]
            if "mcpPort" in config:
                os.environ["MCP_PORT"] = str(config["mcpPort"])
            if "mcpPath" in config:
                os.environ["MCP_PATH"] = config["mcpPath"]

            # Update RAG Core configuration
            # pylint: disable=import-outside-toplevel,consider-using-from-import
            import src.core.rag_core as rag_core

            # Map settings.json keys to rag_core variables
            if "apiEndpoint" in config:
                rag_core.OLLAMA_API_BASE = config["apiEndpoint"]
            elif "ollamaApiUrl" in config:  # Legacy fallback
                rag_core.OLLAMA_API_BASE = config["ollamaApiUrl"]

            if "model" in config:
                rag_core.LLM_MODEL_NAME = config["model"]
                rag_core.ASYNC_LLM_MODEL_NAME = config["model"].split("/")[-1]
            elif "ollamaModel" in config:  # Legacy fallback
                rag_core.LLM_MODEL_NAME = config["ollamaModel"]
                rag_core.ASYNC_LLM_MODEL_NAME = config["ollamaModel"].split("/")[-1]

            if "embeddingModel" in config:
                rag_core.EMBED_MODEL_NAME = config["embeddingModel"]

            logger.info("Loaded configuration from %s", CONFIG_FILE)
            return config
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load configuration: %s", e)
    return {}

app_config = load_app_config()

# Initialize Backend
backend: RAGBackend = get_rag_backend()

# Create FastMCP instance
mcp = FastMCP("retrieval-server")

# Track if we're already shutting down to prevent duplicate saves
shutting_down = False
store_loading = False
store_loaded = False
store_load_thread: Optional[threading.Thread] = None

# Memory management settings
MEMORY_CHECK_INTERVAL = 30  # seconds
MEMORY_LOG_STEP_MB = 256  # log memory usage whenever it crosses another 256MB bucket
from src.servers.mcp_app.api import rest_api
rest_api.add_middleware(AccessLogMiddleware, access_logger=access_logger)


def _background_load_store():
    # pylint: disable=global-statement
    global store_loading, store_loaded
    if store_loading or store_loaded:
        return
    store_loading = True
    try:
        # Load store and rebuild index in background
        backend.load_store()
        backend.rebuild_index()
        refresh_prometheus_metrics(OLLAMA_API_BASE)
        store_loaded = True
        logger.info("Background store load and index rebuild complete")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Background store load failed: %s", exc, exc_info=True)
    finally:
        store_loading = False


def start_background_store_load():
    """Kick off store load/rebuild in a background thread so startup is fast."""
    # pylint: disable=global-statement
    global store_load_thread
    if store_load_thread and store_load_thread.is_alive():
        return
    store_load_thread = threading.Thread(target=_background_load_store, daemon=True)
    store_load_thread.start()

def _safe_int(value: Any) -> Optional[int]:
    """Convert a value to int when possible, otherwise None."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

def _normalize_ollama_base() -> str:
    """Return a normalized Ollama API base URL without trailing slash."""
    base = os.getenv("OLLAMA_API_BASE", OLLAMA_API_BASE)
    return base.rstrip("/")


def graceful_shutdown(signum: Optional[int] = None, _frame: Any = None):
    """Handle shutdown gracefully with proper cleanup."""
    # pylint: disable=import-outside-toplevel
    import logging
    # Suppress logging errors (like 'I/O operation on closed file') during shutdown
    logging.raiseExceptions = False

    # pylint: disable=global-statement
    global shutting_down

    if shutting_down:
        logger.debug("Shutdown already in progress, skipping duplicate")
        return

    shutting_down = True

    if signum:
        logger.info("Received signal %d. Initiating graceful shutdown...", signum)
    else:
        logger.info("Initiating graceful shutdown...")

    try:
        backend.save_store()
        logger.info("Store saved successfully")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error saving store during shutdown: %s", e, exc_info=True)
    finally:
        logger.info("Shutdown complete")
        if signum:
            sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
signal.signal(signal.SIGTERM, graceful_shutdown)  # kill command

# Register cleanup on normal exit
atexit.register(graceful_shutdown)

@mcp.tool()
def upsert_document_tool(uri: str, text: str) -> Dict[str, Any]:
    """Upsert a SINGLE document into the store.

    DO NOT USE this tool for:
    - Indexing directories (use index_documents_tool instead)
    - Indexing multiple files (use index_documents_tool instead)
    - When user says "index", "add docs", "index directory" (use index_documents_tool)

    USE THIS TOOL ONLY when:
    - Adding/updating ONE specific document with provided text content
    - Adding/updating an Internet URL, e.g., http://example.com/doc.txt
    - The text content is already available (not reading from disk)

    IMPORTANT: The 'uri' must be an exact, literal string and must not be modified.

    Args:
        uri: Document identifier/path (must be a FILE, not a directory)
        text: Document content (required, not empty)

    Returns:
        dict with upserted status and whether document existed
    """
    try:
        logger.info("Upserting document: %s", uri)
        normalized_uri = str(Path(uri))
        content = text or ""
        if not content.strip():
            # Fallback to reading from disk when the caller provided only a file path.
            file_path = Path(uri)
            if file_path.exists():
                if file_path.is_dir():
                    return {
                        "error": (f"Cannot upsert directory. Use index_documents_tool "
                                  f"for directories: {normalized_uri}"),
                        "upserted": False
                    }
                try:
                    content = _extract_text_from_file(file_path)
                    if not content:
                        return {
                            "error": f"No text extracted from {normalized_uri}",
                            "upserted": False
                        }
                    logger.debug(
                        "Read %d characters from %s for upsert",
                        len(content),
                        normalized_uri
                    )
                except Exception as file_err:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to read %s: %s", normalized_uri, file_err)
                    return {"error": f"Could not read file: {file_err}", "upserted": False}
            else:
                return {
                    "error": f"Text missing and file not found: {normalized_uri}",
                    "upserted": False
                }

        upsert_result: Dict[str, Any] = backend.upsert_document(normalized_uri, content)
        logger.info("Successfully upserted document: %s", uri)
        return upsert_result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error upserting document %s: %s", uri, e)
        return {"error": str(e), "upserted": False}

def _normalize_glob(glob: Optional[str]) -> str:
    """Normalize the incoming glob string and handle common regex mistakes."""
    default_glob = "**/*"
    if not glob or not glob.strip():
        return default_glob
    cleaned = glob.strip()
    # Some clients mistakenly send regex '**/.*' meaning 'anything'
    if cleaned in {"**/.*", "./**/.*"}:
        logger.warning(
            "Received regex-style glob '%s'; falling back to '%s'",
            cleaned,
            default_glob
        )
        return default_glob
    return cleaned

_URL_PATTERN = re.compile(r"https?://[^\s\"'>]+")

def _extract_url_from_query(query: Optional[str]) -> Optional[str]:
    """Pull the first URL-looking token from a free-form query string."""
    if not query:
        return None
    match = _URL_PATTERN.search(query)
    if match:
        return match.group(0).strip().rstrip(",.")
    return None

def _extract_path_from_query(query: Optional[str]) -> Optional[str]:
    """
    Pull a potential filesystem path from a free-form string like
    'index ./docs' so we can route to index_documents_tool.
    """
    if not query:
        return None
    text = query.strip().strip("\"'")
    lowered = text.lower()
    for prefix in ("index ", "add ", "ingest ", "load "):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip().strip("\"'")
            break
    if text:
        return text
    return None

@mcp.tool()
def index_documents_tool(path: str, glob: str = "**/*") -> Dict[str, Any]:
    """ADD/INDEX documents from a directory into the searchable store.

    USE THIS TOOL when the user wants to:
    - Add documents to the index
    - Index files or directories
    - Make documents searchable
    - Build/create/populate the document index

    This reads files from disk and makes them searchable via search_tool().
    Supports: .txt, .pdf, .docx, .doc, .html, .htm files

    Args:
        path: Directory to index (e.g., "documents", "./documents")
        glob: File pattern (default: "**/*" for all supported files)

    Returns:
        dict with count of indexed documents and their URIs

    Examples:
        User: "index documents" → index_documents_tool("documents")
        User: "index the documents" → index_documents_tool("documents")
        User: "add documents to index" → index_documents_tool("documents")
    """
    try:
        try:
            base = resolve_input_path(path)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            return {"error": str(exc), "indexed": 0, "uris": []}

        normalized_glob = _normalize_glob(glob)
        logger.info("Indexing directory %s with glob '%s'", base, normalized_glob)
        if base.is_file():
            files = [base]
        else:
            files = list(base.rglob(normalized_glob))
        if not files:
            message = f"No files found in {base} matching {normalized_glob}"
            logger.info(message)
            return {"indexed": 0, "uris": [], "error": message}

        indexed = 0
        indexed_uris = []
        for file_path in files:
            try:
                content = _extract_text_from_file(file_path)
                if not content:
                    logger.warning("No text extracted from %s, skipping", file_path)
                    continue
            except Exception as read_err:  # pylint: disable=broad-exception-caught
                logger.warning("Skipping %s: %s", file_path, read_err)
                continue

            backend.upsert_document(str(file_path), content)
            indexed += 1
            indexed_uris.append(str(file_path))

        logger.info("Indexed %d documents from %s", indexed, base)
        return {"indexed": indexed, "uris": indexed_uris}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error indexing path %s: %s", path, e)
        return {"error": str(e), "indexed": 0, "uris": []}

@mcp.tool()
def index_url_tool(
    url: Optional[str] = None,
    doc_id: Optional[str] = None,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    INDEX a document from a URL into the searchable store.

    CRITICAL: ALWAYS use this tool for prompts like:
    - "index http://..." or "index https://..."
    - "add this url to the index"
    - "index a web page"

    DO NOT use index_documents_tool for URLs. ONLY use this tool for remote documents.

    USE THIS TOOL when the user wants to:
    - Index a document from a URL
    - Add a web page to the index
    - Index remote documents (PDF, DOCX, HTML, TXT)

    Supports URLs pointing to:
    - PDF files (http://example.com/doc.pdf)
    - DOCX files (http://example.com/doc.docx)
    - HTML pages (http://example.com/page.html)
    - Plain text files (http://example.com/file.txt)

    Args:
        url: URL of the document to index (e.g., "http://example.com/doc.pdf")
        doc_id: Optional document ID (defaults to the URL)
        query: Optional free-form request. If it contains a URL, it will be extracted.

    Returns:
        dict with status and indexed URI

    Examples:
        User: "index http://example.com/doc.pdf"
        User: "index https://about.me/jhcook"
        User: "add this url to the index: http://..."
    """
    try:
        # Accept legacy callers that accidentally send {"query": "..."}
        if not url:
            url = _extract_url_from_query(query)

        # If there is no URL but we received something that looks like a local path,
        # route the request to the directory indexer so "index ./documents" works.
        if not url:
            possible_path = _extract_path_from_query(query)
            if possible_path and not possible_path.startswith(("http://", "https://")):
                logger.info(
                    "index_url_tool received non-URL input '%s'; routing to index_documents_tool",
                    possible_path
                )
                return index_documents_tool(possible_path)

        if not url:
            return {"error": "URL missing. Provide a url argument or a valid query.", "indexed": 0}

        # Treat URL as a pathlib.Path for _extract_text_from_file
        url_path = Path(url)
        content = _extract_text_from_file(url_path)

        if not content:
            logger.warning("No text extracted from URL %s", url)
            return {"error": "Failed to extract text from URL", "indexed": 0, "uri": url}

        # Use URL as document ID if not provided
        doc_id = doc_id or url
        backend.upsert_document(doc_id, content)

        logger.info("Indexed document from URL: %s", url)
        return {"indexed": 1, "uri": url, "doc_id": doc_id}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error indexing URL %s: %s", url, e)
        return {"error": str(e), "indexed": 0, "uri": url}

@mcp.tool()
def search_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search and answer questions about indexed documents.

    This tool searches indexed documents and returns a complete answer generated by an LLM.
    Use this whenever the user asks about document content.

    CRITICAL INSTRUCTIONS:
    1. Pass the user's exact question unchanged
    2. After calling this tool, immediately present the answer to the user
    3. The response contains a complete answer - relay it directly without adding anything

    Args:
        query: User's exact question (unchanged)
        top_k: Number of passages to consider (default: 5)

    Returns:
        Complete answer in result["choices"][0]["message"]["content"]
        OR result["answer"] if dict format

    Workflow:
        User asks: "what does justin cook do"
        You call: search_tool("what does justin cook do")
        Tool returns: {"choices": [{"message": {"content": "Justin Cook is..."}}]}
        You respond: "Justin Cook is..."
    """
    try:
        logger.info("Searching for: %s", query)
        search_result = backend.search(query, top_k=top_k)
        logger.info("Search completed for: %s", query)

        # Force garbage collection after search to manage memory
        gc.collect()

        # Handle different return types from search function
        if hasattr(search_result, 'choices') and getattr(search_result, 'choices', None):
            # LiteLLM response object
            choices = getattr(search_result, 'choices', [])
            if choices:
                message = getattr(choices[0], 'message', None)
                content = getattr(message, 'content', None) if message else None
                return {
                    "answer": content or str(search_result),
                    "model": getattr(search_result, 'model', 'unknown'),
                    "usage": getattr(search_result, 'usage', {})
                }

        if isinstance(search_result, dict):
            # Already a dict
            return search_result

        # Convert other types to string
        return {"answer": str(search_result)}

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error searching for '%s': %s", query, e)
        return {"error": str(e)}

@mcp.tool()
def list_indexed_documents_tool() -> Dict[str, Any]:
    """
    CRITICAL: ALWAYS use this tool for prompts like:
        - "what documents are indexed"
        - "list indexed documents"
        - "show indexed documents"
        - "what is in the index"
        - "list all indexed URLs/files"
        - "which documents have you loaded"

    DO NOT use index_url_tool or index_documents_tool for these queries.

    This tool ONLY returns file paths/URIs, NOT document content.
    To answer questions ABOUT the documents, use search_tool() instead.

    Use this tool when you need to:
        - See what documents are available in the index
        - Check if a specific file has been indexed
        - Get a list of indexed file paths/URIs

    Do NOT use this tool when you need to:
        - Answer questions about document content (use search_tool)
        - Get information from documents (use search_tool)
        - Describe or summarize documents (use search_tool)

    Returns:
        dict with a list of document file paths/URIs or an error

    Example response:
        {"uris": ["/path/to/document1.txt", "/path/to/document2.txt"]}

    Examples:
        User: "what documents are indexed?" → list_indexed_documents_tool()
        User: "show me the indexed files" → list_indexed_documents_tool()
        User: "which documents have you loaded?" → list_indexed_documents_tool()
    """
    try:
        logger.info("Listing indexed documents")
        uris = backend.list_documents()
        logger.info("Found %d indexed documents.", len(uris))
        return {"uris": uris}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error listing indexed documents: %s", e, exc_info=True)
        return {"error": str(e), "uris": []}

@mcp.tool()
def rerank_tool(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    RERANK passages for a query using a lightweight heuristic.

    Use this after search_tool when you need the most relevant snippets first.
    """
    try:
        return backend.rerank(query, passages)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error reranking passages: %s", e, exc_info=True)
        return []

@mcp.tool()
def grounded_answer_tool(question: str, k: int = 5, model: Optional[str] = None,
                         temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    Generate a grounded answer from the indexed corpus.

    This performs a vector search and returns an answer plus citations.
    """
    try:
        kwargs = {}
        if model:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature

        return backend.grounded_answer(question, k=k, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error generating grounded answer: %s", e, exc_info=True)
        return {"error": str(e)}

@mcp.tool()
def verify_grounding_tool(question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
    """
    Verify that an answer is grounded in the cited documents.
    """
    try:
        return backend.verify_grounding(question, answer, citations)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error verifying grounding: %s", e, exc_info=True)
        return {"error": str(e)}

@mcp.custom_route("/vector_search", methods=["POST"])
async def vector_search_endpoint(request) -> Response:
    """Vector search endpoint that returns raw search results without LLM synthesis."""
    try:
        body = await request.json()
        query = body.get("query", "")
        k = int(body.get("k", 5))
        
        # Import rag_core to access _vector_search
        from src.core import rag_core
        results = await anyio.to_thread.run_sync(rag_core._vector_search, query, k)
        
        return Response(
            content=json.dumps({"results": results}),
            media_type="application/json"
        )
    except Exception as exc:
        logger.error("Vector search error: %s", exc, exc_info=True)
        return Response(
            content=json.dumps({"error": str(exc)}),
            status_code=500,
            media_type="application/json"
        )

@mcp.custom_route("/metrics", methods=["GET"])
async def metrics_endpoint(_request) -> Response:
    """Expose Prometheus metrics for the MCP server and Ollama backend."""
    refresh_error = None
    try:
        await anyio.to_thread.run_sync(refresh_prometheus_metrics, OLLAMA_API_BASE)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        refresh_error = exc
        logger.error("Metrics refresh failed (serving last known values): %s", exc, exc_info=True)

    return metrics_response(refresh_error)

# Patch FastMCP to ensure every streamable HTTP app instance includes our middleware
# and metrics endpoint; FastMCP builds a fresh Starlette app on each call, so we
# wrap the constructor rather than mutating a one-off instance.

if __name__ == "__main__":
    try:
        # Bail out early if port is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        in_use = False
        try:
            host = (mcp.settings.host if hasattr(mcp, "settings") else
                    os.getenv("MCP_HOST", "127.0.0.1"))
            port = int(os.getenv("MCP_PORT", "8000"))
            result = sock.connect_ex((host, port))
            if result == 0:
                in_use = True
        finally:
            sock.close()
        if in_use:
            logger.error("Port %s already in use. Refusing to start MCP.",
                         os.getenv("MCP_PORT", "8000"))
            sys.exit(1)

        # Set memory limits before doing anything else
        set_memory_limits()

        # Configure MCP
        mcp.settings.log_level = "DEBUG"
        mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))

        # Start memory monitoring
        start_memory_monitor(graceful_shutdown)

        # Load store asynchronously to speed startup
        logger.info("Starting background store load...")
        start_background_store_load()

        # Log server configuration and memory info
        memory_mb = get_memory_usage()
        logger.info(
            "Starting MCP server on %s:%d%s",
            mcp.settings.host, mcp.settings.port, mcp.settings.streamable_http_path
        )
        logger.info("Current memory usage: %.1fMB (limit: %dMB)", memory_mb, MAX_MEMORY_MB)
        logger.info("Press Ctrl+C to gracefully shutdown")

        # Start server
        mcp.run(transport="streamable-http")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        graceful_shutdown()
    except MemoryError:
        logger.error("Out of memory! Consider increasing MAX_MEMORY_MB or reducing document size")
        graceful_shutdown()
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Fatal server error: %s", e, exc_info=True)
        graceful_shutdown()
        sys.exit(1)
