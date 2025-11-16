#!/usr/bin/env python3
"""
MCP Retrieval Server — FastMCP Streamable HTTP

Environment Variables:
    MCP_PATH: HTTP path for FastMCP (default: /mcp)
    MCP_HOST: Address to listen     (default: 127.0.0.1)
    MCP_PORT: TCP port to listen    (default: 8000)
    MAX_MEMORY_MB: Memory limit in MB (default: 75% of system memory)

Run (dev):
    uv run python http_server.py
    # or
    python http_server.py

Connect URL (client side):
    http://127.0.0.1:8000/mcp
"""

from __future__ import annotations
import gc
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import logging
from typing import Dict, Any, Optional, List
import re

import psutil
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Import shared logic
from rag_core import (
    search,
    load_store,
    save_store,
    upsert_document,
    get_store,
    resolve_input_path,
    _extract_text_from_file,
    rerank,
    grounded_answer,
    verify_grounding,
)

# Set up logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/http_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env if present before evaluating settings
load_dotenv()

# Create FastMCP instance
mcp = FastMCP("retrieval-server")

# Track if we're already shutting down to prevent duplicate saves
SHUTTING_DOWN = False

# Memory management settings
MEMORY_CHECK_INTERVAL = 30  # seconds
MEMORY_LOG_STEP_MB = 256  # log memory usage whenever it crosses another 256MB bucket
# macOS has a global jetsam killer that will SIGKILL processes under memory pressure
# before Python can log anything. We still monitor usage so we have a trail when the
# OS lets us run, but jetsam kills will appear as sudden exits with no last log line.
MEMORY_MONITOR_THREAD = None
LAST_MEMORY_LOG_BUCKET = None

def get_system_memory_mb():
    """Get total system memory in MB."""
    return psutil.virtual_memory().available / 1024 / 1024

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_max_memory_mb():
    """Get the maximum memory limit, defaulting to 75% of system memory."""
    max_memory_env = os.getenv("MAX_MEMORY_MB")
    if max_memory_env:
        return int(max_memory_env)
    else:
        system_memory_mb = get_system_memory_mb()
        return int(system_memory_mb * 0.75)  # 75% of system memory

MAX_MEMORY_MB = get_max_memory_mb()

def memory_monitor():
    """Monitor memory usage and trigger cleanup if needed."""
    global LAST_MEMORY_LOG_BUCKET
    consecutive_limit_hits = 0
    while not SHUTTING_DOWN:
        try:
            memory_mb = get_memory_usage()
            current_bucket = int(memory_mb // MEMORY_LOG_STEP_MB)
            if LAST_MEMORY_LOG_BUCKET is None or current_bucket > LAST_MEMORY_LOG_BUCKET:
                LAST_MEMORY_LOG_BUCKET = current_bucket
                logger.info(
                    "Memory usage crossed %dMB: %.1fMB used (limit %dMB)",
                    current_bucket * MEMORY_LOG_STEP_MB, memory_mb, MAX_MEMORY_MB
                )
            if memory_mb > MAX_MEMORY_MB:
                consecutive_limit_hits += 1
                logger.warning(
                    "Memory usage %.1fMB exceeds limit %dMB. Triggering garbage collection (strike %d).",
                    memory_mb, MAX_MEMORY_MB, consecutive_limit_hits
                )
                gc.collect()

                # Check again after GC
                memory_mb = get_memory_usage()
                if memory_mb > MAX_MEMORY_MB * 0.95:  # Still high after GC
                    logger.error("Memory usage still high after GC: %.1fMB.", memory_mb)
                    if consecutive_limit_hits >= 3:
                        logger.critical(
                            "Memory usage %.1fMB stayed above limit for %d checks. "
                            "Initiating graceful shutdown before the OS kills the process.",
                            memory_mb, consecutive_limit_hits
                        )
                        graceful_shutdown()
                        os._exit(1)
                else:
                    logger.info("Memory usage after GC: %.1fMB", memory_mb)
                    consecutive_limit_hits = 0
            else:
                consecutive_limit_hits = 0

            time.sleep(MEMORY_CHECK_INTERVAL)
        except Exception as e:
            logger.error("Error in memory monitor: %s", e)
            time.sleep(MEMORY_CHECK_INTERVAL)

def start_memory_monitor():
    """Start the memory monitoring thread."""
    global MEMORY_MONITOR_THREAD
    if MEMORY_MONITOR_THREAD is None:
        MEMORY_MONITOR_THREAD = threading.Thread(target=memory_monitor, daemon=True)
        MEMORY_MONITOR_THREAD.start()
        logger.info("Started memory monitor. Max memory limit: %dMB", MAX_MEMORY_MB)

def set_memory_limits():
    """Set system memory limits for the process."""
    if sys.platform == "darwin":
        logger.warning(
            "Skipping RLIMIT_AS enforcement on macOS — the kernel sends SIGKILL "
            "when the soft limit is hit, so we rely on the memory monitor instead. "
            "Use a container/cgroup if you need a hard cap."
        )
        return
    try:
        # Set virtual memory limit (soft only). Keep hard limit high so OS signals don't kill us.
        _, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        max_memory_bytes = MAX_MEMORY_MB * 1024 * 1024
        hard_limit = current_hard if current_hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        # Only lower the hard limit if we already have a finite cap; otherwise leave it untouched.
        if hard_limit != resource.RLIM_INFINITY and max_memory_bytes > hard_limit:
            hard_limit = max_memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard_limit))
        logger.info("Set process memory limit to %dMB", MAX_MEMORY_MB)
    except Exception as e:
        logger.warning("Could not set memory limit: %s", e)

def graceful_shutdown(signum: Optional[int] = None, _frame: Any = None):
    """Handle shutdown gracefully with proper cleanup."""
    global SHUTTING_DOWN

    if SHUTTING_DOWN:
        logger.debug("Shutdown already in progress, skipping duplicate")
        return

    SHUTTING_DOWN = True

    if signum:
        logger.info("Received signal %d. Initiating graceful shutdown...", signum)
    else:
        logger.info("Initiating graceful shutdown...")

    try:
        save_store()
        logger.info("Store saved successfully")
    except Exception as e:
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
                    return {"error": f"Cannot upsert directory. Use index_documents_tool for directories: {normalized_uri}", "upserted": False}
                try:
                    content = _extract_text_from_file(file_path)
                    if not content:
                        return {"error": f"No text extracted from {normalized_uri}", "upserted": False}
                    logger.debug("Read %d characters from %s for upsert", len(content), normalized_uri)
                except Exception as file_err:
                    logger.error("Failed to read %s: %s", normalized_uri, file_err)
                    return {"error": f"Could not read file: {file_err}", "upserted": False}
            else:
                return {"error": f"Text missing and file not found: {normalized_uri}", "upserted": False}

        result: Dict[str, Any] = upsert_document(normalized_uri, content)
        logger.info("Successfully upserted document: %s", uri)
        return result
    except Exception as e:
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
        logger.warning("Received regex-style glob '%s'; falling back to '%s'", cleaned, default_glob)
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
        path: Directory to index (e.g., "docs", "./documents")
        glob: File pattern (default: "**/*" for all supported files)

    Returns:
        dict with count of indexed documents and their URIs

    Examples:
        User: "index docs" → index_documents_tool("docs")
        User: "index the documents" → index_documents_tool("docs")
        User: "add docs to index" → index_documents_tool("docs")
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
            except Exception as read_err:
                logger.warning("Skipping %s: %s", file_path, read_err)
                continue

            upsert_document(str(file_path), content)
            indexed += 1
            indexed_uris.append(str(file_path))

        logger.info("Indexed %d documents from %s", indexed, base)
        return {"indexed": indexed, "uris": indexed_uris}
    except Exception as e:
        logger.error("Error indexing path %s: %s", path, e)
        return {"error": str(e), "indexed": 0, "uris": []}

@mcp.tool()
def index_url_tool(
    url: Optional[str] = None,
    doc_id: Optional[str] = None,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
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
        # route the request to the directory indexer so "index ./docs" works.
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

        import pathlib
        # Treat URL as a pathlib.Path for _extract_text_from_file
        url_path = pathlib.Path(url)
        content = _extract_text_from_file(url_path)

        if not content:
            logger.warning("No text extracted from URL %s", url)
            return {"error": "Failed to extract text from URL", "indexed": 0, "uri": url}

        # Use URL as document ID if not provided
        doc_id = doc_id or url
        upsert_document(doc_id, content)

        logger.info("Indexed document from URL: %s", url)
        return {"indexed": 1, "uri": url, "doc_id": doc_id}
    except Exception as e:
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
        result = search(query, top_k=top_k)
        logger.info("Search completed for: %s", query)

        # Force garbage collection after search to manage memory
        gc.collect()

        # Handle different return types from search function
        if hasattr(result, 'choices') and getattr(result, 'choices', None):
            # LiteLLM response object
            choices = getattr(result, 'choices', [])
            if choices:
                message = getattr(choices[0], 'message', None)
                content = getattr(message, 'content', None) if message else None
                return {
                    "answer": content or str(result),
                    "model": getattr(result, 'model', 'unknown'),
                    "usage": getattr(result, 'usage', {})
                }

        if isinstance(result, dict):
            # Already a dict
            return result
        else:
            # Convert other types to string
            return {"answer": str(result)}

    except Exception as e:
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
        store = get_store()
        if not store:
            return {"uris": [], "message": "Store not loaded or empty."}

        uris = list(store.docs.keys())
        logger.info("Found %d indexed documents.", len(uris))
        return {"uris": uris}
    except Exception as e:
        logger.error("Error listing indexed documents: %s", e, exc_info=True)
        return {"error": str(e), "uris": []}

@mcp.tool()
def rerank_tool(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    RERANK passages for a query using a lightweight heuristic.

    Use this after search_tool when you need the most relevant snippets first.
    """
    try:
        return rerank(query, passages)
    except Exception as e:
        logger.error("Error reranking passages: %s", e, exc_info=True)
        return []

@mcp.tool()
def grounded_answer_tool(question: str, k: int = 5) -> Dict[str, Any]:
    """
    Generate a grounded answer from the indexed corpus.

    This performs a vector search and returns an answer plus citations.
    """
    try:
        return grounded_answer(question, k=k)
    except Exception as e:
        logger.error("Error generating grounded answer: %s", e, exc_info=True)
        return {"error": str(e)}

@mcp.tool()
def verify_grounding_tool(question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
    """
    Verify that an answer is grounded in the cited documents.
    """
    try:
        return verify_grounding(question, answer, citations)
    except Exception as e:
        logger.error("Error verifying grounding: %s", e, exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        # Set memory limits before doing anything else
        set_memory_limits()

        # Configure MCP
        mcp.settings.log_level = "DEBUG"
        mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))

        # Start memory monitoring
        start_memory_monitor()

        # Load store
        logger.info("Loading document store...")
        load_store()
        logger.info("Document store loaded successfully")

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
    except Exception as e:
        logger.error("Fatal server error: %s", e, exc_info=True)
        graceful_shutdown()
        sys.exit(1)
