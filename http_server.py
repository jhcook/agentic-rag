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
import atexit
import gc
import os
from pathlib import Path
import psutil
import resource
import signal
import sys
import threading
import time
import logging
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

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

# Import shared logic
from rag_core import (
    search, 
    load_store, 
    save_store,
    upsert_document,
    get_store
)

# Create FastMCP instance
mcp = FastMCP("retrieval-server")

# Track if we're already shutting down to prevent duplicate saves
_shutting_down = False

# Memory management settings
MEMORY_CHECK_INTERVAL = 30  # seconds
MEMORY_LOG_STEP_MB = 256  # log memory usage whenever it crosses another 256MB bucket
# macOS has a global jetsam killer that will SIGKILL processes under memory pressure
# before Python can log anything. We still monitor usage so we have a trail when the
# OS lets us run, but jetsam kills will appear as sudden exits with no last log line.
_memory_monitor_thread = None
_last_memory_log_bucket = None

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
    global _last_memory_log_bucket
    consecutive_limit_hits = 0
    while not _shutting_down:
        try:
            memory_mb = get_memory_usage()
            current_bucket = int(memory_mb // MEMORY_LOG_STEP_MB)
            if _last_memory_log_bucket is None or current_bucket > _last_memory_log_bucket:
                _last_memory_log_bucket = current_bucket
                logger.info(
                    f"Memory usage crossed {(current_bucket) * MEMORY_LOG_STEP_MB}MB: "
                    f"{memory_mb:.1f}MB used (limit {MAX_MEMORY_MB}MB)"
                )
            if memory_mb > MAX_MEMORY_MB:
                consecutive_limit_hits += 1
                logger.warning(
                    f"Memory usage {memory_mb:.1f}MB exceeds limit {MAX_MEMORY_MB}MB. Triggering garbage collection "
                    f"(strike {consecutive_limit_hits})."
                )
                gc.collect()
                
                # Check again after GC
                memory_mb = get_memory_usage()
                if memory_mb > MAX_MEMORY_MB * 0.95:  # Still high after GC
                    logger.error(f"Memory usage still high after GC: {memory_mb:.1f}MB.")
                    if consecutive_limit_hits >= 3:
                        logger.critical(
                            f"Memory usage {memory_mb:.1f}MB stayed above limit for {consecutive_limit_hits} checks. "
                            "Initiating graceful shutdown before the OS kills the process."
                        )
                        graceful_shutdown()
                        os._exit(1)
                else:
                    logger.info(f"Memory usage after GC: {memory_mb:.1f}MB")
                    consecutive_limit_hits = 0
            else:
                consecutive_limit_hits = 0
            
            time.sleep(MEMORY_CHECK_INTERVAL)
        except Exception as e:
            logger.error(f"Error in memory monitor: {e}")
            time.sleep(MEMORY_CHECK_INTERVAL)

def start_memory_monitor():
    """Start the memory monitoring thread."""
    global _memory_monitor_thread
    if _memory_monitor_thread is None:
        _memory_monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        _memory_monitor_thread.start()
        logger.info(f"Started memory monitor. Max memory limit: {MAX_MEMORY_MB}MB")

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
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        max_memory_bytes = MAX_MEMORY_MB * 1024 * 1024
        hard_limit = current_hard if current_hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        # Only lower the hard limit if we already have a finite cap; otherwise leave it untouched.
        if hard_limit != resource.RLIM_INFINITY and max_memory_bytes > hard_limit:
            hard_limit = max_memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard_limit))
        logger.info(f"Set process memory limit to {MAX_MEMORY_MB}MB")
    except Exception as e:
        logger.warning(f"Could not set memory limit: {e}")

def graceful_shutdown(signum: Optional[int] = None, frame: Any = None):
    """Handle shutdown gracefully with proper cleanup."""
    global _shutting_down
    
    if _shutting_down:
        logger.debug("Shutdown already in progress, skipping duplicate")
        return
    
    _shutting_down = True
    
    if signum:
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    else:
        logger.info("Initiating graceful shutdown...")
    
    try:
        save_store()
        logger.info("Store saved successfully")
    except Exception as e:
        logger.error(f"Error saving store during shutdown: {e}", exc_info=True)
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
    """Upsert a single document into the store.

    IMPORTANT: The 'uri' must be an exact, literal string and must not be modified.
    Args:
        uri: Document identifier/path
        text: Document content
        
    Returns:
        dict with upserted status and whether document existed
    """
    try:
        logger.info(f"Upserting document: {uri}")
        normalized_uri = str(Path(uri))
        content = text or ""
        if not content.strip():
            # Fallback to reading from disk when the caller provided only a file path.
            file_path = Path(uri)
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    logger.debug(f"Read {len(content)} characters from {normalized_uri} for upsert")
                except Exception as file_err:
                    logger.error(f"Failed to read {normalized_uri}: {file_err}")
                    return {"error": f"Could not read file: {file_err}", "upserted": False}
            else:
                return {"error": f"Text missing and file not found: {normalized_uri}", "upserted": False}

        result: Dict[str, Any] = upsert_document(normalized_uri, content)
        logger.info(f"Successfully upserted document: {uri}")
        return result
    except Exception as e:
        logger.error(f"Error upserting document {uri}: {e}")
        return {"error": str(e), "upserted": False}

@mcp.tool()
def index_documents_tool(path: str, glob: str = "**/*.txt") -> Dict[str, Any]:
    """Index every file under `path` that matches `glob` by upserting each document."""
    try:
        base = Path(path)
        if not base.exists():
            return {"error": f"Path not found: {path}", "indexed": 0}
        
        files = list(base.rglob(glob))
        if not files:
            logger.info(f"No files found in {base} matching {glob}")
            return {"indexed": 0, "uris": []}
        
        indexed = 0
        indexed_uris = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as read_err:
                logger.warning(f"Skipping {file_path}: {read_err}")
                continue
            
            upsert_document(str(file_path), content)
            indexed += 1
            indexed_uris.append(str(file_path))
        
        logger.info(f"Indexed {indexed} documents from {base}")
        return {"indexed": indexed, "uris": indexed_uris}
    except Exception as e:
        logger.error(f"Error indexing path {path}: {e}")
        return {"error": str(e), "indexed": 0, "uris": []}

@mcp.tool()
def search_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for passages relevant to the query.
    
    Args:
        query: Search query string
        top_k: Number of top results to return (default: 5)
        
    Returns:
        dict with search results or error
    """
    try:
        logger.info(f"Searching for: {query}")
        result = search(query, top_k=top_k)
        logger.info(f"Search completed for: {query}")
        
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
        logger.error(f"Error searching for '{query}': {e}")
        return {"error": str(e)}

@mcp.tool()
def list_indexed_documents_tool() -> Dict[str, Any]:
    """List all document URIs currently in the store.

    Returns:
        dict with a list of document URIs or an error
    """
    try:
        logger.info("Listing indexed documents")
        store = get_store()
        if not store:
            return {"uris": [], "message": "Store not loaded or empty."}
        
        uris = list(store.docs.keys())
        logger.info(f"Found {len(uris)} indexed documents.")
        return {"uris": uris}
    except Exception as e:
        logger.error(f"Error listing indexed documents: {e}", exc_info=True)
        return {"error": str(e), "uris": []}

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
        logger.info(f"Starting MCP server on {mcp.settings.host}:{mcp.settings.port}{mcp.settings.streamable_http_path}")
        logger.info(f"Current memory usage: {memory_mb:.1f}MB (limit: {MAX_MEMORY_MB}MB)")
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
        logger.error(f"Fatal server error: {e}", exc_info=True)
        graceful_shutdown()
        sys.exit(1)
