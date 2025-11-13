#!/usr/bin/env python3
"""
MCP Retrieval Server — FastMCP Streamable HTTP

Environment Variables:
    MCP_PATH: HTTP path for FastMCP (default: /mcp)
    MCP_HOST: Address to listen     (default: 127.0.0.1)
    MCP_PORT: TCP port to listen    (default: 8000)

Run (dev):
    uv run python server_http.py
    # or
    python server_http.py

Connect URL (client side):
    http://127.0.0.1:8000/mcp
"""

from __future__ import annotations
import atexit
import os
import signal
import sys
import logging
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Set up logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/server_http.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import shared logic
from rag_core import (
    index_documents, 
    index_path, 
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

def graceful_shutdown(signum=None, frame=None):
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
def upsert_document_tool(uri: str, text: str) -> dict:
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
        result = upsert_document(uri, text)
        logger.info(f"Successfully upserted document: {uri}")
        return result
    except Exception as e:
        logger.error(f"Error upserting document {uri}: {e}")
        return {"error": str(e), "upserted": False}

@mcp.tool()
def add_documents_to_index_tool(uris: List[str]) -> Dict[str, Any]:
    """Add a list of local document file paths to the index.
def index_documents_tool(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs.
    
    Args:
        uris: List of file paths to index
        
    Returns:
        dict with count of indexed documents
    """
    try:
        logger.info(f"Indexing {len(uris)} documents")
        result = index_documents(uris)
        logger.info(f"Successfully indexed {result.get('indexed', 0)} documents")
        return result
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        return {"error": str(e), "indexed": 0}

@mcp.tool()
def index_path_tool(path: str, glob: str = "**/*.txt") -> Optional[Dict[str, Any]]:
    """Index all text files in path that match the pattern glob.
    
    Args:
        path: Directory path to search
        glob: File pattern to match (default: **/*.txt)
        
    Returns:
        dict with indexed count and total vectors
    """
    try:
        logger.info(f"Indexing path: {path} with pattern: {glob}")
        result = index_path(path, glob)
        logger.info(f"Successfully indexed path: {path}")
        return result
    except Exception as e:
        logger.error(f"Error indexing path {path}: {e}")
        return {"error": str(e), "indexed": 0, "total_vectors": 0}

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
        return result
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
        # Configure MCP
        mcp.settings.log_level = "debug"
        mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
        
        # Load store
        logger.info("Loading document store...")
        load_store()
        logger.info("Document store loaded successfully")
        
        # Log server configuration
        logger.info(f"Starting MCP server on {mcp.settings.host}:{mcp.settings.port}{mcp.settings.streamable_http_path}")
        logger.info("Press Ctrl+C to gracefully shutdown")
        
        # Start server
        mcp.run(transport="streamable-http")
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Fatal server error: {e}", exc_info=True)
        graceful_shutdown()
        sys.exit(1)