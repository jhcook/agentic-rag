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
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import shared logic
from rag_core import (index_documents, index_path, search, load_store, save_store, search,
                      upsert_document)

# Create FastMCP instance
mcp = FastMCP("retrieval-server")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Saving store and shutting down...")
    save_store()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register store save on exit
atexit.register(save_store)

@mcp.tool()
def upsert_document_tool(uri: str, text: str) -> dict:
    """Upsert a single document into the store."""
    return upsert_document(uri, text)

@mcp.tool()
def index_documents_tool(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs."""
    return index_documents(uris)

@mcp.tool()
def index_path_tool(path: str, glob: str = "**/*.txt") -> Dict[str, Any]:
    """Index all text files in a given path matching the glob pattern."""
    return index_path(path, glob)

@mcp.tool()
def search_tool(query: str, k: int = 12, hybrid: bool = True):
    """Search for passages relevant to the query."""
    return search(query, k, hybrid)

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
        
        # Start server
        mcp.run(
            transport="streamable-http")
    except Exception as e:
        logger.error(f"Server error: {e}")
        save_store()
        sys.exit(1)
