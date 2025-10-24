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
import asyncio
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server_http.log'),
        logging.StreamHandler()  # Keep console output as well
    ]
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
    loop = asyncio.run(upsert_document(uri, text))
    asyncio.get_event_loop().set_debug(True)
    return loop

@mcp.tool()
def index_documents_tool(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs."""
    loop = asyncio.run(index_documents(uris))
    asyncio.get_event_loop().set_debug(True)
    return loop

@mcp.tool()
def index_path_tool(path: str, glob: str = "**/*.txt"):
    """Index all text files in path that match the pattern glob."""
    return index_path(path, glob)     

@mcp.tool()
def search_tool(query: str):
    """Search for passages relevant to the query."""
    resp = search(query)
    #contents = [choice["message"]["content"] for choice in resp["response"]["choices"]]
    #return "\n".join(contents)
    return resp


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
