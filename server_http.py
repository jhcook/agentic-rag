#!/usr/bin/env python3
"""
MCP Retrieval Server — FastMCP Streamable HTTP

Env:ironment Variables:
    MCP_PATH: HTTP path for FastMCP (default: /mcp)
    MCP_HOST: 127.0.0.1
    MCP_PORT: 8000"

Run (dev):
    uv run python server_http.py
    # or
    python server_http.py

Connect URL (client side):
    http://127.0.0.1:8000/mcp

Notes:
- Uses FastMCP streamable HTTP transport (recommended over SSE).
- For browser clients, configure CORS (see below).
"""

from __future__ import annotations
import os
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# import shared logic
from rag_core import (index_documents, index_path, search, rerank, grounded_answer,
                      verify_grounding, load_store, save_store, upsert_document)

# On exit save the store
import atexit
atexit.register(save_store)

mcp = FastMCP("retrieval-server")

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

@mcp.tool()
def rerank_tool(query: str, passages: List[Dict[str, Any]], model: str = "cross-encoder-mini"):
    """Rerank passages based on their relevance to the query."""
    return rerank(query, passages, model)

@mcp.tool()
def grounded_answer_tool(query: str, passages: List[Dict[str, Any]] | None = None, k: int = 8):
    """Generate a grounded answer based on the query and passages."""
    return grounded_answer(query, passages, k)

@mcp.tool()
def verify_grounding_tool(query: str, answer: str, citations: List[str] | None = None):
    """Verify the grounding of the answer based on the citations."""
    return verify_grounding(query, answer, citations)

if __name__ == "__main__":
    mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
    load_store()
    mcp.run(transport="streamable-http")
