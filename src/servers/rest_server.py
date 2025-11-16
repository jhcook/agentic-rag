#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os, sys, logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.core.rag_core import (index_path, search, upsert_document, send_store_to_llm, get_store)

# Set up logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/rest_server.log'),
        logging.StreamHandler()
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# Ensure log file flushes immediately
sys.stdout.flush()
sys.stderr.flush()

# Get base path
pth = os.getenv("RAG_PATH", "api")

app = FastAPI(title="retrieval-rest-server")

# Log startup information
logger.info(f"REST server initialized with base path: /{pth}")
logger.info(f"Log file: log/rest_server.log")

# Load the document store and rebuild FAISS index on module import
logger.info("Loading document store and rebuilding search index...")
store = get_store()
logger.info(f"Document store loaded with {len(store.docs)} documents")

class IndexPathReq(BaseModel):
    """Request model for indexing a filesystem path."""
    path: str
    glob: Optional[str] = "**/*.txt"

class SearchReq(BaseModel):
    """Request model for performing a search."""
    query: str

class UpsertReq(BaseModel):
    """Request model for upserting a document."""
    uri: str
    text: str

class LoadStoreReq(BaseModel):
    """Request model for sending the store to an LLM."""
    _ : Optional[bool] = True

@app.post(f"/{pth}/upsert_document")
def api_upsert(req: UpsertReq):
    """Upsert a document into the store."""
    logger.info(f"Upserting document: uri={req.uri}")
    import sys
    sys.stdout.flush()
    try:
        result = upsert_document(req.uri, req.text)
        logger.info(f"Successfully upserted document: uri={req.uri}")
        sys.stdout.flush()
        return result
    except Exception as e:
        logger.error(f"Error upserting document {req.uri}: {e}")
        sys.stdout.flush()
        raise

@app.post(f"/{pth}/index_path")
def api_index_path(req: IndexPathReq):
    """Index a filesystem path into the retrieval store."""
    logger.info(f"Indexing path: path={req.path}, glob={req.glob}")
    import sys
    sys.stdout.flush()
    try:
        result = index_path(req.path, req.glob)
        logger.info(f"Successfully indexed path: {req.path}")
        sys.stdout.flush()
        return result
    except Exception as e:
        logger.error(f"Error indexing path {req.path}: {e}")
        sys.stdout.flush()
        raise

@app.post(f"/{pth}/search")
def api_search(req: SearchReq):
    """Search the retrieval store."""
    logger.info(f"Processing search query: {req.query}")
    import sys
    sys.stdout.flush()
    try:
        result = search(req.query)
        logger.info(f"Search completed successfully for query: {req.query}")
        sys.stdout.flush()
        return result
    except Exception as e:
        logger.error(f"Error processing search query '{req.query}': {e}")
        sys.stdout.flush()
        raise

@app.post(f"/{pth}/load_store")
def api_load(req: LoadStoreReq):
    """Send the current store to the LLM."""
    logger.info("Loading store to LLM")
    try:
        store = send_store_to_llm()
        logger.info("Store successfully sent to LLM")
        return {"status": "store sent to LLM", "store_summary": store}
    except Exception as e:
        logger.error(f"Error loading store to LLM: {e}")
        raise

if __name__ == "__main__":
    try:
        # Configure app settings
        app.host = os.getenv("RAG_HOST", "127.0.0.1")
        app.port = int(os.getenv("RAG_PORT", "8001"))
        
        logger.info(f"Starting REST server on {app.host}:{app.port}")
        logger.info(f"API base path: /{pth}")
        
        # Load the document store and rebuild FAISS index on startup
        logger.info("Loading document store and rebuilding search index...")
        store = get_store()
        logger.info(f"Document store loaded with {len(store.docs)} documents")

    except Exception as e:
        logger.error(f"Server startup error: {e}")
        sys.exit(1)
