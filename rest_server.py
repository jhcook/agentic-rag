#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os, sys, logging, asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from rag_core import (index_path, search, upsert_document, send_store_to_llm)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get base path
pth = os.getenv("RAG_PATH", "api")

app = FastAPI(title="retrieval-rest-server")

class IndexPathReq(BaseModel):
    path: str
    glob: Optional[str] = "**/*.txt"

class SearchReq(BaseModel):
    query: str

class UpsertReq(BaseModel):
    uri: str
    text: str

class LoadStoreReq(BaseModel):
    _ : Optional[bool] = True

@app.post(f"/{pth}/upsert_document")
def api_upsert(req: UpsertReq):
    return upsert_document(req.uri, req.text)

@app.post(f"/{pth}/index_path")
def api_index_path(req: IndexPathReq):
    return index_path(req.path, req.glob)

@app.post(f"/{pth}/search")
def api_search(req: SearchReq):
    return search(req.query)

@app.post(f"/{pth}/load_store")
def api_load(req: LoadStoreReq):
    store = send_store_to_llm()
    return {"status": "store sent to LLM", "store_summary": store}

if __name__ == "__main__":
    try:
        # Configure app settings
        app.host = os.getenv("RAG_HOST", "127.0.0.1")
        app.port = int(os.getenv("RAG_PORT", "8001"))
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)