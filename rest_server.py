#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os, sys, logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from rag_core import (index_path, search, rerank, grounded_answer,
                      verify_grounding, upsert_document)

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
    k: int = 12
    hybrid: bool = True

class RerankReq(BaseModel):
    query: str
    passages: List[Dict[str, Any]]
    model: str = "cross-encoder-mini"

class GroundedReq(BaseModel):
    query: str
    passages: Optional[List[Dict[str, Any]]] = None
    k: int = 8

class VerifyReq(BaseModel):
    query: str
    answer: str
    citations: Optional[List[str]] = None

# ----- model + route -----
class UpsertReq(BaseModel):
    uri: str
    text: str

@app.post(f"/{pth}/upsert_document")
def api_upsert(req: UpsertReq):
    return upsert_document(req.uri, req.text)

@app.post(f"/{pth}/index_path")
def api_index_path(req: IndexPathReq):
    return index_path(req.path, req.glob)

@app.post(f"/{pth}/search")
def api_search(req: SearchReq):
    return search(req.query, req.k, req.hybrid)

@app.post(f"/{pth}/rerank")
def api_rerank(req: RerankReq):
    return rerank(req.query, req.passages, req.model)

@app.post(f"/{pth}/grounded_answer")
def api_grounded(req: GroundedReq):
    return grounded_answer(req.query, req.passages, req.k)

@app.post(f"/{pth}/verify_grounding")
def api_verify(req: VerifyReq):
    return verify_grounding(req.query, req.answer, req.citations)

if __name__ == "__main__":
    try:
        # Configure app settings
        app.host = os.getenv("RAG_HOST", "127.0.0.1")
        app.port = int(os.getenv("RAG_PORT", "8001"))
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)