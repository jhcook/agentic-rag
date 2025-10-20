#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

# rest_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from rag_core import (index_path, search, rerank, grounded_answer,
                      verify_grounding, upsert_document)

app = FastAPI(title="retrieval-server REST")

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

# rest_server.py (new model + route)
class UpsertReq(BaseModel):
    uri: str
    text: str

@app.post("/api/upsert_document")
def api_upsert(req: UpsertReq):
    return upsert_document(req.uri, req.text)

@app.post("/api/index_path")
def api_index_path(req: IndexPathReq):
    return index_path(req.path, req.glob)

@app.post("/api/search")
def api_search(req: SearchReq):
    return search(req.query, req.k, req.hybrid)

@app.post("/api/rerank")
def api_rerank(req: RerankReq):
    return rerank(req.query, req.passages, req.model)

@app.post("/api/grounded_answer")
def api_grounded(req: GroundedReq):
    return grounded_answer(req.query, req.passages, req.k)

@app.post("/api/verify_grounding")
def api_verify(req: VerifyReq):
    return verify_grounding(req.query, req.answer, req.citations)
