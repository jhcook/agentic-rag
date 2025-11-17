#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os, sys, logging, time
import psutil
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import Optional
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.core.rag_core import (index_path, search, upsert_document, send_store_to_llm, get_store)

# Set up logging
os.makedirs('log', exist_ok=True)

# Process logger (application logs)
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

# Access logger (HTTP access logs)
access_logger = logging.getLogger('rest_access')
access_logger.setLevel(logging.INFO)
access_handler = logging.FileHandler('log/rest_server_access.log')
access_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
access_logger.addHandler(access_handler)
access_logger.propagate = False  # Don't propagate to root logger

# Ensure log file flushes immediately
sys.stdout.flush()
sys.stderr.flush()

# Get base path
pth = os.getenv("RAG_PATH", "api")

app = FastAPI(title="retrieval-rest-server")

# Prometheus metrics (REST)
HTTP_REQUESTS_TOTAL = Counter(
    "rest_http_requests_total",
    "Total HTTP requests served by the REST API.",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "rest_http_request_duration_seconds",
    "Latency for REST requests in seconds.",
    ["method", "path", "status"],
    buckets=[0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
HTTP_INFLIGHT = Gauge(
    "rest_inflight_requests",
    "In-flight REST requests.",
)
REST_DOCUMENTS_INDEXED = Gauge(
    "rest_documents_indexed_total",
    "Number of documents indexed in the REST store.",
)
REST_MEMORY_USAGE_MB = Gauge(
    "rest_memory_usage_megabytes",
    "Current REST server process RSS memory in MB.",
)
REST_EMBEDDING_VECTORS = Gauge(
    "rest_embedding_vectors_total",
    "Total vectors stored in the embedding index (REST server view).",
)
REST_EMBEDDING_CHUNKS = Gauge(
    "rest_embedding_chunks_total",
    "Total text chunks tracked for embeddings (REST server view).",
)
REST_EMBEDDING_DIM = Gauge(
    "rest_embedding_dimension",
    "Embedding dimension used by the current model (REST server view).",
)

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

def _get_route_path_template(request: Request) -> str:
    """Return the path template to keep label cardinality bounded."""
    route = request.scope.get("route")
    template = getattr(route, "path", None) if route else None
    return str(template or request.url.path)

def _get_memory_usage_mb() -> float:
    """Return current process RSS in megabytes."""
    return psutil.Process().memory_info().rss / 1024 / 1024

def refresh_prometheus_metrics():
    """Refresh gauges reflecting store/index state and process usage."""
    try:
        store = get_store()
        REST_DOCUMENTS_INDEXED.set(len(getattr(store, "docs", {})))
    except Exception as exc:
        logger.debug("refresh metrics: document gauge failed: %s", exc)

    try:
        REST_MEMORY_USAGE_MB.set(_get_memory_usage_mb())
    except Exception as exc:
        logger.debug("refresh metrics: memory gauge failed: %s", exc)

    try:
        from src.core.rag_core import get_faiss_globals

        index, index_to_meta, embed_dim = get_faiss_globals()
        total_vectors = index.ntotal if index is not None else 0  # type: ignore[attr-defined]
        REST_EMBEDDING_VECTORS.set(total_vectors)
        REST_EMBEDDING_CHUNKS.set(len(index_to_meta) if index_to_meta is not None else 0)
        REST_EMBEDDING_DIM.set(embed_dim or 0)
    except Exception as exc:
        logger.debug("refresh metrics: embedding gauges failed: %s", exc)

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    """Record HTTP metrics and access logs for each request."""
    start = time.perf_counter()
    method = request.method.upper()
    path = _get_route_path_template(request)
    HTTP_INFLIGHT.inc()
    status_code = "500"

    try:
        response = await call_next(request)
        status_code = str(getattr(response, "status_code", 500))

        # Log access in Apache combined log format style
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        query = str(request.url.query) if request.url.query else ''
        full_path = f"{request.url.path}?{query}" if query else request.url.path
        user_agent = request.headers.get('user-agent', '-')
        content_length = response.headers.get('content-length', '-')
        duration = time.perf_counter() - start

        access_logger.info(
            f'{client_ip} - - [{time.strftime("%d/%b/%Y:%H:%M:%S %z", time.localtime())}] '
            f'"{method} {full_path} HTTP/{request.scope.get("http_version", "1.1")}" '
            f'{status_code} {content_length} "-" "{user_agent}" {duration:.4f}s'
        )

        return response
    except Exception:
        status_code = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_INFLIGHT.dec()
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(method=method, path=path, status=status_code).observe(duration)

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

@app.get("/metrics")
def metrics_endpoint():
    """Expose Prometheus metrics for the REST server."""
    refresh_prometheus_metrics()
    payload = generate_latest()
    return Response(payload, media_type=CONTENT_TYPE_LATEST)

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
