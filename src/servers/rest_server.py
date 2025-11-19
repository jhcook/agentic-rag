#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os, sys, logging, time, threading, queue
from pathlib import Path
from datetime import datetime
import psutil
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Set up logging
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Process logger (application logs)
# Configure root logger explicitly to avoid duplicate handlers
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log_handlers = [
    logging.FileHandler(LOG_DIR / 'rest_server.log'),
]
for _h in log_handlers:
    _h.setFormatter(formatter)
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.setLevel(logging.INFO)
for h in log_handlers:
    root_logger.addHandler(h)
root_logger.propagate = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Let this module logger bubble up to root handlers configured above
logger.handlers.clear()
logger.propagate = True

# Access logger (HTTP access logs)
access_logger = logging.getLogger('rest_access')
access_logger.setLevel(logging.INFO)
access_logger.handlers.clear()
access_logger.propagate = False

# Explicitly disable uvicorn's default access logger so access lines don't hit rest_server.log
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers.clear()
uvicorn_access_logger.propagate = False
uvicorn_access_logger.disabled = True
# Also tame uvicorn's default logger to avoid double writes
uvicorn_error_logger = logging.getLogger("uvicorn")
uvicorn_error_logger.handlers.clear()
uvicorn_error_logger.propagate = False
uvicorn_error_logger.disabled = True

# Ensure log file flushes immediately
sys.stdout.flush()
sys.stderr.flush()

# Get base path
pth = os.getenv("RAG_PATH", "api")

app = FastAPI(title="retrieval-rest-server")

# Allow cross-origin calls from mobile/web clients (permissive by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
REST_MEMORY_LIMIT_MB = Gauge(
    "rest_memory_limit_megabytes",
    "Configured memory limit for REST server in MB.",
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
logger.info(f"Log file: {LOG_DIR / 'rest_server.log'}")


@app.on_event("startup")
async def _on_startup():
    logger.info("REST server startup complete (host=%s port=%s base=/api)", os.getenv("RAG_HOST", "127.0.0.1"), os.getenv("RAG_PORT", "8001"))


@app.on_event("shutdown")
async def _on_shutdown():
    logger.info("REST server shutting down")

class IndexPathReq(BaseModel):
    """Request model for indexing a filesystem path."""
    path: str
    glob: Optional[str] = "**/*.txt"

class SearchReq(BaseModel):
    """Request model for performing a search."""
    query: str
    async_mode: bool = Field(default=False, alias="async")
    timeout_seconds: Optional[int] = 300

    model_config = ConfigDict(populate_by_name=True)

class UpsertReq(BaseModel):
    """Request model for upserting a document."""
    uri: str
    text: Optional[str] = None
    binary_base64: Optional[str] = None

class LoadStoreReq(BaseModel):
    """Request model for sending the store to an LLM."""
    _ : Optional[bool] = True

class GroundedAnswerReq(BaseModel):
    """Request model for grounded answer generation."""
    question: str
    k: Optional[int] = 3

class RerankReq(BaseModel):
    """Request model for reranking passages."""
    query: str
    passages: list[dict]
    top_k: Optional[int] = None

class VerifyReq(BaseModel):
    """Request model for grounding verification (citations)."""
    question: str
    draft_answer: str
    citations: list[str]

class VerifySimpleReq(BaseModel):
    """Request model for grounding verification (passages provided)."""
    question: str
    draft_answer: str
    passages: list[dict]

class HealthResp(BaseModel):
    """Response model for health checks."""
    status: str
    base_path: str
    documents: int
    vectors: int
    memory_mb: float
    memory_limit_mb: int
    total_size_bytes: int = 0
    store_file_bytes: int = 0

class DocumentInfo(BaseModel):
    """Document metadata for listings."""
    uri: str
    size_bytes: int

class DocumentsResp(BaseModel):
    documents: list[DocumentInfo]

class DeleteDocsReq(BaseModel):
    """Request model for deleting documents by URI."""
    uris: list[str]

class QualityMetricsResp(BaseModel):
    """Aggregated quality metrics for searches."""
    total_searches: int
    failed_searches: int
    responses_with_sources: int
    total_sources: int
    fallback_responses: int
    success_rate: float
    avg_sources: float

class Job(BaseModel):
    id: str
    type: str
    status: str
    error: Optional[str] = None
    result: Optional[dict] = None


# Search job queue (async search to avoid blocking HTTP while LLM works)
SEARCH_JOBS: dict[str, dict] = {}
SEARCH_JOBS_LOCK = threading.Lock()
SEARCH_JOB_QUEUE: queue.Queue = queue.Queue()
SEARCH_WORKER_STARTED = False


def _start_search_worker():
    global SEARCH_WORKER_STARTED
    if SEARCH_WORKER_STARTED:
        return

    def _worker():
        while True:
            job = SEARCH_JOB_QUEUE.get()
            if not job:
                continue
            job_id = job["id"]
            try:
                result = _proxy_to_mcp("POST", "/rest/search", {"query": job["query"]})
                status = "completed"
                error = None
            except Exception as exc:
                result = None
                status = "failed"
                error = str(exc)
            with SEARCH_JOBS_LOCK:
                SEARCH_JOBS[job_id].update({"status": status, "result": result, "error": error})
            SEARCH_JOB_QUEUE.task_done()

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    SEARCH_WORKER_STARTED = True


# MCP proxy configuration
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = os.getenv("MCP_PORT", "8000")
MCP_PATH = os.getenv("MCP_PATH", "/mcp")

def _mcp_base() -> str:
    prefix = MCP_PATH.strip("/")
    base = f"http://{MCP_HOST}:{MCP_PORT}"
    return f"{base}/{prefix}" if prefix else base

def _proxy_to_mcp(method: str, path: str, json_payload: Optional[dict] = None):
    """
    Proxy to MCP; tries prefixed path first, then fallback to root path if 404.
    Raises HTTPException on failure so callers can return proper errors.
    """
    prefix_base = _mcp_base()
    urls = []
    if path.startswith("/"):
        urls.append(f"{prefix_base}{path}")
        if prefix_base.endswith("/"):
            urls.append(f"{prefix_base[:-1]}{path}")
    else:
        urls.append(f"{prefix_base}/{path}")
    # Fallback to host root if prefix path not found
    urls.append(f"http://{MCP_HOST}:{MCP_PORT}{path if path.startswith('/') else '/' + path}")

    last_exc = None
    # Allow long-running searches: bump timeout for /search
    timeout = 300 if "/search" in path else 30

    for url in urls:
        try:
            resp = requests.request(method, url, json=json_payload, timeout=timeout)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            try:
                return resp.json()
            except requests.exceptions.JSONDecodeError:
                return resp.text
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as exc:
            last_exc = exc
            continue
    logger.error("MCP proxy failed for %s %s: %s", method, path, last_exc or "no response")
    detail = {"error": "MCP unavailable", "detail": str(last_exc) if last_exc else "no response", "path": path}
    raise HTTPException(status_code=502, detail=detail)
def _get_route_path_template(request: Request) -> str:
    """Return the path template to keep label cardinality bounded."""
    route = request.scope.get("route")
    template = getattr(route, "path", None) if route else None
    return str(template or request.url.path)

def _get_system_memory_mb() -> float:
    """Return available system memory in megabytes."""
    return psutil.virtual_memory().available / 1024 / 1024

def _get_memory_limit_mb() -> int:
    """Return configured memory limit (env REST_MAX_MEMORY_MB/MAX_MEMORY_MB or 75% available)."""
    env_val = os.getenv("REST_MAX_MEMORY_MB") or os.getenv("MAX_MEMORY_MB")
    try:
        return int(env_val) if env_val is not None else int(_get_system_memory_mb() * 0.75)
    except (TypeError, ValueError):
        return int(_get_system_memory_mb() * 0.75)

REST_MAX_MEMORY_MB = _get_memory_limit_mb()

def _get_memory_usage_mb() -> float:
    """Return current process RSS in megabytes."""
    return psutil.Process().memory_info().rss / 1024 / 1024

# Quality metrics counters (process lifetime)
QUALITY_TOTAL_SEARCHES = 0
QUALITY_FAILED_SEARCHES = 0
QUALITY_TOTAL_SOURCES = 0
QUALITY_RESPONSES_WITH_SOURCES = 0
QUALITY_FALLBACK_RESPONSES = 0

def _record_quality_metrics(result: Optional[dict], error: Optional[Exception] = None) -> None:
    """Update quality counters based on a search result or error."""
    global QUALITY_TOTAL_SEARCHES, QUALITY_FAILED_SEARCHES, QUALITY_TOTAL_SOURCES, QUALITY_RESPONSES_WITH_SOURCES, QUALITY_FALLBACK_RESPONSES
    QUALITY_TOTAL_SEARCHES += 1

    if error is not None:
        QUALITY_FAILED_SEARCHES += 1
        return

    if not isinstance(result, dict):
        QUALITY_FAILED_SEARCHES += 1
        return

    if "error" in result:
        QUALITY_FAILED_SEARCHES += 1
        return

    sources = result.get("sources") or []
    if isinstance(sources, list):
        QUALITY_TOTAL_SOURCES += len(sources)
        if len(sources) > 0:
            QUALITY_RESPONSES_WITH_SOURCES += 1

    if result.get("warning"):
        QUALITY_FALLBACK_RESPONSES += 1

def refresh_prometheus_metrics():
    """Refresh gauges reflecting proxy state and process usage."""
    try:
        REST_MEMORY_USAGE_MB.set(_get_memory_usage_mb())
        REST_MEMORY_LIMIT_MB.set(REST_MAX_MEMORY_MB)
    except Exception as exc:
        logger.debug("refresh metrics: memory gauge failed: %s", exc)

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

        # Write directly to access log file instead of using logger
        log_entry = (
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]} - '
            f'{client_ip} - - [{time.strftime("%d/%b/%Y:%H:%M:%S %z", time.localtime())}] '
            f'"{method} {full_path} HTTP/{request.scope.get("http_version", "1.1")}" '
            f'{status_code} {content_length} "-" "{user_agent}" {duration:.4f}s\n'
        )

        _append_access_log(log_entry)

        return response
    except Exception:
        status_code = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_INFLIGHT.dec()
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(method=method, path=path, status=status_code).observe(duration)

ACCESS_LOG_PATH = LOG_DIR / "rest_server_access.log"

def _append_access_log(log_entry: str) -> None:
    """Write and flush an access log line for reliable tail/follow."""
    with open(ACCESS_LOG_PATH, "a", buffering=1) as f:
        f.write(log_entry)
        f.flush()
        os.fsync(f.fileno())

@app.post(f"/{pth}/upsert_document")
def api_upsert(req: UpsertReq):
    """Upsert a document into the store."""
    logger.info(f"Upserting document: uri={req.uri}")
    try:
        if not req.text and not req.binary_base64:
            return JSONResponse({"error": "either text or binary_base64 is required"}, status_code=422)
        return _proxy_to_mcp("POST", "/rest/upsert_document", {"uri": req.uri, "text": req.text, "binary_base64": req.binary_base64})
    except Exception as e:
        logger.error(f"Error upserting document {req.uri}: {e}")
        raise

@app.post(f"/{pth}/index_path")
def api_index_path(req: IndexPathReq):
    """Index a filesystem path into the retrieval store."""
    logger.info(f"Indexing path: path={req.path}, glob={req.glob}")
    try:
        return _proxy_to_mcp("POST", "/rest/index_path", {"path": req.path, "glob": req.glob})
    except Exception as e:
        logger.error(f"Error indexing path {req.path}: {e}")
        raise

@app.post(f"/{pth}/search")
def api_search(req: SearchReq):
    """Search the retrieval store."""
    logger.info(f"Processing search query: {req.query}")
    # Async mode: queue job and return immediately
    if getattr(req, "async_mode", False):
        import uuid
        _start_search_worker()
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": "search",
            "status": "queued",
            "query": req.query,
            "timeout_seconds": req.timeout_seconds or 300,
        }
        with SEARCH_JOBS_LOCK:
            SEARCH_JOBS[job_id] = job
        SEARCH_JOB_QUEUE.put(job)
        return {"job_id": job_id, "status": "queued"}

    try:
        result = _proxy_to_mcp("POST", "/rest/search", {"query": req.query})
        _record_quality_metrics(result)
        return result
    except HTTPException as e:
        _record_quality_metrics(None, error=e)
        raise
    except Exception as e:
        logger.error(f"Error processing search query '{req.query}': {e}")
        _record_quality_metrics(None, error=e)
        raise HTTPException(status_code=502, detail={"error": "search failed", "detail": str(e)})

@app.post(f"/{pth}/load_store")
def api_load(req: LoadStoreReq):
    """Send the current store to the LLM."""
    logger.info("Loading store to LLM")
    try:
        # Delegate to MCP
        return _proxy_to_mcp("POST", "/rest/flush_cache", {})  # noop placeholder
    except Exception as e:
        logger.error(f"Error loading store to LLM: {e}")
        raise

@app.get("/metrics")
def metrics_endpoint():
    """Expose Prometheus metrics for the REST server."""
    refresh_error = None
    try:
        refresh_prometheus_metrics()
    except Exception as exc:
        refresh_error = exc
        logger.error("Metrics refresh failed (serving last known values): %s", exc, exc_info=True)
    try:
        payload = generate_latest()
        if refresh_error:
            payload += f"# metrics refresh error: {refresh_error}\n".encode()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)
    except Exception as exc:
        logger.error("Failed to render metrics: %s", exc, exc_info=True)
        return Response(
            f"# metrics unavailable: {exc}\n", status_code=200, media_type="text/plain"
        )

@app.post(f"/{pth}/grounded_answer")
def api_grounded_answer(req: GroundedAnswerReq):
    """Return a grounded answer using vector search + synthesis."""
    logger.info("Grounded answer requested: %s", req.question)
    try:
        answer = grounded_answer(req.question, k=req.k or 3)
        return answer
    except Exception as exc:
        logger.error("grounded_answer failed: %s", exc)
        raise

@app.post(f"/{pth}/rerank")
def api_rerank(req: RerankReq):
    """Rerank provided passages for a query."""
    logger.info("Rerank requested for query: %s", req.query)
    try:
        ranked = rerank(req.query, req.passages)
        if req.top_k:
            ranked = ranked[: req.top_k]
        return {"results": ranked}
    except Exception as exc:
        logger.error("rerank failed: %s", exc)
        raise

@app.post(f"/{pth}/verify_grounding")
def api_verify(req: VerifyReq):
    """Verify grounding against store using citation URIs."""
    logger.info("Verify grounding requested")
    try:
        return verify_grounding(req.question, req.draft_answer, req.citations)
    except Exception as exc:
        logger.error("verify_grounding failed: %s", exc)
        raise

@app.post(f"/{pth}/verify_grounding_simple")
def api_verify_simple(req: VerifySimpleReq):
    """Simple grounding verification against provided passages."""
    logger.info("Verify grounding (simple) requested")
    try:
        return verify_grounding_simple(req.question, req.draft_answer, req.passages)
    except Exception as exc:
        logger.error("verify_grounding_simple failed: %s", exc)
        raise

@app.get(f"/{pth}/health", response_model=HealthResp)
def api_health():
    """Lightweight health check with basic store and index stats."""
    try:
        data = _proxy_to_mcp("GET", "/rest/health")
        return HealthResp(
            status=data.get("status", "ok"),
            base_path=f"/{pth}",
            documents=int(data.get("documents", 0)),
            vectors=int(data.get("vectors", 0)),
            memory_mb=float(data.get("memory_mb", 0)),
            memory_limit_mb=int(data.get("memory_limit_mb", REST_MAX_MEMORY_MB)),
            total_size_bytes=int(data.get("total_size_bytes", 0)),
            store_file_bytes=int(data.get("store_file_bytes", 0)),
        )
    except HTTPException as exc:
        logger.error("Health check failed: %s", exc.detail)
        raise
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=502, detail={"error": "MCP unreachable", "detail": str(exc)})

@app.get(f"/{pth}/documents", response_model=DocumentsResp)
def api_documents():
    """List indexed document URIs and their approximate text size (bytes)."""
    try:
        return _proxy_to_mcp("GET", "/rest/documents")
    except HTTPException as exc:
        logger.error("documents list failed: %s", exc.detail)
        raise
    except Exception as exc:
        logger.error("documents list failed: %s", exc)
        raise HTTPException(status_code=502, detail={"error": "MCP unreachable", "detail": str(exc)})

@app.post(f"/{pth}/documents/delete")
def api_documents_delete(req: DeleteDocsReq):
    """Delete documents by URI from the store and rebuild index."""
    try:
        return _proxy_to_mcp("POST", "/rest/documents/delete", {"uris": req.uris})
    except Exception as exc:
        logger.error("documents delete failed: %s", exc)
        return {"deleted": 0, "error": str(exc)}

@app.get(f"/{pth}/jobs", response_model=dict)
def api_jobs():
    """List async indexing jobs (pass-through to MCP)."""
    try:
        return _proxy_to_mcp("GET", "/rest/jobs")
    except HTTPException as exc:
        logger.error("jobs list failed: %s", exc.detail)
        raise
    except Exception as exc:
        logger.error("jobs list failed: %s", exc)
        raise HTTPException(status_code=502, detail={"error": "MCP unreachable", "detail": str(exc)})

@app.get(f"/{pth}/jobs/{{job_id}}", response_model=dict)
def api_job(job_id: str):
    """Get a single async indexing job (pass-through to MCP)."""
    try:
        return _proxy_to_mcp("GET", f"/rest/jobs/{job_id}")
    except HTTPException as exc:
        logger.error("job lookup failed: %s", exc.detail)
        raise
    except Exception as exc:
        logger.error("job lookup failed: %s", exc)
        raise HTTPException(status_code=502, detail={"error": "MCP unreachable", "detail": str(exc)})

@app.get(f"/{pth}/search/jobs/{{job_id}}", response_model=dict)
def api_search_job(job_id: str):
    """Get status/result for an async search job."""
    with SEARCH_JOBS_LOCK:
        job = SEARCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "not found", "id": job_id})
    return job

@app.post(f"/{pth}/flush_cache")
def api_flush_cache():
    """Flush the document store and delete the backing DB file."""
    try:
        return _proxy_to_mcp("POST", "/rest/flush_cache")
    except Exception as exc:
        logger.error("flush_cache failed: %s", exc)
        return Response(status_code=500, content=f"flush failed: {exc}")

@app.get(f"/{pth}/metrics/quality", response_model=QualityMetricsResp)
def api_quality_metrics():
    """Return aggregated quality metrics for searches."""
    total = QUALITY_TOTAL_SEARCHES
    failed = QUALITY_FAILED_SEARCHES
    with_sources = QUALITY_RESPONSES_WITH_SOURCES
    total_sources = QUALITY_TOTAL_SOURCES
    fallback = QUALITY_FALLBACK_RESPONSES
    success_rate = 0.0 if total == 0 else float((total - failed) / total)
    avg_sources = 0.0 if total == 0 else float(total_sources / max(1, total))
    return {
        "total_searches": total,
        "failed_searches": failed,
        "responses_with_sources": with_sources,
        "total_sources": total_sources,
        "fallback_responses": fallback,
        "success_rate": success_rate,
        "avg_sources": avg_sources,
    }

if __name__ == "__main__":
    try:
        # Configure app settings
        app.host = os.getenv("RAG_HOST", "127.0.0.1")
        app.port = int(os.getenv("RAG_PORT", "8001"))
        
        logger.info(f"Starting REST server on {app.host}:{app.port}")
        logger.info(f"API base path: /{pth}")

    except Exception as e:
        logger.error(f"Server startup error: {e}")
        sys.exit(1)
