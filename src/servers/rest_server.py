#!/usr/bin/env python3
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os
import sys
import logging
import time
import threading
import queue
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

import psutil
import requests
from fastapi import FastAPI, Request, Response, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.core.factory import get_rag_backend
from src.core.interfaces import RAGBackend
from src.core.google_auth import GoogleAuthManager
from src.core.extractors import extract_text_from_bytes

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

from contextlib import asynccontextmanager

# Initialize Backend
backend: RAGBackend = get_rag_backend()
auth_manager = GoogleAuthManager()

# MCP proxy configuration
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = os.getenv("MCP_PORT", "8000")
MCP_PATH = os.getenv("MCP_PATH", "/mcp")

def _mcp_base() -> str:
    """Get the base URL for MCP server."""
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
                return {"status": "ok", "message": resp.text}
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            continue
    
    # If all URLs failed, raise HTTPException
    error_msg = f"MCP proxy failed: {last_exc}" if last_exc else "MCP proxy failed: no response"
    logger.error(error_msg)
    raise HTTPException(status_code=502, detail=error_msg)

# Mutable server state
class ServerState:
    search_worker_started: bool = False
    search_jobs: Dict[str, Dict[str, Any]] = {}
    search_jobs_lock: threading.Lock = threading.Lock()
    search_job_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    
    # Quality metrics
    total_searches: int = 0
    failed_searches: int = 0
    total_sources: int = 0
    responses_with_sources: int = 0
    fallback_responses: int = 0

state = ServerState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("REST server startup complete (host=%s port=%s base=/api)", 
                os.getenv("RAG_HOST", "127.0.0.1"), os.getenv("RAG_PORT", "8001"))
    yield
    logger.info("REST server shutting down")
    try:
        backend.save_store()
    except Exception as e:
        logger.error(f"Error saving store on shutdown: {e}")

app = FastAPI(title="retrieval-rest-server", lifespan=lifespan)

# Allow cross-origin calls from mobile/web clients (permissive by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    model: Optional[str] = None

class RerankReq(BaseModel):
    """Request model for reranking passages."""
    query: str
    passages: List[Dict[str, Any]]
    top_k: Optional[int] = None

class VerifyReq(BaseModel):
    """Request model for grounding verification (citations)."""
    question: str
    draft_answer: str
    citations: List[str]

class VerifySimpleReq(BaseModel):
    """Request model for grounding verification (passages provided)."""
    question: str
    draft_answer: str
    passages: List[Dict[str, Any]]

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

class ConfigModeReq(BaseModel):
    """Request model for setting the backend mode."""
    mode: str

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
    result: Optional[Dict[str, Any]] = None


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

def _start_search_worker():
    if state.search_worker_started:
        return

    def _worker():
        while True:
            job = state.search_job_queue.get()
            if not job:
                continue
            job_id = job["id"]
            try:
                result = backend.search(job["query"])
                status = "completed"
                error = None
            except Exception as exc:
                result = None
                status = "failed"
                error = str(exc)
            with state.search_jobs_lock:
                state.search_jobs[job_id].update({"status": status, "result": result, "error": error})
            state.search_job_queue.task_done()

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    state.search_worker_started = True

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

def _record_quality_metrics(result: Optional[Dict[str, Any]], error: Optional[Exception] = None) -> None:
    """Update quality counters based on a search result or error."""
    state.total_searches += 1

    if error is not None:
        state.failed_searches += 1
        return

    if not isinstance(result, dict):
        state.failed_searches += 1
        return

    if "error" in result:
        state.failed_searches += 1
        return

    sources = result.get("sources") or []
    if isinstance(sources, list):
        state.total_sources += len(sources)
        if len(sources) > 0:
            state.responses_with_sources += 1

    if result.get("warning"):
        state.fallback_responses += 1


def refresh_prometheus_metrics():
    """Refresh gauges reflecting proxy state and process usage."""
    try:
        REST_MEMORY_USAGE_MB.set(_get_memory_usage_mb())
        REST_MEMORY_LIMIT_MB.set(REST_MAX_MEMORY_MB)
    except Exception as exc:
        logger.debug("refresh metrics: memory gauge failed: %s", exc)

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next: Callable):
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
        # Proxy to MCP worker which creates jobs for progress tracking
        return _proxy_to_mcp("POST", "/rest/upsert_document", {"uri": req.uri, "text": req.text, "binary_base64": req.binary_base64})
    except Exception as e:
        logger.error(f"Error upserting document {req.uri}: {e}")
        raise

@app.post(f"/{pth}/index_path")
def api_index_path(req: IndexPathReq):
    """Index a filesystem path into the retrieval store."""
    logger.info(f"Indexing path: path={req.path}, glob={req.glob}")
    try:
        return backend.index_path(req.path, req.glob or "**/*")
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
        with state.search_jobs_lock:
            state.search_jobs[job_id] = job
        state.search_job_queue.put(job)
        return {"job_id": job_id, "status": "queued"}

    try:
        result = backend.search(req.query)
        _record_quality_metrics(result)
        # Ensure result is JSON-serializable (already normalized by _normalize_llm_response)
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
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
        backend.load_store()
        return {"status": "loaded"}
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
        # Pass model if supported by backend
        kwargs = {"k": req.k or 3}
        if req.model and hasattr(backend, "grounded_answer_manual"):
             # Only manual mode supports dynamic model switching for now
             kwargs["model"] = req.model
             
        answer = backend.grounded_answer(req.question, **kwargs)
        return answer
    except Exception as exc:
        logger.error("grounded_answer failed: %s", exc)
        raise

@app.post(f"/{pth}/rerank")
def api_rerank(req: RerankReq):
    """Rerank provided passages for a query."""
    logger.info("Rerank requested for query: %s", req.query)
    try:
        ranked = backend.rerank(req.query, req.passages)
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
        return backend.verify_grounding(req.question, req.draft_answer, req.citations)
    except Exception as exc:
        logger.error("verify_grounding failed: %s", exc)
        raise

@app.post(f"/{pth}/verify_grounding_simple")
def api_verify_simple(req: VerifySimpleReq):
    """Simple grounding verification against provided passages."""
    logger.info("Verify grounding (simple) requested")
    try:
        # verify_grounding_simple is not in RAGBackend yet, but it's a utility.
        # It doesn't need the store, just the passages.
        # We can import it directly or add it to backend.
        # For now, let's import it directly as it's stateless.
        from src.core.rag_core import verify_grounding_simple
        return verify_grounding_simple(req.question, req.draft_answer, req.passages)
    except Exception as exc:
        logger.error("verify_grounding_simple failed: %s", exc)
        raise

@app.get(f"/{pth}/health", response_model=HealthResp)
def api_health():
    """Lightweight health check with basic store and index stats."""
    try:
        data = backend.get_stats()
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
        raise HTTPException(status_code=502, detail={"error": "Backend unreachable", "detail": str(exc)})

@app.get(f"/{pth}/documents", response_model=DocumentsResp)
def api_documents():
    """List indexed document URIs and their approximate text size (bytes)."""
    try:
        uris = backend.list_documents()
        # RAGBackend.list_documents returns list[str], but DocumentsResp expects list[DocumentInfo]
        # We need to fetch size info. LocalBackend.list_documents only returns keys.
        # We might need to update list_documents to return more info or fetch it here.
        # For now, return 0 size.
        return {"documents": [{"uri": uri, "size_bytes": 0} for uri in uris]}
    except HTTPException as exc:
        logger.error("documents list failed: %s", exc.detail)
        raise
    except Exception as exc:
        logger.error("documents list failed: %s", exc)
        raise HTTPException(status_code=502, detail={"error": "Backend unreachable", "detail": str(exc)})

@app.post(f"/{pth}/documents/delete")
def api_documents_delete(req: DeleteDocsReq):
    """Delete documents by URI from the store and rebuild index."""
    try:
        return backend.delete_documents(req.uris)
    except Exception as exc:
        logger.error("documents delete failed: %s", exc)
        return {"deleted": 0, "error": str(exc)}

@app.get(f"/{pth}/config/mode")
def api_get_mode():
    """Get the current backend mode."""
    if hasattr(backend, "get_mode"):
        return {
            "mode": backend.get_mode(),
            "available_modes": backend.get_available_modes()
        }
    return {"mode": "unknown", "available_modes": []}

@app.post(f"/{pth}/config/mode")
def api_set_mode(req: ConfigModeReq):
    """Set the backend mode (e.g., 'local', 'google')."""
    if hasattr(backend, "set_mode"):
        success = backend.set_mode(req.mode)
        if success:
            return {"status": "ok", "mode": req.mode}
        else:
            raise HTTPException(status_code=400, detail=f"Mode '{req.mode}' not available. Available: {backend.get_available_modes()}")
    raise HTTPException(status_code=501, detail="Backend does not support mode switching")

@app.get(f"/{pth}/config/models")
def api_list_models():
    """List available models for the current backend."""
    if hasattr(backend, "list_models"):
        return {"models": backend.list_models()}
    return {"models": []}

@app.get(f"/{pth}/jobs", response_model=dict)
def api_jobs():
    """List async indexing jobs (pass-through to MCP)."""
    # Proxy to MCP server for jobs
    try:
        return _proxy_to_mcp("GET", "/rest/jobs")
    except HTTPException:
        # If MCP is not available, return local search jobs as fallback
        jobs = list(state.search_jobs.values())
        return {"jobs": jobs}

@app.get(f"/{pth}/jobs/{{job_id}}", response_model=dict)
def api_job(job_id: str):
    """Get a single async indexing job (pass-through to MCP)."""
    # See api_jobs comment.
    return {"error": "Not implemented for generic jobs, use /search/jobs/{job_id}"}

@app.get(f"/{pth}/search/jobs/{{job_id}}", response_model=dict)
def api_search_job(job_id: str):
    """Get status/result for an async search job."""
    with state.search_jobs_lock:
        job = state.search_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=44, detail={"error": "not found", "id": job_id})
    return job

@app.post(f"/{pth}/flush_cache")
def api_flush_cache():
    """Flush the document store and delete the backing DB file."""
    try:
        return backend.flush_cache()
    except Exception as exc:
        logger.error("flush_cache failed: %s", exc)
        return Response(status_code=500, content=f"flush failed: {exc}")

@app.get(f"/{pth}/metrics/quality", response_model=QualityMetricsResp)
def api_quality_metrics():
    """Return aggregated quality metrics for searches."""
    total = state.total_searches
    failed = state.failed_searches
    with_sources = state.responses_with_sources
    total_sources = state.total_sources
    fallback = state.fallback_responses
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

@app.get(f"/{pth}/auth/login")
def auth_login(request: Request):
    """Initiate Google OAuth2 flow."""
    # Construct callback URL based on the request's host
    # This requires the Google Cloud Console to have this exact URI whitelisted
    redirect_uri = str(request.url_for('auth_callback'))
    
    # Force localhost if 127.0.0.1 is used, as Google often requires localhost
    if "127.0.0.1" in redirect_uri:
        redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")
    
    # If running behind a proxy (like ngrok or in some container setups), 
    # you might need to force https or a specific host.
    # For now, we trust the request headers.
    
    logger.info(f"Initiating auth flow with redirect_uri: {redirect_uri}")
    
    try:
        flow = auth_manager.flow_from_client_secrets(redirect_uri=redirect_uri)
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        logger.info(f"Generated auth URL: {authorization_url}")
        response = RedirectResponse(authorization_url)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        logger.error(f"Error initiating auth: {e}")
        # If client_secrets.json is missing, redirect to setup page
        if "Client secrets file not found" in str(e) or "No such file" in str(e):
             return RedirectResponse(url=request.url_for('auth_setup'))
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{pth}/auth/setup", name="auth_setup")
def auth_setup(request: Request):
    """Show the setup page for Google Auth."""
    return HTMLResponse(content="""
        <html>
            <head>
                <title>Setup Google Integration</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 40px; max-width: 600px; margin: 0 auto; background-color: #f5f5f7; }
                    .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                    h1 { margin-top: 0; color: #1d1d1f; }
                    p { color: #86868b; line-height: 1.5; }
                    label { display: block; margin-top: 20px; font-weight: 500; color: #1d1d1f; }
                    input[type="text"] { width: 100%; padding: 12px; margin-top: 8px; border: 1px solid #d2d2d7; border-radius: 8px; font-size: 16px; box-sizing: border-box; }
                    button { background-color: #0071e3; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; font-weight: 600; margin-top: 30px; cursor: pointer; width: 100%; }
                    button:hover { background-color: #0077ed; }
                    .details { margin-top: 20px; font-size: 14px; }
                    summary { cursor: pointer; color: #0071e3; }
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>Connect Google Drive</h1>
                    <p>To enable "Premium" features like Drive search and Gemini integration, we need to connect to your Google Cloud project.</p>
                    
                    <form action="./setup" method="post">
                        <label for="client_id">Client ID</label>
                        <input type="text" id="client_id" name="client_id" placeholder="e.g., 12345...apps.googleusercontent.com" required>
                        
                        <label for="client_secret">Client Secret</label>
                        <input type="text" id="client_secret" name="client_secret" placeholder="e.g., GOCSPX-..." required>
                        
                        <button type="submit">Save & Connect</button>
                    </form>
                    
                    <div class="details">
                        <details>
                            <summary>How do I get these?</summary>
                            <ol>
                                <li>Go to the <a href="https://console.cloud.google.com/apis/credentials" target="_blank">Google Cloud Console</a>.</li>
                                <li>Create a new Project (or select existing).</li>
                                <li>Go to <strong>APIs & Services > Credentials</strong>.</li>
                                <li>Click <strong>Create Credentials > OAuth client ID</strong>.</li>
                                <li>Select <strong>Web application</strong>.</li>
                                <li>Add this Redirect URI: <br><code>http://localhost:8001/api/auth/callback</code></li>
                                <li>Click Create and copy the ID and Secret.</li>
                            </ol>
                        </details>
                    </div>
                </div>
            </body>
        </html>
    """)

@app.post(f"/{pth}/auth/setup")
def auth_setup_post(request: Request, client_id: str = Form(...), client_secret: str = Form(...)):
    """Save the provided credentials to client_secrets.json."""
    import json
    
    # Basic validation
    if not client_id or not client_secret:
        return HTMLResponse("Missing fields", status_code=400)
        
    # Construct the JSON structure expected by google-auth-oauthlib
    secrets = {
        "web": {
            "client_id": client_id.strip(),
            "client_secret": client_secret.strip(),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [
                str(request.url_for('auth_callback')),
                "http://localhost:8001/api/auth/callback"
            ]
        }
    }
    
    try:
        with open("client_secrets.json", "w") as f:
            json.dump(secrets, f, indent=2)
            
        # Redirect back to login to start the flow
        return RedirectResponse(url=request.url_for('auth_login'), status_code=303)
    except Exception as e:
        logger.error(f"Failed to save secrets: {e}")
        return HTMLResponse(f"Failed to save configuration: {e}", status_code=500)

@app.get(f"/{pth}/auth/callback", name="auth_callback")
def auth_callback(request: Request, code: str, state: Optional[str] = None):
    """Handle Google OAuth2 callback."""
    redirect_uri = str(request.url_for('auth_callback'))
    logger.info(f"Handling auth callback with redirect_uri: {redirect_uri}")
    
    try:
        flow = auth_manager.flow_from_client_secrets(redirect_uri=redirect_uri)
        flow.fetch_token(code=code)
        creds = flow.credentials
        auth_manager.save_credentials(creds)
        
        # Reload the backend's auth if necessary
        if hasattr(backend, 'reload_auth'):
            backend.reload_auth()
        
        return HTMLResponse(content="""
            <html>
                <head><title>Auth Success</title
                    <h1 style="color: green;">Authentication Successful</h1>
                    <p>You have successfully connected your Google account.</p>
                    <p>You can close this window and return to the app.</p>
                </body>
            </html>
        """)
    except Exception as e:
        logger.error(f"Error in auth callback: {e}")
        return HTMLResponse(content=f"""
            <html>
                <head><title>Auth Failed</title></head>
                <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                    <h1 style="color: red;">Authentication Failed</h1>
                    <p>{e}</p>
                </body>
            </html>
        """, status_code=500)

@app.get(f"/{pth}/auth/logout")
def auth_logout(request: Request):
    """Log out by deleting the stored token."""
    try:
        logger.info(f"Logout requested. Checking for token file...")
        auth_manager.logout()
        if hasattr(backend, 'logout'):
            backend.logout()
        logger.info("Logout successful")
        return JSONResponse({"status": "logged_out"})
    except Exception as e:
        logger.error(f"Error logging out: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None

@app.post(f"/{pth}/chat")
def api_chat(req: ChatReq):
    """Conversational chat."""
    logger.info("Chat request received")
    if hasattr(backend, "chat"):
        # Check if chat accepts model arg
        import inspect
        sig = inspect.signature(backend.chat)
        if "model" in sig.parameters:
            return backend.chat([m.model_dump() for m in req.messages], model=req.model)
        return backend.chat([m.model_dump() for m in req.messages])
    raise HTTPException(status_code=501, detail="Backend does not support chat")

@app.get(f"/{pth}/drive/files")
def api_drive_files(folder_id: Optional[str] = None):
    """List Drive files."""
    if hasattr(backend, "list_drive_files"):
        return {"files": backend.list_drive_files(folder_id)}
    # If backend doesn't support drive files (e.g. Local), return empty or error?
    # For now, return empty list to avoid breaking UI
    return {"files": []}

@app.post(f"/{pth}/drive/upload")
async def api_drive_upload(file: UploadFile = File(...), folder_id: Optional[str] = Form(None)):
    """Upload file to Drive."""
    if hasattr(backend, "upload_file"):
        content = await file.read()
        # Check if upload_file accepts folder_id parameter
        import inspect
        sig = inspect.signature(backend.upload_file)
        if "folder_id" in sig.parameters:
            result = backend.upload_file(file.filename, content, file.content_type, folder_id)
        else:
            result = backend.upload_file(file.filename, content, file.content_type)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    raise HTTPException(status_code=501, detail="Backend does not support drive upload")

@app.delete(f"/{pth}/drive/files/{{file_id}}")
def api_drive_delete(file_id: str):
    """Delete a file or folder from Google Drive."""
    if hasattr(backend, "delete_drive_file"):
        result = backend.delete_drive_file(file_id)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    raise HTTPException(status_code=501, detail="Backend does not support drive delete")

@app.post(f"/{pth}/drive/folders")
def api_drive_create_folder(name: str = Form(...), parent_id: Optional[str] = Form(None)):
    """Create a folder in Google Drive."""
    if hasattr(backend, "create_drive_folder"):
        result = backend.create_drive_folder(name, parent_id)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    raise HTTPException(status_code=501, detail="Backend does not support folder creation")

@app.post(f"/{pth}/extract")
async def api_extract(file: UploadFile = File(...)):
    """Extract text from an uploaded file (in-memory, no storage)."""
    try:
        content = await file.read()
        text = extract_text_from_bytes(content, file.filename)
        return {"text": text, "filename": file.filename}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class VertexConfigReq(BaseModel):
    project_id: str
    location: str
    data_store_id: str

@app.post(f"/{pth}/config/vertex")
def api_save_vertex_config(req: VertexConfigReq):
    """Save Vertex AI configuration."""
    config = {
        "VERTEX_PROJECT_ID": req.project_id,
        "VERTEX_LOCATION": req.location,
        "VERTEX_DATA_STORE_ID": req.data_store_id
    }
    # Save to file
    try:
        with open("vertex_config.json", "w") as f:
            import json
            json.dump(config, f, indent=2)
        
        # Update current process env vars so it works immediately
        os.environ.update(config)
        
        return {"status": "saved"}
    except Exception as e:
        logger.error(f"Failed to save vertex config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{pth}/config/vertex")
def api_get_vertex_config():
    """Get Vertex AI configuration."""
    try:
        if os.path.exists("vertex_config.json"):
            with open("vertex_config.json", "r") as f:
                import json
                return json.load(f)
        return {
            "VERTEX_PROJECT_ID": os.getenv("VERTEX_PROJECT_ID", ""),
            "VERTEX_LOCATION": os.getenv("VERTEX_LOCATION", "us-central1"),
            "VERTEX_DATA_STORE_ID": os.getenv("VERTEX_DATA_STORE_ID", "")
        }
    except Exception as e:
        logger.error(f"Failed to read vertex config: {e}")
        return {}

@app.get(f"/{pth}/logs/{{log_type}}")
def api_get_logs(log_type: str, lines: int = 500):
    """
    Get log file contents.
    
    Args:
        log_type: Type of log (rest, rest_access, mcp, mcp_access, ollama)
        lines: Number of lines to return from the end of the file (default: 500)
    
    Returns:
        JSON with log lines, total count, and file path
    """
    log_files = {
        'rest': LOG_DIR / 'rest_server.log',
        'rest_access': LOG_DIR / 'rest_server_access.log',
        'mcp': LOG_DIR / 'mcp_server.log',
        'mcp_access': LOG_DIR / 'mcp_server_access.log',
        'ollama': LOG_DIR / 'ollama_server.log',
    }
    
    if log_type not in log_files:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid log type. Available: {', '.join(log_files.keys())}")
    
    log_path = log_files[log_type]
    if not log_path.exists():
        return {"lines": [], "total": 0, "file": str(log_path)}
    
    try:
        # Read file efficiently, getting last N lines
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read all lines for small files, or use efficient tail for large files
            all_lines = f.readlines()
        
        total = len(all_lines)
        # Get last N lines
        start_idx = max(0, total - lines)
        selected_lines = all_lines[start_idx:] if total > 0 else []
        
        return {
            "lines": [line.rstrip('\n\r') for line in selected_lines],
            "total": total,
            "file": str(log_path),
            "start": start_idx,
            "end": total
        }
    except Exception as e:
        logger.error("Failed to read log file %s: %s", log_type, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"/{pth}/logs/{{log_type}}/stream")
async def api_stream_logs(log_type: str, request: Request):
    """
    Stream log file contents using Server-Sent Events (SSE).
    
    Args:
        log_type: Type of log (rest, rest_access, mcp, mcp_access, ollama)
        request: FastAPI request object for client disconnect detection
    
    Returns:
        StreamingResponse with SSE format
    """
    log_files = {
        'rest': LOG_DIR / 'rest_server.log',
        'rest_access': LOG_DIR / 'rest_server_access.log',
        'mcp': LOG_DIR / 'mcp_server.log',
        'mcp_access': LOG_DIR / 'mcp_server_access.log',
        'ollama': LOG_DIR / 'ollama_server.log',
    }
    
    if log_type not in log_files:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid log type. Available: {', '.join(log_files.keys())}")
    
    log_path = log_files[log_type]
    
    async def generate_log_stream():
        """Generate SSE stream of log lines."""
        last_position = 0
        if log_path.exists():
            # Get initial file size
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # Seek to end
                last_position = f.tell()
        
        # Send initial data
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(max(0, last_position - 10000))  # Read last 10KB initially
                    initial_lines = f.readlines()[-100:]  # Last 100 lines
                    for line in initial_lines:
                        if await request.is_disconnected():
                            break
                        yield f"data: {line.rstrip()}\n\n"
            except Exception as e:
                logger.error("Error reading initial log lines: %s", e)
                yield f"data: [ERROR] Failed to read log file: {e}\n\n"
        
        # Stream new lines
        while not await request.is_disconnected():
            try:
                if log_path.exists():
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        current_size = f.seek(0, 2)  # Get file size
                        
                        if current_size > last_position:
                            # New content available
                            f.seek(last_position)
                            new_lines = f.readlines()
                            for line in new_lines:
                                if await request.is_disconnected():
                                    break
                                yield f"data: {line.rstrip()}\n\n"
                            last_position = current_size
                        elif current_size < last_position:
                            # File was rotated or truncated
                            last_position = 0
                            f.seek(0)
                            recent_lines = f.readlines()[-50:]  # Last 50 lines
                            for line in recent_lines:
                                if await request.is_disconnected():
                                    break
                                yield f"data: {line.rstrip()}\n\n"
                            last_position = f.tell()
                
                # Wait before checking again
                await asyncio.sleep(1)
            except Exception as e:
                logger.error("Error streaming logs: %s", e)
                yield f"data: [ERROR] {e}\n\n"
                await asyncio.sleep(2)
    
    return StreamingResponse(
        generate_log_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

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
