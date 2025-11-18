#!/usr/bin/env python3
"""
MCP Retrieval Server — FastMCP Streamable HTTP

Environment Variables:
    MCP_PATH: HTTP path for FastMCP (default: /mcp)
    MCP_HOST: Address to listen     (default: 127.0.0.1)
    MCP_PORT: TCP port to listen    (default: 8000)
    MAX_MEMORY_MB: Memory limit in MB (default: 75% of system memory)

Run (dev):
    uv run python src/servers/mcp_server.py
    # or
    python src/servers/mcp_server.py

Connect URL (client side):
    http://127.0.0.1:8000/mcp
"""

from __future__ import annotations
import atexit
import gc
import os
from pathlib import Path
import base64
import tempfile
import resource
import signal
import sys
import threading
import time
import logging
import multiprocessing
import queue
from typing import Dict, Any, Optional, List
import re
import uuid

import psutil
import json
import requests
from mcp.server.fastmcp import FastMCP
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount
from starlette.applications import Starlette
from fastapi import FastAPI, Request
import anyio
from dotenv import load_dotenv

# Import shared logic
from src.core.rag_core import (
    search,
    load_store,
    save_store,
    upsert_document,
    get_store,
    resolve_input_path,
    _extract_text_from_file,
    rerank,
    grounded_answer,
    verify_grounding,
    get_faiss_globals,
    OLLAMA_API_BASE,
    _rebuild_faiss_index,
    DB_PATH,
)

# Set up logging
os.makedirs('log', exist_ok=True)

# Process logger (application logs)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/mcp_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Access logger (HTTP access logs)
access_logger = logging.getLogger('mcp_access')
access_logger.setLevel(logging.INFO)
access_logger.handlers.clear()
access_logger.propagate = False
_access_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_access_handlers = [
    logging.FileHandler('log/mcp_access.log'),
]
for _h in _access_handlers:
    _h.setFormatter(_access_formatter)
    access_logger.addHandler(_h)

# Disable uvicorn's default access logger to keep access lines out of mcp_server.log
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers.clear()
uvicorn_access_logger.propagate = False
uvicorn_access_logger.disabled = True

# Force uvicorn to use a log config without the access logger handlers (uvicorn resets
# logging with its LOGGING_CONFIG, so we patch that too).
try:
    import uvicorn
    from copy import deepcopy

    patched = deepcopy(getattr(uvicorn.config, "LOGGING_CONFIG", {}))
    if patched and "loggers" in patched and "uvicorn.access" in patched["loggers"]:
        patched["loggers"]["uvicorn.access"]["handlers"] = []
        patched["loggers"]["uvicorn.access"]["propagate"] = False
        patched["loggers"]["uvicorn.access"]["level"] = "WARNING"
        uvicorn.config.LOGGING_CONFIG = patched
    # Also set env flag that uvicorn respects
    os.environ.setdefault("UVICORN_ACCESS_LOG", "false")
except Exception as exc:  # pragma: no cover
    logger.debug("Unable to patch uvicorn access logging: %s", exc)

# Monkey-patch FastMCP's streamable HTTP runner to disable uvicorn access logging.
async def _run_streamable_http_no_access(self):
    import uvicorn

    starlette_app = self.streamable_http_app()
    config = uvicorn.Config(
        starlette_app,
        host=self.settings.host,
        port=self.settings.port,
        log_level=self.settings.log_level.lower(),
        access_log=False,  # critical: disable uvicorn access logger
        log_config=getattr(uvicorn.config, "LOGGING_CONFIG", None),
    )
    server = uvicorn.Server(config)
    await server.serve()

# Bind monkey patch so FastMCP uses access-log-disabled server startup
from mcp.server.fastmcp import FastMCP as _FastMCP
_FastMCP.run_streamable_http_async = _run_streamable_http_no_access

# Replace inner Starlette app with one that mounts REST routes at /rest and prefixed path
# Access logging middleware for MCP/REST Starlette apps
class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        access_logger.info(
            "%s %s %s %s %dms",
            request.method,
            request.url.path,
            request.client.host if request.client else "-",
            response.status_code,
            duration_ms,
        )
        return response

_orig_streamable_http_app = _FastMCP.streamable_http_app
def _streamable_http_app_with_rest(self):
    """Wrap FastMCP's app to expose REST-style routes."""
    base_app = _orig_streamable_http_app(self)
    try:
        base_app.add_middleware(AccessLogMiddleware)
    except Exception:
        pass
    prefix = getattr(self.settings, "streamable_http_path", "/")
    routes = [Mount("/rest", rest_api)]
    if prefix and prefix != "/":
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        routes.append(Mount(f"{prefix}/rest", rest_api))
    routes.append(Mount("/", base_app))
    combined = Starlette(routes=routes)
    return combined

_FastMCP.streamable_http_app = _streamable_http_app_with_rest

# Load environment variables from .env if present before evaluating settings
load_dotenv()

# Create FastMCP instance
mcp = FastMCP("retrieval-server")

# Track if we're already shutting down to prevent duplicate saves
SHUTTING_DOWN = False
STORE_LOADING = False
STORE_LOADED = False
STORE_LOAD_THREAD: Optional[threading.Thread] = None
INDEX_JOB_QUEUE: Optional[multiprocessing.Queue] = None
INDEX_RESULT_QUEUE: Optional[multiprocessing.Queue] = None
INDEX_JOBS: Dict[str, Dict[str, Any]] = {}
INDEX_JOBS_LOCK = threading.Lock()
INDEX_RESULT_THREAD: Optional[threading.Thread] = None

# Memory management settings
MEMORY_CHECK_INTERVAL = 30  # seconds
MEMORY_LOG_STEP_MB = 256  # log memory usage whenever it crosses another 256MB bucket
# macOS has a global jetsam killer that will SIGKILL processes under memory pressure
# before Python can log anything. We still monitor usage so we have a trail when the
# OS lets us run, but jetsam kills will appear as sudden exits with no last log line.
MEMORY_MONITOR_THREAD = None
LAST_MEMORY_LOG_BUCKET = None

# Prometheus metrics
METRICS_REGISTRY = CollectorRegistry()
MCP_DOCUMENTS_INDEXED = Gauge(
    "mcp_documents_indexed_total",
    "Number of documents indexed in the MCP store.",
    registry=METRICS_REGISTRY,
)
MCP_MEMORY_USAGE_MB = Gauge(
    "mcp_memory_usage_megabytes",
    "Current process RSS memory usage in megabytes.",
    registry=METRICS_REGISTRY,
)
MCP_MEMORY_LIMIT_MB = Gauge(
    "mcp_memory_limit_megabytes",
    "Configured memory limit in megabytes.",
    registry=METRICS_REGISTRY,
)
MCP_EMBEDDING_VECTORS = Gauge(
    "mcp_embedding_vectors_total",
    "Total vectors stored in the embedding index.",
    registry=METRICS_REGISTRY,
)
MCP_EMBEDDING_CHUNKS = Gauge(
    "mcp_embedding_chunks_total",
    "Total text chunks tracked for embeddings.",
    registry=METRICS_REGISTRY,
)
MCP_EMBEDDING_DIM = Gauge(
    "mcp_embedding_dimension",
    "Embedding dimension used by the current model.",
    registry=METRICS_REGISTRY,
)
OLLAMA_UP = Gauge(
    "ollama_up",
    "Whether the Ollama API responded successfully.",
    registry=METRICS_REGISTRY,
)
OLLAMA_RUNNING_MODELS = Gauge(
    "ollama_running_models",
    "Count of running models reported by Ollama /api/ps.",
    registry=METRICS_REGISTRY,
)
OLLAMA_AVAILABLE_MODELS = Gauge(
    "ollama_available_models",
    "Count of available models reported by Ollama /api/tags.",
    registry=METRICS_REGISTRY,
)
OLLAMA_RUNNING_MODEL_SIZE_BYTES = Gauge(
    "ollama_running_model_size_bytes",
    "Resident size of running models reported by Ollama.",
    ["model", "digest"],
    registry=METRICS_REGISTRY,
)
OLLAMA_RUNNING_MODEL_VRAM_BYTES = Gauge(
    "ollama_running_model_vram_bytes",
    "VRAM usage of running models reported by Ollama.",
    ["model", "digest"],
    registry=METRICS_REGISTRY,
)
OLLAMA_MODEL_SIZE_BYTES = Gauge(
    "ollama_model_size_bytes",
    "Size of available Ollama models.",
    ["model", "digest"],
    registry=METRICS_REGISTRY,
)

def get_system_memory_mb():
    """Get total system memory in MB."""
    return psutil.virtual_memory().available / 1024 / 1024

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_max_memory_mb():
    """Get the maximum memory limit, defaulting to 75% of system memory."""
    max_memory_env = os.getenv("MAX_MEMORY_MB")
    if max_memory_env:
        return int(max_memory_env)
    else:
        system_memory_mb = get_system_memory_mb()
        return int(system_memory_mb * 0.75)  # 75% of system memory

MAX_MEMORY_MB = get_max_memory_mb()

#
# REST shim (mounts under /rest and <MCP_PATH>/rest)
#
rest_api = FastAPI(title="mcp-rest-shim")

rest_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rest_api.add_middleware(AccessLogMiddleware)


@rest_api.post("/upsert_document")
async def rest_upsert_document(request: Request):
    body = await request.json()
    uri = body.get("uri", "")
    text = body.get("text")
    binary_b64 = body.get("binary_base64")
    if not uri:
        return JSONResponse({"error": "uri is required"}, status_code=422)
    if text is None and binary_b64 is None:
        return JSONResponse({"error": "either text or binary_base64 is required"}, status_code=422)
    try:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(DB_PATH).touch(exist_ok=True)
    except Exception:
        pass
    try:
        job_payload = {"uri": uri, "text": text, "binary_base64": binary_b64}
        job_id = _enqueue_index_job("upsert_document", job_payload)
        return JSONResponse({"job_id": job_id, "status": "queued"})
    except Exception as exc:
        logger.error("rest_upsert_document failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/index_path")
async def rest_index_path(request: Request):
    body = await request.json()
    path = body.get("path", "")
    glob = body.get("glob", "**/*.txt")
    try:
        job_id = _enqueue_index_job("index_path", {"path": path, "glob": glob or "**/*.txt"})
        return JSONResponse({"job_id": job_id, "status": "queued"})
    except Exception as exc:
        logger.error("rest_index_path failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)
@rest_api.post("/search")
async def rest_search(request: Request):
    body = await request.json()
    query = body.get("query", "")
    try:
        # Offload embedding + FAISS lookup to a worker thread to keep the event loop responsive
        result = await anyio.to_thread.run_sync(search, query)
        # Normalize litellm ModelResponse to a simple JSON shape the UI expects
        normalized: Dict[str, Any] = {}
        if isinstance(result, dict):
            normalized = result
        elif hasattr(result, "choices"):
            try:
                choices = getattr(result, "choices") or []
                msg = choices[0].message if choices else None  # type: ignore[attr-defined]
                content = getattr(msg, "content", None) if msg else None
                if content:
                    normalized = {"answer": content, "sources": []}
            except Exception:
                normalized = {}
        if not normalized:
            # Fallback to string form
            normalized = {"answer": str(result)}
        # Ensure JSON-serializable
        normalized = json.loads(json.dumps(normalized, default=str))
        return JSONResponse(normalized)
    except Exception as exc:
        logger.error("rest_search failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/documents")
async def rest_documents(_request: Request):
    try:
        store = get_store()
        docs = []
        for uri, text in getattr(store, "docs", {}).items():
            try:
                size = len(str(text).encode("utf-8", errors="ignore"))
            except Exception:
                size = 0
            docs.append({"uri": uri, "size_bytes": size})
        return JSONResponse({"documents": docs})
    except Exception as exc:
        logger.error("rest_documents failed: %s", exc, exc_info=True)
        return JSONResponse({"documents": []})

@rest_api.get("/jobs/{job_id}")
async def rest_job_status(job_id: str):
    with INDEX_JOBS_LOCK:
        job = INDEX_JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(job)

@rest_api.get("/jobs")
async def rest_jobs():
    with INDEX_JOBS_LOCK:
        jobs = list(INDEX_JOBS.values())
    return JSONResponse({"jobs": jobs})


@rest_api.post("/documents/delete")
async def rest_documents_delete(request: Request):
    try:
        body = await request.json()
        uris = body.get("uris", [])
        store = get_store()
        deleted = 0
        for uri in uris:
            if uri in store.docs:
                del store.docs[uri]
                deleted += 1
        save_store()
        _rebuild_faiss_index()
        refresh_prometheus_metrics()
        return JSONResponse({"deleted": deleted})
    except Exception as exc:
        logger.error("rest_documents_delete failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/flush_cache")
async def rest_flush_cache(_request: Request):
    try:
        store = get_store()
        store.docs.clear()
        removed = False
        if DB_PATH and Path(DB_PATH).exists():
            try:
                Path(DB_PATH).unlink()
                removed = True
            except OSError:
                removed = False
        save_store()
        _rebuild_faiss_index()
        refresh_prometheus_metrics()
        return JSONResponse({"status": "flushed", "db_removed": removed, "documents": len(store.docs)})
    except Exception as exc:
        logger.error("rest_flush_cache failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/health")
async def rest_health(_request: Request):
    try:
        store = get_store()
        index, _, _ = get_faiss_globals()
        docs = len(getattr(store, "docs", {}))
        vectors = index.ntotal if index is not None else 0  # type: ignore[attr-defined]
        return JSONResponse({
            "status": "ok",
            "documents": docs,
            "vectors": vectors,
            "memory_mb": get_memory_usage(),
            "memory_limit_mb": MAX_MEMORY_MB,
            "total_size_bytes": sum(len(str(t).encode("utf-8", errors="ignore")) for t in getattr(store, "docs", {}).values())
        })
    except Exception as exc:
        logger.error("rest_health failed: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


# -------------------------
# Indexing worker (separate process)
# -------------------------
def _index_worker_loop(job_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, env: Dict[str, str]):
    """Run indexing jobs in an isolated process to keep the main server responsive."""
    for k, v in env.items():
        os.environ[k] = v
    from dotenv import load_dotenv as _ld
    _ld()
    from src.core.rag_core import (
        index_path as _worker_index_path,
        upsert_document as _worker_upsert,
        load_store as _worker_load_store,
        save_store as _worker_save_store,
        _rebuild_faiss_index as _worker_rebuild,
    )
    while True:
        job = job_queue.get()
        if not job:
            continue
        if job.get("type") == "stop":
            break
        job_id = job.get("id")
        try:
            _worker_load_store()
            if job["type"] == "index_path":
                res = _worker_index_path(job["path"], job.get("glob", "**/*.txt"))
            elif job["type"] == "upsert_document":
                uri = job["uri"]
                text = job.get("text") or ""
                binary_b64 = job.get("binary_base64")
                uri_suffix = Path(uri).suffix.lower()
                binary_exts = {".pdf", ".doc", ".docx", ".pages"}
                if binary_b64:
                    try:
                        data = base64.b64decode(binary_b64)
                        suffix = uri_suffix or ".tmp"
                        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                            tmp.write(data)
                            tmp.flush()
                            extracted = _extract_text_from_file(Path(tmp.name))
                            text = extracted or text
                    except Exception as exc:
                        logger.error("Failed to extract text from binary upload %s: %s", uri, exc)
                if not text:
                    # For known binary doc types, fail fast instead of indexing binary noise
                    if binary_b64 and uri_suffix in binary_exts:
                        result_queue.put({"id": job_id, "status": "failed", "error": "no text extracted", "uri": uri})
                        continue
                    # Last resort: try to decode as utf-8 text for unknown binaries
                    try:
                        if binary_b64:
                            raw = base64.b64decode(binary_b64)
                            text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                if not text:
                    result_queue.put({"id": job_id, "status": "failed", "error": "empty or non-text content", "uri": uri})
                    continue
                res = _worker_upsert(uri, text)
            else:
                res = {"error": f"unknown job type {job['type']}"}
            _worker_rebuild()
            _worker_save_store()
            result_queue.put({"id": job_id, "status": "completed", "result": res})
        except Exception as exc:
            result_queue.put({"id": job_id, "status": "failed", "error": str(exc)})


def _start_index_worker_if_needed():
    global INDEX_JOB_QUEUE, INDEX_RESULT_QUEUE, INDEX_RESULT_THREAD
    if INDEX_JOB_QUEUE is not None:
        return
    INDEX_JOB_QUEUE = multiprocessing.Queue()
    INDEX_RESULT_QUEUE = multiprocessing.Queue()
    env_copy = {k: v for k, v in os.environ.items() if k.startswith(("RAG_", "EMBED_", "LLM_", "MCP_", "OLLAMA_"))}
    proc = multiprocessing.Process(
        target=_index_worker_loop,
        args=(INDEX_JOB_QUEUE, INDEX_RESULT_QUEUE, env_copy),
        daemon=True,
    )
    proc.start()

    def _result_listener():
        while True:
            try:
                msg = INDEX_RESULT_QUEUE.get()
            except (EOFError, OSError):
                break
            if not msg:
                continue
            job_id = msg.get("id")
            with INDEX_JOBS_LOCK:
                job = INDEX_JOBS.get(job_id, {"id": job_id})
                job.update(msg)
                INDEX_JOBS[job_id] = job
            try:
                load_store()
                _rebuild_faiss_index()
                refresh_prometheus_metrics()
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to refresh store after job %s: %s", job_id, exc, exc_info=True)
            # Ensure store persists even if background load had emptied it
            try:
                save_store()
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to save store after job %s: %s", job_id, exc, exc_info=True)

    INDEX_RESULT_THREAD = threading.Thread(target=_result_listener, daemon=True)
    INDEX_RESULT_THREAD.start()


def _enqueue_index_job(job_type: str, payload: Dict[str, Any]) -> str:
    _start_index_worker_if_needed()
    job_id = str(uuid.uuid4())
    job = {"id": job_id, "type": job_type, **payload, "status": "queued"}
    with INDEX_JOBS_LOCK:
        INDEX_JOBS[job_id] = job
    INDEX_JOB_QUEUE.put(job)
    return job_id


def _background_load_store():
    global STORE_LOADING, STORE_LOADED
    if STORE_LOADING or STORE_LOADED:
        return
    STORE_LOADING = True
    try:
        load_store()
        refresh_prometheus_metrics()
        STORE_LOADED = True
        logger.info("Background store load complete")
    except Exception as exc:
        logger.error("Background store load failed: %s", exc, exc_info=True)
    finally:
        STORE_LOADING = False


def start_background_store_load():
    """Kick off store load/rebuild in a background thread so startup is fast."""
    global STORE_LOAD_THREAD
    if STORE_LOAD_THREAD and STORE_LOAD_THREAD.is_alive():
        return
    STORE_LOAD_THREAD = threading.Thread(target=_background_load_store, daemon=True)
    STORE_LOAD_THREAD.start()

def _safe_int(value: Any) -> Optional[int]:
    """Convert a value to int when possible, otherwise None."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

def _normalize_ollama_base() -> str:
    """Return a normalized Ollama API base URL without trailing slash."""
    base = os.getenv("OLLAMA_API_BASE", OLLAMA_API_BASE)
    return base.rstrip("/")

def memory_monitor():
    """Monitor memory usage and trigger cleanup if needed."""
    global LAST_MEMORY_LOG_BUCKET
    consecutive_limit_hits = 0
    while not SHUTTING_DOWN:
        try:
            memory_mb = get_memory_usage()
            current_bucket = int(memory_mb // MEMORY_LOG_STEP_MB)
            if LAST_MEMORY_LOG_BUCKET is None or current_bucket > LAST_MEMORY_LOG_BUCKET:
                LAST_MEMORY_LOG_BUCKET = current_bucket
                logger.info(
                    "Memory usage crossed %dMB: %.1fMB used (limit %dMB)",
                    current_bucket * MEMORY_LOG_STEP_MB, memory_mb, MAX_MEMORY_MB
                )
            if memory_mb > MAX_MEMORY_MB:
                consecutive_limit_hits += 1
                logger.warning(
                    "Memory usage %.1fMB exceeds limit %dMB. Triggering garbage collection (strike %d).",
                    memory_mb, MAX_MEMORY_MB, consecutive_limit_hits
                )
                gc.collect()

                # Check again after GC
                memory_mb = get_memory_usage()
                if memory_mb > MAX_MEMORY_MB * 0.95:  # Still high after GC
                    logger.error("Memory usage still high after GC: %.1fMB.", memory_mb)
                    if consecutive_limit_hits >= 3:
                        logger.critical(
                            "Memory usage %.1fMB stayed above limit for %d checks. "
                            "Initiating graceful shutdown before the OS kills the process.",
                            memory_mb, consecutive_limit_hits
                        )
                        graceful_shutdown()
                        os._exit(1)
                else:
                    logger.info("Memory usage after GC: %.1fMB", memory_mb)
                    consecutive_limit_hits = 0
            else:
                consecutive_limit_hits = 0

            time.sleep(MEMORY_CHECK_INTERVAL)
        except Exception as e:
            logger.error("Error in memory monitor: %s", e)
            time.sleep(MEMORY_CHECK_INTERVAL)

def start_memory_monitor():
    """Start the memory monitoring thread."""
    global MEMORY_MONITOR_THREAD
    if MEMORY_MONITOR_THREAD is None:
        MEMORY_MONITOR_THREAD = threading.Thread(target=memory_monitor, daemon=True)
        MEMORY_MONITOR_THREAD.start()
        logger.info("Started memory monitor. Max memory limit: %dMB", MAX_MEMORY_MB)

def set_memory_limits():
    """Set system memory limits for the process."""
    if sys.platform == "darwin":
        logger.warning(
            "Skipping RLIMIT_AS enforcement on macOS — the kernel sends SIGKILL "
            "when the soft limit is hit, so we rely on the memory monitor instead. "
            "Use a container/cgroup if you need a hard cap."
        )
        return
    try:
        # Set virtual memory limit (soft only). Keep hard limit high so OS signals don't kill us.
        _, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        max_memory_bytes = MAX_MEMORY_MB * 1024 * 1024
        hard_limit = current_hard if current_hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        # Only lower the hard limit if we already have a finite cap; otherwise leave it untouched.
        if hard_limit != resource.RLIM_INFINITY and max_memory_bytes > hard_limit:
            hard_limit = max_memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard_limit))
        logger.info("Set process memory limit to %dMB", MAX_MEMORY_MB)
    except Exception as e:
        logger.warning("Could not set memory limit: %s", e)

def _fetch_ollama_metrics(timeout: float = 2.5) -> Dict[str, Any]:
    """Collect light-weight metrics from the Ollama server."""
    metrics: Dict[str, Any] = {
        "up": False,
        "running_models": [],
        "available_models": [],
    }
    base = _normalize_ollama_base()
    if not base:
        return metrics

    try:
        resp = requests.get(f"{base}/api/ps", timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        metrics["running_models"] = body.get("models", []) or []
        metrics["up"] = True
    except Exception as exc:
        logger.debug("Ollama /api/ps metrics unavailable: %s", exc)

    try:
        resp = requests.get(f"{base}/api/tags", timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        metrics["available_models"] = body.get("models", []) or []
        metrics["up"] = metrics["up"] or True
    except Exception as exc:
        logger.debug("Ollama /api/tags metrics unavailable: %s", exc)

    return metrics

def _set_labeled_gauge(gauge: Gauge, labels: Dict[str, str], value: Optional[int]):
    """Safely set a labeled gauge when a numeric value is available."""
    if value is None:
        return
    gauge.labels(**labels).set(value)

def refresh_prometheus_metrics():
    """Update Prometheus gauges with current server state."""
    try:
        store = get_store()
        MCP_DOCUMENTS_INDEXED.set(len(getattr(store, "docs", {})))
    except Exception as exc:
        logger.debug("Failed to update document metrics: %s", exc)

    try:
        MCP_MEMORY_USAGE_MB.set(get_memory_usage())
        MCP_MEMORY_LIMIT_MB.set(MAX_MEMORY_MB)
    except Exception as exc:
        logger.debug("Failed to update memory metrics: %s", exc)

    try:
        index, index_to_meta, embed_dim = get_faiss_globals()
        total_vectors = index.ntotal if index is not None else 0  # type: ignore[attr-defined]
        MCP_EMBEDDING_VECTORS.set(total_vectors)
        MCP_EMBEDDING_CHUNKS.set(len(index_to_meta) if index_to_meta is not None else 0)
        MCP_EMBEDDING_DIM.set(embed_dim or 0)
    except Exception as exc:
        logger.debug("Failed to update embedding metrics: %s", exc)

    try:
        ollama = _fetch_ollama_metrics()
        OLLAMA_UP.set(1 if ollama.get("up") else 0)

        running = ollama.get("running_models", []) or []
        available = ollama.get("available_models", []) or []

        OLLAMA_RUNNING_MODELS.set(len(running))
        OLLAMA_AVAILABLE_MODELS.set(len(available))

        OLLAMA_RUNNING_MODEL_SIZE_BYTES.clear()
        OLLAMA_RUNNING_MODEL_VRAM_BYTES.clear()
        for model in running:
            model_name = model.get("name") or model.get("model") or "unknown"
            digest = model.get("digest", "")
            labels = {"model": str(model_name), "digest": str(digest)}
            _set_labeled_gauge(OLLAMA_RUNNING_MODEL_SIZE_BYTES, labels, _safe_int(model.get("size")))
            _set_labeled_gauge(OLLAMA_RUNNING_MODEL_VRAM_BYTES, labels, _safe_int(model.get("size_vram")))

        OLLAMA_MODEL_SIZE_BYTES.clear()
        for model in available:
            model_name = model.get("name") or model.get("model") or "unknown"
            digest = model.get("digest", "")
            labels = {"model": str(model_name), "digest": str(digest)}
            _set_labeled_gauge(OLLAMA_MODEL_SIZE_BYTES, labels, _safe_int(model.get("size")))
    except Exception as exc:
        logger.debug("Failed to update Ollama metrics: %s", exc)

def graceful_shutdown(signum: Optional[int] = None, _frame: Any = None):
    """Handle shutdown gracefully with proper cleanup."""
    global SHUTTING_DOWN

    if SHUTTING_DOWN:
        logger.debug("Shutdown already in progress, skipping duplicate")
        return

    SHUTTING_DOWN = True

    if signum:
        logger.info("Received signal %d. Initiating graceful shutdown...", signum)
    else:
        logger.info("Initiating graceful shutdown...")

    try:
        save_store()
        logger.info("Store saved successfully")
    except Exception as e:
        logger.error("Error saving store during shutdown: %s", e, exc_info=True)
    finally:
        logger.info("Shutdown complete")
        if signum:
            sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
signal.signal(signal.SIGTERM, graceful_shutdown)  # kill command

# Register cleanup on normal exit
atexit.register(graceful_shutdown)

@mcp.tool()
def upsert_document_tool(uri: str, text: str) -> Dict[str, Any]:
    """Upsert a SINGLE document into the store.

    DO NOT USE this tool for:
    - Indexing directories (use index_documents_tool instead)
    - Indexing multiple files (use index_documents_tool instead)
    - When user says "index", "add docs", "index directory" (use index_documents_tool)

    USE THIS TOOL ONLY when:
    - Adding/updating ONE specific document with provided text content
    - Adding/updating an Internet URL, e.g., http://example.com/doc.txt
    - The text content is already available (not reading from disk)

    IMPORTANT: The 'uri' must be an exact, literal string and must not be modified.

    Args:
        uri: Document identifier/path (must be a FILE, not a directory)
        text: Document content (required, not empty)

    Returns:
        dict with upserted status and whether document existed
    """
    try:
        logger.info("Upserting document: %s", uri)
        normalized_uri = str(Path(uri))
        content = text or ""
        if not content.strip():
            # Fallback to reading from disk when the caller provided only a file path.
            file_path = Path(uri)
            if file_path.exists():
                if file_path.is_dir():
                    return {"error": f"Cannot upsert directory. Use index_documents_tool for directories: {normalized_uri}", "upserted": False}
                try:
                    content = _extract_text_from_file(file_path)
                    if not content:
                        return {"error": f"No text extracted from {normalized_uri}", "upserted": False}
                    logger.debug("Read %d characters from %s for upsert", len(content), normalized_uri)
                except Exception as file_err:
                    logger.error("Failed to read %s: %s", normalized_uri, file_err)
                    return {"error": f"Could not read file: {file_err}", "upserted": False}
            else:
                return {"error": f"Text missing and file not found: {normalized_uri}", "upserted": False}

        result: Dict[str, Any] = upsert_document(normalized_uri, content)
        logger.info("Successfully upserted document: %s", uri)
        return result
    except Exception as e:
        logger.error("Error upserting document %s: %s", uri, e)
        return {"error": str(e), "upserted": False}

def _normalize_glob(glob: Optional[str]) -> str:
    """Normalize the incoming glob string and handle common regex mistakes."""
    default_glob = "**/*"
    if not glob or not glob.strip():
        return default_glob
    cleaned = glob.strip()
    # Some clients mistakenly send regex '**/.*' meaning 'anything'
    if cleaned in {"**/.*", "./**/.*"}:
        logger.warning("Received regex-style glob '%s'; falling back to '%s'", cleaned, default_glob)
        return default_glob
    return cleaned

_URL_PATTERN = re.compile(r"https?://[^\s\"'>]+")

def _extract_url_from_query(query: Optional[str]) -> Optional[str]:
    """Pull the first URL-looking token from a free-form query string."""
    if not query:
        return None
    match = _URL_PATTERN.search(query)
    if match:
        return match.group(0).strip().rstrip(",.")
    return None

def _extract_path_from_query(query: Optional[str]) -> Optional[str]:
    """
    Pull a potential filesystem path from a free-form string like
    'index ./docs' so we can route to index_documents_tool.
    """
    if not query:
        return None
    text = query.strip().strip("\"'")
    lowered = text.lower()
    for prefix in ("index ", "add ", "ingest ", "load "):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip().strip("\"'")
            break
    if text:
        return text
    return None

@mcp.tool()
def index_documents_tool(path: str, glob: str = "**/*") -> Dict[str, Any]:
    """ADD/INDEX documents from a directory into the searchable store.

    USE THIS TOOL when the user wants to:
    - Add documents to the index
    - Index files or directories
    - Make documents searchable
    - Build/create/populate the document index

    This reads files from disk and makes them searchable via search_tool().
    Supports: .txt, .pdf, .docx, .doc, .html, .htm files

    Args:
        path: Directory to index (e.g., "documents", "./documents")
        glob: File pattern (default: "**/*" for all supported files)

    Returns:
        dict with count of indexed documents and their URIs

    Examples:
        User: "index documents" → index_documents_tool("documents")
        User: "index the documents" → index_documents_tool("documents")
        User: "add documents to index" → index_documents_tool("documents")
    """
    try:
        try:
            base = resolve_input_path(path)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            return {"error": str(exc), "indexed": 0, "uris": []}

        normalized_glob = _normalize_glob(glob)
        logger.info("Indexing directory %s with glob '%s'", base, normalized_glob)
        if base.is_file():
            files = [base]
        else:
            files = list(base.rglob(normalized_glob))
        if not files:
            message = f"No files found in {base} matching {normalized_glob}"
            logger.info(message)
            return {"indexed": 0, "uris": [], "error": message}

        indexed = 0
        indexed_uris = []
        for file_path in files:
            try:
                content = _extract_text_from_file(file_path)
                if not content:
                    logger.warning("No text extracted from %s, skipping", file_path)
                    continue
            except Exception as read_err:
                logger.warning("Skipping %s: %s", file_path, read_err)
                continue

            upsert_document(str(file_path), content)
            indexed += 1
            indexed_uris.append(str(file_path))

        logger.info("Indexed %d documents from %s", indexed, base)
        return {"indexed": indexed, "uris": indexed_uris}
    except Exception as e:
        logger.error("Error indexing path %s: %s", path, e)
        return {"error": str(e), "indexed": 0, "uris": []}

@mcp.tool()
def index_url_tool(
    url: Optional[str] = None,
    doc_id: Optional[str] = None,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    INDEX a document from a URL into the searchable store.

    CRITICAL: ALWAYS use this tool for prompts like:
    - "index http://..." or "index https://..."
    - "add this url to the index"
    - "index a web page"

    DO NOT use index_documents_tool for URLs. ONLY use this tool for remote documents.

    USE THIS TOOL when the user wants to:
    - Index a document from a URL
    - Add a web page to the index
    - Index remote documents (PDF, DOCX, HTML, TXT)

    Supports URLs pointing to:
    - PDF files (http://example.com/doc.pdf)
    - DOCX files (http://example.com/doc.docx)
    - HTML pages (http://example.com/page.html)
    - Plain text files (http://example.com/file.txt)

    Args:
        url: URL of the document to index (e.g., "http://example.com/doc.pdf")
        doc_id: Optional document ID (defaults to the URL)
        query: Optional free-form request. If it contains a URL, it will be extracted.

    Returns:
        dict with status and indexed URI

    Examples:
        User: "index http://example.com/doc.pdf"
        User: "index https://about.me/jhcook"
        User: "add this url to the index: http://..."
    """
    try:
        # Accept legacy callers that accidentally send {"query": "..."}
        if not url:
            url = _extract_url_from_query(query)

        # If there is no URL but we received something that looks like a local path,
        # route the request to the directory indexer so "index ./documents" works.
        if not url:
            possible_path = _extract_path_from_query(query)
            if possible_path and not possible_path.startswith(("http://", "https://")):
                logger.info(
                    "index_url_tool received non-URL input '%s'; routing to index_documents_tool",
                    possible_path
                )
                return index_documents_tool(possible_path)

        if not url:
            return {"error": "URL missing. Provide a url argument or a valid query.", "indexed": 0}

        import pathlib
        # Treat URL as a pathlib.Path for _extract_text_from_file
        url_path = pathlib.Path(url)
        content = _extract_text_from_file(url_path)

        if not content:
            logger.warning("No text extracted from URL %s", url)
            return {"error": "Failed to extract text from URL", "indexed": 0, "uri": url}

        # Use URL as document ID if not provided
        doc_id = doc_id or url
        upsert_document(doc_id, content)

        logger.info("Indexed document from URL: %s", url)
        return {"indexed": 1, "uri": url, "doc_id": doc_id}
    except Exception as e:
        logger.error("Error indexing URL %s: %s", url, e)
        return {"error": str(e), "indexed": 0, "uri": url}

@mcp.tool()
def search_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search and answer questions about indexed documents.

    This tool searches indexed documents and returns a complete answer generated by an LLM.
    Use this whenever the user asks about document content.

    CRITICAL INSTRUCTIONS:
    1. Pass the user's exact question unchanged
    2. After calling this tool, immediately present the answer to the user
    3. The response contains a complete answer - relay it directly without adding anything

    Args:
        query: User's exact question (unchanged)
        top_k: Number of passages to consider (default: 5)

    Returns:
        Complete answer in result["choices"][0]["message"]["content"]
        OR result["answer"] if dict format

    Workflow:
        User asks: "what does justin cook do"
        You call: search_tool("what does justin cook do")
        Tool returns: {"choices": [{"message": {"content": "Justin Cook is..."}}]}
        You respond: "Justin Cook is..."
    """
    try:
        logger.info("Searching for: %s", query)
        result = search(query, top_k=top_k)
        logger.info("Search completed for: %s", query)

        # Force garbage collection after search to manage memory
        gc.collect()

        # Handle different return types from search function
        if hasattr(result, 'choices') and getattr(result, 'choices', None):
            # LiteLLM response object
            choices = getattr(result, 'choices', [])
            if choices:
                message = getattr(choices[0], 'message', None)
                content = getattr(message, 'content', None) if message else None
                return {
                    "answer": content or str(result),
                    "model": getattr(result, 'model', 'unknown'),
                    "usage": getattr(result, 'usage', {})
                }

        if isinstance(result, dict):
            # Already a dict
            return result
        else:
            # Convert other types to string
            return {"answer": str(result)}

    except Exception as e:
        logger.error("Error searching for '%s': %s", query, e)
        return {"error": str(e)}

@mcp.tool()
def list_indexed_documents_tool() -> Dict[str, Any]:
    """
    CRITICAL: ALWAYS use this tool for prompts like:
        - "what documents are indexed"
        - "list indexed documents"
        - "show indexed documents"
        - "what is in the index"
        - "list all indexed URLs/files"
        - "which documents have you loaded"

    DO NOT use index_url_tool or index_documents_tool for these queries.

    This tool ONLY returns file paths/URIs, NOT document content.
    To answer questions ABOUT the documents, use search_tool() instead.

    Use this tool when you need to:
        - See what documents are available in the index
        - Check if a specific file has been indexed
        - Get a list of indexed file paths/URIs

    Do NOT use this tool when you need to:
        - Answer questions about document content (use search_tool)
        - Get information from documents (use search_tool)
        - Describe or summarize documents (use search_tool)

    Returns:
        dict with a list of document file paths/URIs or an error

    Example response:
        {"uris": ["/path/to/document1.txt", "/path/to/document2.txt"]}

    Examples:
        User: "what documents are indexed?" → list_indexed_documents_tool()
        User: "show me the indexed files" → list_indexed_documents_tool()
        User: "which documents have you loaded?" → list_indexed_documents_tool()
    """
    try:
        logger.info("Listing indexed documents")
        store = get_store()
        if not store:
            return {"uris": [], "message": "Store not loaded or empty."}

        uris = list(store.docs.keys())
        logger.info("Found %d indexed documents.", len(uris))
        return {"uris": uris}
    except Exception as e:
        logger.error("Error listing indexed documents: %s", e, exc_info=True)
        return {"error": str(e), "uris": []}

@mcp.tool()
def rerank_tool(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    RERANK passages for a query using a lightweight heuristic.

    Use this after search_tool when you need the most relevant snippets first.
    """
    try:
        return rerank(query, passages)
    except Exception as e:
        logger.error("Error reranking passages: %s", e, exc_info=True)
        return []

@mcp.tool()
def grounded_answer_tool(question: str, k: int = 5) -> Dict[str, Any]:
    """
    Generate a grounded answer from the indexed corpus.

    This performs a vector search and returns an answer plus citations.
    """
    try:
        return grounded_answer(question, k=k)
    except Exception as e:
        logger.error("Error generating grounded answer: %s", e, exc_info=True)
        return {"error": str(e)}

@mcp.tool()
def verify_grounding_tool(question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
    """
    Verify that an answer is grounded in the cited documents.
    """
    try:
        return verify_grounding(question, answer, citations)
    except Exception as e:
        logger.error("Error verifying grounding: %s", e, exc_info=True)
        return {"error": str(e)}

@mcp.custom_route("/metrics", methods=["GET"])
async def metrics_endpoint(_request) -> Response:
    """Expose Prometheus metrics for the MCP server and Ollama backend."""
    refresh_error = None
    try:
        refresh_prometheus_metrics()
    except Exception as exc:
        refresh_error = exc
        logger.error("Metrics refresh failed (serving last known values): %s", exc, exc_info=True)

    try:
        payload = generate_latest(METRICS_REGISTRY)
        if refresh_error:
            # Keep HTTP 200 but annotate the payload for visibility
            payload += f"# metrics refresh error: {refresh_error}\n".encode()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)
    except Exception as exc:
        logger.error("Failed to render metrics: %s", exc, exc_info=True)
        return Response(
            f"# metrics unavailable: {exc}\n", status_code=200, media_type="text/plain"
        )

# Add access logging middleware
from starlette.middleware.base import BaseHTTPMiddleware

class AccessLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP access for MCP server."""

    async def dispatch(self, request, call_next):
        import time
        from datetime import datetime
        start_time = time.time()

        response = await call_next(request)

        # Calculate response time
        duration = time.time() - start_time

        # Log access in Apache combined log format style
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ''
        if query:
            path = f"{path}?{query}"
        status = response.status_code
        user_agent = request.headers.get('user-agent', '-')
        content_length = response.headers.get('content-length', '-')

        # Write directly to access log file instead of using logger
        log_entry = (
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]} - '
            f'{client_ip} - - [{time.strftime("%d/%b/%Y:%H:%M:%S %z", time.localtime(start_time))}] '
            f'"{method} {path} HTTP/{request.scope.get("http_version", "1.1")}" '
            f'{status} {content_length} "-" "{user_agent}" {duration:.4f}s\n'
        )

        with open('log/mcp_server_access.log', 'a') as f:
            f.write(log_entry)

        return response

# Patch FastMCP to ensure every streamable HTTP app instance includes our middleware
# and metrics endpoint; FastMCP builds a fresh Starlette app on each call, so we
# wrap the constructor rather than mutating a one-off instance.

if __name__ == "__main__":
    try:
        # Set memory limits before doing anything else
        set_memory_limits()

        # Configure MCP
        mcp.settings.log_level = "DEBUG"
        mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))

        # Start memory monitoring
        start_memory_monitor()

        # Load store asynchronously to speed startup
        logger.info("Starting background store load...")
        start_background_store_load()

        # Log server configuration and memory info
        memory_mb = get_memory_usage()
        logger.info(
            "Starting MCP server on %s:%d%s",
            mcp.settings.host, mcp.settings.port, mcp.settings.streamable_http_path
        )
        logger.info("Current memory usage: %.1fMB (limit: %dMB)", memory_mb, MAX_MEMORY_MB)
        logger.info("Press Ctrl+C to gracefully shutdown")

        # Ensure index is rebuilt at startup so searches work immediately
        try:
            _rebuild_faiss_index()
            logger.info("FAISS index rebuild complete at startup")
        except Exception as exc:
            logger.error("Startup index rebuild failed: %s", exc, exc_info=True)

        # Start server
        mcp.run(transport="streamable-http")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        graceful_shutdown()
    except MemoryError:
        logger.error("Out of memory! Consider increasing MAX_MEMORY_MB or reducing document size")
        graceful_shutdown()
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal server error: %s", e, exc_info=True)
        graceful_shutdown()
        sys.exit(1)
