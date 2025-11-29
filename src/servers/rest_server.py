#!/usr/bin/env python3
# pylint: disable=too-many-lines
"""
Provide a REST API for the retrieval server using FastAPI.
Run with:
    uvicorn rest_server:app --host
"""

import os
import re
import sys
import json
import logging
import time
import socket
import signal
import subprocess
import threading
import queue
import asyncio
import uuid
import inspect
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Callable, Tuple
from contextlib import asynccontextmanager

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
from src.core.models import (
    IndexPathReq, SearchReq, UpsertReq, IndexUrlReq, LoadStoreReq,
    GroundedAnswerReq, RerankReq, VerifyReq, VerifySimpleReq,
    HealthResp, DocumentInfo, DocumentsResp, DeleteDocsReq,
    ConfigModeReq, QualityMetricsResp, Job, OpenAIConfigModel,
    ChatMessage, ChatReq, VertexConfigReq, AppConfigReq
)
from src.core.interfaces import RAGBackend
from src.core.google_auth import GoogleAuthManager
from src.core.extractors import extract_text_from_bytes
from src.core import rag_core
from src.core.rag_core import verify_grounding_simple

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

# Check for debug mode in config before setting log level
CONFIG_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "settings.json"
_initial_debug_mode = False
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            initial_config = json.load(f)
            _initial_debug_mode = bool(initial_config.get("debugMode", False))
    except Exception:
        pass

_initial_log_level = logging.DEBUG if _initial_debug_mode else logging.INFO

root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.setLevel(_initial_log_level)
for h in log_handlers:
    root_logger.addHandler(h)
root_logger.propagate = False

logger = logging.getLogger(__name__)
logger.setLevel(_initial_log_level)
# Let this module logger bubble up to root handlers configured above
logger.handlers.clear()
logger.propagate = True

# Access logger (HTTP access logs)
access_logger = logging.getLogger('rest_access')
access_logger.setLevel(logging.INFO)
access_logger.handlers.clear()
access_logger.propagate = False

PROCESS_CONTROLLER_PATH = Path(__file__).resolve().parent.parent / "utils" / "process_controller.py"
PROCESS_CONTROLLER_LOG = LOG_DIR / "process_controller.log"

DEFAULT_PORT_RANGES = {
    "rest": (int(os.getenv("RAG_PORT", "8001")), int(os.getenv("RAG_PORT", "8001")) + 9),
    "mcp": (int(os.getenv("MCP_PORT", "8000")), int(os.getenv("MCP_PORT", "8000")) + 9),
    "ui": (int(os.getenv("UI_PORT", "5173")), int(os.getenv("UI_PORT", "5173")) + 9),
    "ollama": (int(os.getenv("OLLAMA_PORT", "11434")), int(os.getenv("OLLAMA_PORT", "11434")) + 9),
}
SUPPORTED_SERVICES = {"rest", "mcp", "ui", "ollama"}
SERVICE_CONTROLLERS: Dict[str, Dict[str, Any]] = {}
SERVICE_LOCK = threading.Lock()

# Background store loading
store_loading = False
store_loaded = False
store_load_thread: Optional[threading.Thread] = None

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

def update_logging_level(debug_mode: bool):
    """Update logging levels for all loggers based on debug mode setting."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Update root logger
    root_logger.setLevel(log_level)
    logger.setLevel(log_level)
    access_logger.setLevel(log_level)
    
    # Update core loggers
    logging.getLogger("src.core").setLevel(log_level)
    logging.getLogger("src.core.rag_core").setLevel(log_level)
    logging.getLogger("src.core.factory").setLevel(log_level)
    logging.getLogger("src.core.faiss_index").setLevel(log_level)
    
    # Update server loggers
    logging.getLogger("src.servers").setLevel(log_level)
    logging.getLogger("src.servers.rest_server").setLevel(log_level)
    
    # Update environment variable for rag_core DEBUG_MODE
    os.environ["RAG_DEBUG_MODE"] = "true" if debug_mode else "false"
    
    logger.info("Logging level updated to %s (debug_mode=%s)", 
                logging.getLevelName(log_level), debug_mode)

def load_app_config():
    """Load application configuration from settings.json."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Update environment variables for MCP
            if "mcpHost" in config:
                os.environ["MCP_HOST"] = config["mcpHost"]
            if "mcpPort" in config:
                os.environ["MCP_PORT"] = str(config["mcpPort"])
            if "mcpPath" in config:
                os.environ["MCP_PATH"] = config["mcpPath"]

            # Update RAG Core configuration
            # Map settings.json keys to rag_core variables
            if "apiEndpoint" in config:
                rag_core.OLLAMA_API_BASE = config["apiEndpoint"]
            elif "ollamaApiUrl" in config:  # Legacy fallback
                rag_core.OLLAMA_API_BASE = config["ollamaApiUrl"]

            if "model" in config:
                rag_core.LLM_MODEL_NAME = config["model"]
                rag_core.ASYNC_LLM_MODEL_NAME = config["model"].split("/")[-1]
            elif "ollamaModel" in config:  # Legacy fallback
                rag_core.LLM_MODEL_NAME = config["ollamaModel"]
                rag_core.ASYNC_LLM_MODEL_NAME = config["ollamaModel"].split("/")[-1]

            if "embeddingModel" in config:
                rag_core.EMBED_MODEL_NAME = config["embeddingModel"]

            logger.info("Loaded configuration from %s", CONFIG_FILE)
            return config
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load configuration: %s", e)
    return {}

app_config = load_app_config()

# Apply debug mode from settings on startup
if app_config and "debugMode" in app_config:
    update_logging_level(bool(app_config["debugMode"]))

# Get base path (normalize leading/trailing slashes)
_pth_env = os.getenv("RAG_PATH", "api")
pth = _pth_env.strip("/") or "api"

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

def _notify_mcp_logging_update(debug_mode: bool):
    """Notify MCP server to update its logging level."""
    try:
        _proxy_to_mcp("POST", "/rest/config/logging", {"debug_mode": debug_mode})
        logger.info("Notified MCP server to update logging level (debug_mode=%s)", debug_mode)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Don't fail config save if MCP notification fails
        logger.debug("MCP logging notification failed (non-critical): %s", exc)

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
    path_suffix = path if path.startswith('/') else '/' + path
    urls.append(f"http://{MCP_HOST}:{MCP_PORT}{path_suffix}")

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
class ServerState:  # pylint: disable=too-many-instance-attributes
    """Mutable server state."""
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

    # Daily metrics
    today_date: str = date.today().isoformat()
    queries_today: int = 0
    documents_added_today: int = 0
    latency_sum_today: float = 0.0
    latency_count_today: int = 0

state = ServerState()


def _reset_today_if_needed() -> None:
    """Reset per-day counters when the calendar day changes."""
    current = date.today().isoformat()
    if state.today_date != current:
        state.today_date = current
        state.queries_today = 0
        state.documents_added_today = 0
        state.latency_sum_today = 0.0
        state.latency_count_today = 0


def _record_today_request(duration_seconds: float, path: str) -> None:
    """Update daily request counters (search-only to match dashboard intent)."""
    try:
        _reset_today_if_needed()
        if "/search" not in str(path):
            return
        state.queries_today += 1
        state.latency_sum_today += max(duration_seconds, 0.0)
        state.latency_count_today += 1
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("record today metrics failed: %s", exc)


def _record_documents_added(count: int) -> None:
    """Update daily documents added counter."""
    try:
        _reset_today_if_needed()
        state.documents_added_today += max(0, int(count))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("record documents added failed: %s", exc)


def _fetch_mcp_memory_mb() -> Optional[float]:
    """Scrape MCP /metrics for memory usage and return MB if available."""
    prefix = MCP_PATH.strip("/")
    urls = [f"http://{MCP_HOST}:{MCP_PORT}/metrics"]
    if prefix:
        urls.insert(0, f"http://{MCP_HOST}:{MCP_PORT}/{prefix}/metrics")

    for url in urls:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            body = resp.text
            match = re.search(r"mcp_memory_usage_megabytes\\s+(\\d+(?:\\.\\d+)?)", body)
            if match:
                return float(match.group(1))
        except requests.RequestException:
            continue
    return None


def _parse_port_range(service: str) -> Tuple[int, int]:
    """Return a port range for the given service (env override or default)."""
    env_key = f"{service.upper()}_PORT_RANGE"
    override = os.getenv(env_key)
    if override and "-" in override:
        try:
            start, end = override.split("-", 1)
            return int(start), int(end)
        except ValueError:
            logger.warning("Invalid %s, using defaults: %s", env_key, override)
    return DEFAULT_PORT_RANGES.get(service, (0, 0))


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _is_service_running(service: str) -> bool:
    """Check if a service is running by checking if its port is in use."""
    host = _service_host(service)
    start_port, end_port = _parse_port_range(service)
    if not start_port:
        return False
    
    # Check if any port in the service's range is in use
    for port in range(start_port, end_port + 1):
        if not _is_port_free(host, port):
            return True
    return False


def _pick_port(service: str, host: str) -> int:
    start, end = _parse_port_range(service)
    if not start or not end:
        raise RuntimeError(f"No port range configured for service '{service}'")
    for port in range(start, end + 1):
        if _is_port_free(host, port):
            return port
    raise RuntimeError(f"No available ports for {service} in range {start}-{end}")


def _service_host(service: str) -> str:
    if service == "rest":
        return os.getenv("RAG_HOST", "127.0.0.1")
    if service == "mcp":
        return os.getenv("MCP_HOST", "127.0.0.1")
    if service == "ui":
        return os.getenv("UI_HOST", "127.0.0.1")
    if service == "ollama":
        return "127.0.0.1"
    return "127.0.0.1"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_service_controller(service: str) -> Dict[str, Any]:
    """Launch a controller daemon for the requested service."""
    if not PROCESS_CONTROLLER_PATH.exists():
        raise RuntimeError("process_controller.py is missing; cannot start controller.")

    host = _service_host(service)
    port = _pick_port(service, host)
    start, end = _parse_port_range(service)
    port_range_arg = f"{start}-{end}"

    PROCESS_CONTROLLER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROCESS_CONTROLLER_LOG.open("a", encoding="utf-8") as ctl_log:
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                sys.executable,
                str(PROCESS_CONTROLLER_PATH),
                "--service",
                service,
                "--host",
                host,
                "--port",
                str(port),
                "--port-range",
                port_range_arg,
            ],
            stdout=ctl_log,
            stderr=ctl_log,
        )

    SERVICE_CONTROLLERS[service] = {
        "pid": proc.pid,
        "port": port,
        "host": host,
        "started_at": time.time(),
    }
    logger.info("Started %s controller (PID %s) on %s:%s", service, proc.pid, host, port)
    return {
        "service": service,
        "status": "starting",
        "controller_pid": proc.pid,
        "port": port,
        "host": host,
        "port_range": port_range_arg,
    }


def _stop_service_controller(service: str) -> Dict[str, Any]:
    """Stop a running controller (and its child) if present."""
    info = SERVICE_CONTROLLERS.get(service)
    if not info:
        return {"service": service, "status": "not_running"}

    pid = info.get("pid")
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        # brief wait
        for _ in range(10):
            if not _pid_alive(pid):
                break
            time.sleep(0.5)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    SERVICE_CONTROLLERS.pop(service, None)
    logger.info("Stopped controller for %s (PID %s)", service, pid)
    return {"service": service, "status": "stopped", "controller_pid": pid}

@asynccontextmanager
async def lifespan(_fastapi_app: FastAPI):
    """Manage application lifecycle."""
    logger.info(
        "REST server startup complete (host=%s port=%s base=/api)",
        os.getenv("RAG_HOST", "127.0.0.1"),
        os.getenv("RAG_PORT", "8001")
    )

    yield
    logger.info("REST server shutting down")
    try:
        backend.save_store()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error saving store on shutdown: %s", e)

app = FastAPI(title="retrieval-rest-server", lifespan=lifespan)

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
MCP_MEMORY_USAGE_MB = Gauge(
    "mcp_memory_usage_megabytes",
    "Current MCP server process RSS memory in MB (proxied via REST).",
)

def _start_search_worker():
    """Start the background search worker thread if not already started."""
    if state.search_worker_started:
        return

    def _worker():
        while True:
            job = state.search_job_queue.get()
            if not job:
                continue
            job_id = job["id"]
            try:
                # Build kwargs from job parameters
                kwargs = {}
                if job.get("top_k") is not None:
                    kwargs['top_k'] = job["top_k"]
                if job.get("model") is not None:
                    kwargs['model'] = job["model"]
                if job.get("temperature") is not None:
                    kwargs['temperature'] = job["temperature"]
                if job.get("max_tokens") is not None:
                    kwargs['max_tokens'] = job["max_tokens"]
                
                result = backend.search(job["query"], **kwargs)
                status = "completed"
                error = None
            except Exception as exc:  # pylint: disable=broad-exception-caught
                result = None
                status = "failed"
                error = str(exc)
            with state.search_jobs_lock:
                state.search_jobs[job_id].update({
                    "status": status,
                    "result": result,
                    "error": error
                })
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
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit for upserts

def _get_memory_usage_mb() -> float:
    """Return current process RSS in megabytes."""
    return psutil.Process().memory_info().rss / 1024 / 1024
def _record_quality_metrics(
    result: Optional[Dict[str, Any]],
    error: Optional[Exception] = None
) -> None:
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
        mcp_mem = _fetch_mcp_memory_mb()
        if mcp_mem is not None:
            MCP_MEMORY_USAGE_MB.set(mcp_mem)
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
    except Exception:  # pylint: disable=broad-exception-caught
        status_code = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_INFLIGHT.dec()
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(method=method, path=path, status=status_code).observe(duration)
        _record_today_request(duration, path)

ACCESS_LOG_PATH = LOG_DIR / "rest_server_access.log"

def _append_access_log(log_entry: str) -> None:
    """Write and flush an access log line for reliable tail/follow."""
    with open(ACCESS_LOG_PATH, "a", buffering=1, encoding="utf-8") as f:
        f.write(log_entry)
        f.flush()
        os.fsync(f.fileno())

@app.post(f"/{pth}/upsert_document")
def api_upsert(req: UpsertReq):
    """Upsert a document into the store."""
    logger.info("Upserting document: uri=%s", req.uri)
    try:
        if not req.text and not req.binary_base64:
            return JSONResponse(
                {"error": "either text or binary_base64 is required"},
                status_code=422
            )

        # Enforce size limits
        payload_size = len(req.text) if req.text else 0
        if req.binary_base64:
            payload_size += len(req.binary_base64)

        if payload_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Payload too large ({payload_size} bytes). "
                    f"Limit is {MAX_UPLOAD_SIZE_BYTES} bytes."
                )
            )

        # Proxy to MCP worker which creates jobs for progress tracking
        result = _proxy_to_mcp(
            "POST",
            "/rest/upsert_document",
            {"uri": req.uri, "text": req.text, "binary_base64": req.binary_base64}
        )
        if isinstance(result, dict) and result.get("upserted") and not result.get("existed"):
            _record_documents_added(1)
        return result
    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error upserting document %s: %s", req.uri, e)
        raise


@app.post(f"/{pth}/index_url")
def api_index_url(req: IndexUrlReq):
    """Download and index a remote URL via MCP worker."""
    logger.info("Index URL requested: %s", req.url)
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Only http and https URLs are supported")
    try:
        payload = {"url": req.url, "doc_id": req.doc_id}
        return _proxy_to_mcp("POST", "/rest/index_url", payload)
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Error indexing URL %s: %s", req.url, exc)
        raise

@app.post(f"/{pth}/index_path")
def api_index_path(req: IndexPathReq):
    """Index a filesystem path into the retrieval store."""
    logger.info("Indexing path: path=%s, glob=%s", req.path, req.glob)
    try:
        result = backend.index_path(req.path, req.glob or "**/*")
        if isinstance(result, dict) and "indexed" in result:
            _record_documents_added(int(result.get("indexed") or 0))
        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error indexing path %s: %s", req.path, e)
        raise

@app.post(f"/{pth}/search")
def api_search(req: SearchReq):
    """Search the retrieval store."""
    logger.info("Processing search query: %s", req.query)
    # Async mode: queue job and return immediately
    if getattr(req, "async_mode", False):
        _start_search_worker()
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": "search",
            "status": "queued",
            "query": req.query,
            "timeout_seconds": req.timeout_seconds or 300,
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_k": req.top_k,
        }
        with state.search_jobs_lock:
            state.search_jobs[job_id] = job
        state.search_job_queue.put(job)
        return {"job_id": job_id, "status": "queued"}

    try:
        # Build kwargs for backend.search
        kwargs = {}
        if req.top_k is not None:
            kwargs['top_k'] = req.top_k
        if req.model is not None:
            kwargs['model'] = req.model
        if req.temperature is not None:
            kwargs['temperature'] = req.temperature
        if req.max_tokens is not None:
            kwargs['max_tokens'] = req.max_tokens
        
        result = backend.search(req.query, **kwargs)
        _record_quality_metrics(result)
        # Ensure result is JSON-serializable (already normalized by _normalize_llm_response)
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        return result
    except HTTPException as e:
        _record_quality_metrics(None, error=e)
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error processing search query '%s': %s", req.query, e)
        _record_quality_metrics(None, error=e)
        raise HTTPException(
            status_code=502,
            detail={"error": "search failed", "detail": str(e)}
        ) from e

@app.post(f"/{pth}/load_store")
def api_load(_req: LoadStoreReq):
    """Send the current store to the LLM."""
    logger.info("Loading store to LLM")
    try:
        backend.load_store()
        return {"status": "loaded"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error loading store to LLM: %s", e)
        raise

@app.get("/metrics")
def metrics_endpoint():
    """Expose Prometheus metrics for the REST server."""
    refresh_error = None
    try:
        refresh_prometheus_metrics()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        refresh_error = exc
        logger.error("Metrics refresh failed (serving last known values): %s", exc, exc_info=True)
    try:
        payload = generate_latest()
        if refresh_error:
            payload += f"# metrics refresh error: {refresh_error}\n".encode()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
        if req.model:
             kwargs["model"] = req.model
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.config:
            kwargs.update(req.config)
             
        answer = backend.grounded_answer(req.question, **kwargs)
        return answer
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("rerank failed: %s", exc)
        raise

@app.post(f"/{pth}/verify_grounding")
def api_verify(req: VerifyReq):
    """Verify grounding against store using citation URIs."""
    logger.info("Verify grounding requested")
    try:
        return backend.verify_grounding(req.question, req.draft_answer, req.citations)
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
        return verify_grounding_simple(req.question, req.draft_answer, req.passages)
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
        return HealthResp(
            status="unavailable",
            base_path=f"/{pth}",
            documents=0,
            vectors=0,
            memory_mb=0,
            memory_limit_mb=int(REST_MAX_MEMORY_MB),
            total_size_bytes=0,
            store_file_bytes=0,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Health check failed: %s", exc)
        return HealthResp(
            status="unavailable",
            base_path=f"/{pth}",
            documents=0,
            vectors=0,
            memory_mb=0,
            memory_limit_mb=int(REST_MAX_MEMORY_MB),
            total_size_bytes=0,
            store_file_bytes=0,
        )

@app.get(f"/{pth}/documents", response_model=DocumentsResp)
def api_documents():
    """List all documents in the store."""
    try:
        docs = backend.list_documents()
        return {
            "documents": [
                {"uri": d["uri"], "size_bytes": d.get("size", 0)} for d in docs
            ]
        }
    except HTTPException as exc:
        logger.error("documents list failed: %s", exc.detail)
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("documents list failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={"error": "Backend unreachable", "detail": str(exc)}
        ) from exc

@app.post(f"/{pth}/documents/delete")
def api_documents_delete(req: DeleteDocsReq):
    """Delete documents by URI from the store and rebuild index."""
    try:
        return backend.delete_documents(req.uris)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("documents delete failed: %s", exc)
        return {"deleted": 0, "error": str(exc)}

@app.get(f"/{pth}/documents/delete/status")
def api_deletion_status():
    """Get deletion queue status."""
    try:
        if hasattr(backend, "get_deletion_status"):
            return backend.get_deletion_status()
        # For backends without queue support (like Google), return empty status
        return {
            "queue_size": 0,
            "processing": False,
            "last_completed": None,
            "total_processed": 0
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("deletion status failed: %s", exc)
        return {"error": str(exc)}

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
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{req.mode}' not available. Available: {backend.get_available_modes()}"
        )
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
def api_job(_job_id: str):
    """Get a single async indexing job (pass-through to MCP)."""
    # See api_jobs comment.
    return {"error": "Not implemented for generic jobs, use /search/jobs/{job_id}"}

@app.get(f"/{pth}/search/jobs/{{job_id}}", response_model=dict)
def api_search_job(job_id: str):
    """Get status/result for an async search job."""
    with state.search_jobs_lock:
        job = state.search_jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "not found", "id": job_id}
        )
    return job

@app.post(f"/{pth}/flush_cache")
def api_flush_cache():
    """Flush the document store and delete the backing DB file."""
    try:
        return backend.flush_cache()
    except Exception as exc:  # pylint: disable=broad-exception-caught
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


@app.get(f"/{pth}/metrics/today")
def api_today_metrics():
    """Return same-day aggregates for queries, docs added, and avg latency."""
    _reset_today_if_needed()
    avg_latency_ms = 0.0
    if state.latency_count_today:
        avg_latency_ms = (state.latency_sum_today / state.latency_count_today) * 1000.0
    return {
        "queries_today": state.queries_today,
        "documents_added_today": state.documents_added_today,
        "avg_latency_today": avg_latency_ms,
    }


@app.get(f"/{pth}/services")
def api_service_status():
    """List controller state for managed services."""
    statuses = []
    with SERVICE_LOCK:
        for service in SUPPORTED_SERVICES:
            # Check both controller state and actual port availability
            info = SERVICE_CONTROLLERS.get(service, {})
            pid = info.get("pid")
            controller_alive = bool(pid and _pid_alive(pid))
            
            # Also check if service is actually running on its port (even if not managed by controller)
            port_alive = _is_service_running(service)
            
            # Service is running if either the controller knows about it OR the port is in use
            is_running = controller_alive or port_alive
            
            if info and not controller_alive:
                SERVICE_CONTROLLERS.pop(service, None)
            
            statuses.append({
                "service": service,
                "status": "running" if is_running else "stopped",
                "controller_pid": pid if controller_alive else None,
                "port": info.get("port"),
                "host": info.get("host"),
                "started_at": info.get("started_at"),
            })
    return {"services": statuses}


@app.post(f"/{pth}/services/{{service}}/start")
def api_start_service(service: str):
    """Start a controller for the given service."""
    svc = service.lower()
    if svc not in SUPPORTED_SERVICES:
        raise HTTPException(status_code=400, detail="Unsupported service")

    with SERVICE_LOCK:
        existing = SERVICE_CONTROLLERS.get(svc)
        if existing and existing.get("pid") and _pid_alive(int(existing["pid"])):
            return {
                "service": svc,
                "status": "running",
                "controller_pid": existing["pid"],
                "port": existing.get("port"),
                "host": existing.get("host"),
                "started_at": existing.get("started_at"),
                "message": "already running",
            }
        try:
            return _start_service_controller(svc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to start %s controller: %s", svc, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"/{pth}/services/{{service}}/stop")
def api_stop_service(service: str):
    """Stop a controller for the given service."""
    svc = service.lower()
    if svc not in SUPPORTED_SERVICES:
        raise HTTPException(status_code=400, detail="Unsupported service")

    with SERVICE_LOCK:
        try:
            return _stop_service_controller(svc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to stop %s controller: %s", svc, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(f"/{pth}/services/{{service}}/restart")
def api_restart_service(service: str):
    """Restart a service controller."""
    svc = service.lower()
    if svc not in SUPPORTED_SERVICES:
        raise HTTPException(status_code=400, detail="Unsupported service")

    with SERVICE_LOCK:
        stop_res = _stop_service_controller(svc)
        try:
            start_res = _start_service_controller(svc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to restart %s: %s", svc, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"stop": stop_res, "start": start_res}

@app.get(f"/{pth}/auth/login")
def auth_login(request: Request):
    """Initiate Google OAuth2 flow."""
    # Construct callback URL based on the request's hos
    # This requires the Google Cloud Console to have this exact URI whitelisted
    redirect_uri = str(request.url_for('auth_callback'))
    
    # Force localhost if 127.0.0.1 is used, as Google often requires localhost
    if "127.0.0.1" in redirect_uri:
        redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")
    
    # If running behind a proxy (like ngrok or in some container setups), 
    # you might need to force https or a specific host.
    # For now, we trust the request headers.
    
    logger.info("Initiating auth flow with redirect_uri: %s", redirect_uri)
    
    try:
        flow = auth_manager.flow_from_client_secrets(redirect_uri=redirect_uri)
        authorization_url, _oauth_state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        logger.info("Generated auth URL: %s", authorization_url)
        response = RedirectResponse(authorization_url)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error initiating auth: %s", e)
        # If secrets/client_secrets.json is missing, redirect to setup page
        if "Client secrets file not found" in str(e) or "No such file" in str(e):
             return RedirectResponse(url=request.url_for('auth_setup'))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/auth/setup", name="auth_setup")
def auth_setup(_request: Request):
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
    """Save the provided credentials to secrets/client_secrets.json."""
    
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
        os.makedirs("secrets", exist_ok=True)
        with open("secrets/client_secrets.json", "w", encoding="utf-8") as f:
            json.dump(secrets, f, indent=2)
            
        # Redirect back to login to start the flow
        return RedirectResponse(url=request.url_for('auth_login'), status_code=303)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save secrets: %s", e)
        return HTMLResponse(f"Failed to save configuration: {e}", status_code=500)

@app.get(f"/{pth}/auth/callback", name="auth_callback")
def auth_callback(request: Request, code: str, _oauth_state: Optional[str] = None):
    """Handle Google OAuth2 callback."""
    redirect_uri = str(request.url_for('auth_callback'))
    logger.info("Handling auth callback with redirect_uri: %s", redirect_uri)
    
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
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error in auth callback: %s", e)
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
def auth_logout(_request: Request):
    """Log out by deleting the stored token."""
    try:
        logger.info("Logout requested. Checking for token file...")
        auth_manager.logout()
        if hasattr(backend, 'logout'):
            backend.logout()
        logger.info("Logout successful")
        return JSONResponse({"status": "logged_out"})
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error logging out: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(f"/{pth}/config/openai")
def save_openai_config(_request: Request, config: OpenAIConfigModel):
    """Save OpenAI API configuration to secrets/openai_config.json."""
    try:
        os.makedirs("secrets", exist_ok=True)
        config_data = {
            "api_key": config.api_key,
            "model": config.model,
            "assistant_id": config.assistant_id
        }
        with open("secrets/openai_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("OpenAI configuration saved to secrets/openai_config.json")
        return JSONResponse({"status": "success", "message": "Configuration saved"})
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save OpenAI config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(f"/{pth}/config/openai/reload")
def reload_openai_backend(_request: Request):
    """Reload OpenAI backend to pick up configuration changes."""
    global backend  # pylint: disable=global-statement
    try:
        if not backend:
            raise HTTPException(status_code=500, detail="Backend not initialized")
        
        result = backend.reload_backend("openai_assistants")
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        logger.info("OpenAI backend reloaded successfully")
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to reload OpenAI backend: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/config/openai")
def get_openai_config(_request: Request):
    """Get OpenAI configuration (without exposing the full API key)."""
    try:
        if not os.path.exists("secrets/openai_config.json"):
            return JSONResponse({"api_key": "", "model": "gpt-4-turbo-preview", "assistant_id": ""})
        
        with open("secrets/openai_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Mask the API key for security (show only last 4 chars)
        api_key = config.get("api_key", "")
        if len(api_key) > 4:
            masked_key = "sk-..." + api_key[-4:]
        else:
            masked_key = ""
        
        return JSONResponse({
            "api_key": masked_key,
            "model": config.get("model", "gpt-4-turbo-preview"),
            "assistant_id": config.get("assistant_id", "")
        })
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to load OpenAI config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/config/openai/models")
def get_openai_models(_request: Request):
    """Get list of available OpenAI models using the configured API key."""
    try:
        if not os.path.exists("secrets/openai_config.json"):
            raise HTTPException(status_code=400, detail="OpenAI not configured")
        
        with open("secrets/openai_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        api_key = config.get("api_key", "")
        if not api_key or not api_key.strip():
            raise HTTPException(status_code=400, detail="API key not configured")
        
        # Fetch models from OpenAI API
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models_response = client.models.list()
        
        all_model_ids = [m.id for m in models_response.data]
        
        # Filter for models suitable for assistants (GPT, o1, and text models)
        suitable_models = []
        for model in models_response.data:
            model_id = model.id
            # Include GPT-4, GPT-3.5, o1, and text-davinci models
            # Exclude embeddings, whisper, tts, dall-e, and sora models
            if any(prefix in model_id for prefix in ['gpt-4', 'gpt-3.5', 'o1', 'text-davinci']):
                suitable_models.append({
                    "id": model_id,
                    "name": model_id,
                    "created": model.created
                })
        
        # If no GPT models found, return warning
        if not suitable_models:
            logger.warning("No GPT models found. Available models: %s", all_model_ids)
            return JSONResponse({
                "models": [],
                "warning": "No GPT models available",
                "message": f"Your OpenAI account only has access to: {', '.join(all_model_ids)}. The Assistants API requires GPT-4, GPT-3.5-turbo, or o1 models. Please add credits to your OpenAI account or use a different API key with GPT model access.",
                "available_models": all_model_ids
            })
        
        # Sort by creation date (newest first)
        suitable_models.sort(key=lambda x: x["created"], reverse=True)
        
        logger.info("Retrieved %d OpenAI models (from %d total)", len(suitable_models), len(models_response.data))
        return JSONResponse({"models": suitable_models})
        
    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to fetch OpenAI models: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(f"/{pth}/chat")
def api_chat(req: ChatReq):
    """Conversational chat."""
    logger.info("Chat request received")
    if hasattr(backend, "chat"):
        # Prepare kwargs
        kwargs = {}
        if req.model:
            kwargs["model"] = req.model
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.config:
            kwargs.update(req.config)

        return backend.chat([m.model_dump() for m in req.messages], **kwargs)
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
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(f"/{pth}/config/app")
def api_save_app_config(req: AppConfigReq):
    """Save application configuration."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "settings.json"

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            # Dump using aliases to match frontend expectation when reading back,
            # or dump by name and let frontend handle it?
            # Frontend expects camelCase. Pydantic .model_dump(by_alias=True) does this.
            json.dump(req.model_dump(by_alias=True), f, indent=2)
        
        # Reload configuration in rag_core
        try:
            from src.core import rag_core
            rag_core.reload_settings()
            logger.info("Reloaded rag_core configuration")
        except Exception as reload_err:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to reload rag_core: %s", reload_err)
        
        # Update logging level immediately based on debug mode
        debug_mode = req.debug_mode if req.debug_mode is not None else False
        update_logging_level(debug_mode)
        
        # Notify MCP server to update its logging level via proxy
        try:
            _notify_mcp_logging_update(debug_mode)
        except Exception as notify_err:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to notify MCP server of logging change: %s", notify_err)
        
        return {"status": "saved"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save app config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/config/app")
def api_get_app_config():
    """Get application configuration."""
    config_path = Path("config") / "settings.json"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Return empty or defaults if not found
        return {}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to read app config: %s", e)
        return {}

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
        with open("vertex_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # Update current process env vars so it works immediately
        os.environ.update(config)
        
        return {"status": "saved"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save vertex config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/config/vertex")
def api_get_vertex_config():
    """Get Vertex AI configuration."""
    try:
        if os.path.exists("vertex_config.json"):
            with open("vertex_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "VERTEX_PROJECT_ID": os.getenv("VERTEX_PROJECT_ID", ""),
            "VERTEX_LOCATION": os.getenv("VERTEX_LOCATION", "us-central1"),
            "VERTEX_DATA_STORE_ID": os.getenv("VERTEX_DATA_STORE_ID", "")
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to read vertex config: %s", e)
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
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to read log file %s: %s", log_type, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            except Exception as e:  # pylint: disable=broad-exception-caught
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
            except Exception as e:  # pylint: disable=broad-exception-caught
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
        
        logger.info("Starting REST server on %s:%s", app.host, app.port)
        logger.info("API base path: /%s", pth)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Server startup error: %s", e)
        sys.exit(1)
