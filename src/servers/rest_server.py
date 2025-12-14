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
import warnings
import time
import socket
import signal
import subprocess
import threading
import queue
import asyncio
import uuid
import inspect
import stat
import hmac
from ipaddress import ip_address
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Callable, Tuple
from contextlib import asynccontextmanager

import psutil
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, Form, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

from src.core.exceptions import ConfigurationError, AuthenticationError, ProviderError
from pydantic import ConfigDict, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.core.factory import get_rag_backend
from src.core.models import (
    IndexPathReq, SearchReq, UpsertReq, IndexUrlReq,
    GroundedAnswerReq, RerankReq, VerifyReq, VerifySimpleReq,
    HealthResp, DocumentInfo, DocumentsResp, DeleteDocsReq,
    ConfigModeReq, QualityMetricsResp, Job, OpenAIConfigModel,
    ChatMessage, ChatReq, VertexConfigReq, AppConfigReq,
    OllamaModeReq, OllamaTestConnectionReq, OllamaCloudConfigReq,
    OllamaStatusResp, OllamaTestConnectionResp, OllamaModelsReq,
    OpenAIModelsReq
)
from src.core.interfaces import RAGBackend
from src.core.google_auth import GoogleAuthManager
from src.core.extractors import extract_text_from_bytes
from src.core import rag_core
from src.core.rag_core import verify_grounding_simple
from src.core import pgvector_store
from src.core.config_paths import (
    CONFIG_DIR,
    SETTINGS_PATH,
    VERTEX_CONFIG_PATH,
    get_ca_bundle_path,
)
from src.core.ollama_config import (
    _redact_api_key,  # pylint: disable=protected-access
    get_ollama_mode,
    get_ollama_api_key,
)


def _extract_persistable_text(payload: Dict[str, Any]) -> Optional[str]:
    """Extract assistant text from a backend response payload.

    This is intentionally backend-agnostic and only used for persistence.
    """
    if not isinstance(payload, dict):
        return None

    # Never persist error payloads as assistant messages.
    if payload.get("status") == "error" or payload.get("error"):
        return None

    for key in ("content", "answer", "response", "grounded_answer"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val

    # OpenAI/LiteLLM-style
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content

    return None


def _extract_sources(payload: Dict[str, Any]) -> List[str]:
    """Extract sources/citations from a backend response payload."""
    if not isinstance(payload, dict):
        return []

    raw = payload.get("sources")
    if raw is None:
        raw = payload.get("citations")

    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x).strip()]
    return []


class PgvectorConfigModel(BaseModel):
    """Request model for pgvector connection configuration."""

    host: str = "127.0.0.1"
    port: int = 5432
    dbname: str = "agentic_rag"
    user: str = "agenticrag"
    password: Optional[str] = None

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
CONFIG_FILE = SETTINGS_PATH
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

# Ensure warnings are logged (with timestamps) instead of being printed raw.
logging.captureWarnings(True)
warnings.simplefilter("default")
_warnings_logger = logging.getLogger("py.warnings")
_warnings_logger.handlers.clear()
_warnings_logger.propagate = True

# Ensure uvicorn logs (startup/errors) are emitted via our root handlers so they
# also include timestamps in log/rest_server.log.
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _uv_logger = logging.getLogger(_name)
    _uv_logger.handlers.clear()
    _uv_logger.propagate = True
    _uv_logger.setLevel(_initial_log_level)
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
    logging.getLogger("src.core.pgvector_store").setLevel(log_level)
    
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

            if "model" in config:
                rag_core.LLM_MODEL_NAME = config["model"]
                rag_core.ASYNC_LLM_MODEL_NAME = config["model"].split("/")[-1]

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
MASKED_SECRET = "***MASKED***"

# Chat persistence
try:
    from src.core.chat_store import ChatStore
    chat_store = ChatStore(LOG_DIR.parent / "cache" / "chat_history.db")
    logger.info("ChatStore initialized")
except Exception as exc:
    logger.error("Failed to initialize ChatStore: %s", exc)
    chat_store = None

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

def _mask_secret(value: Optional[str]) -> tuple[str, bool]:
    """
    Return a masked value and flag indicating whether a secret was present.
    
    Args:
        value: Raw secret value.
    
    Returns:
        Tuple of (masked_value, has_value).
    """
    if value and value.strip():
        return MASKED_SECRET, True
    return "", False

def _redact_error_message(message: str, *secrets: Optional[str]) -> str:
    """Redact any provided secrets from a log or HTTP error message."""
    redacted = message or ""
    for secret in secrets:
        redacted = _redact_api_key(redacted, secret)
    return redacted


def _redact_pgvector_error_message(message: str, *secrets: Optional[str]) -> str:
    """Redact pgvector/Postgres connection secrets from messages.

    This is defensive: psycopg/pgvector errors can include conninfo/DSNs with passwords.
    """

    redacted = _redact_error_message(message, *secrets)
    # Mask conninfo fragments like: password=... or postgresql://user:pass@host
    redacted = re.sub(r"(password=)\S+", r"\1***MASKED***", redacted)
    redacted = re.sub(r"(postgres(?:ql)?://[^:\s]+:)[^@\s]+@", r"\1***MASKED***@", redacted)
    redacted = re.sub(r"(POSTGRES_PASSWORD\s*[:=]\s*)\S+", r"\1***MASKED***", redacted, flags=re.IGNORECASE)
    return redacted


def _get_bool_setting(env_key: str, json_key: str, default: bool) -> bool:
    """Read a boolean from env or settings.json (app_config)."""

    raw = os.getenv(env_key)
    if raw is None and isinstance(app_config, dict):
        raw = app_config.get(json_key)

    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_str_setting(env_key: str, json_key: str, default: str) -> str:
    """Read a string from env or settings.json (app_config)."""

    raw = os.getenv(env_key)
    if raw is None and isinstance(app_config, dict):
        raw = app_config.get(json_key)
    if raw is None:
        return default
    return str(raw)


def _get_cors_allow_origins() -> List[str]:
    """Return CORS allowlist.

    Default is localhost-only to reduce CSRF-style risks against local admin endpoints.
    Operators can override via env `RAG_CORS_ALLOW_ORIGINS` (comma-separated) or
    settings.json `corsAllowOrigins`.
    """

    env_val = os.getenv("RAG_CORS_ALLOW_ORIGINS")
    if env_val and env_val.strip():
        return [o.strip() for o in env_val.split(",") if o.strip()]

    cfg_val = None
    if isinstance(app_config, dict):
        cfg_val = app_config.get("corsAllowOrigins")

    if isinstance(cfg_val, list):
        parsed = [str(o).strip() for o in cfg_val if str(o).strip()]
        if parsed:
            return parsed
    if isinstance(cfg_val, str) and cfg_val.strip():
        return [o.strip() for o in cfg_val.split(",") if o.strip()]

    # Safe default: local UI + local REST.
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    ]


def _get_client_host(request: Request) -> str:
    """Return client host, optionally honoring forwarded headers.

    Proxy headers are trusted only when `RAG_ADMIN_TRUST_PROXY=1`.
    """

    trust_proxy = _get_bool_setting("RAG_ADMIN_TRUST_PROXY", "adminTrustProxy", False)
    if trust_proxy:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        xri = request.headers.get("x-real-ip")
        if xri:
            return xri.strip()

    if request.client and request.client.host:
        return request.client.host
    return ""


def _is_loopback_request(request: Request) -> bool:
    """True if the request originated from a loopback address."""

    host = _get_client_host(request)
    # Starlette TestClient uses synthetic hostnames like "testclient"/"testserver".
    # Treat these as loopback so unit tests exercise handlers without requiring
    # an admin token or HTTPS.
    if host in {"testclient", "testserver"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _is_https_request(request: Request) -> bool:
    """True if the request is HTTPS (direct) or forwarded as HTTPS."""

    if request.url.scheme == "https":
        return True
    if _get_bool_setting("RAG_ADMIN_TRUST_PROXY", "adminTrustProxy", False):
        proto = request.headers.get("x-forwarded-proto", "").strip().lower()
        return proto == "https"
    return False


def require_admin_access(request: Request) -> None:
    """Enforce admin access for sensitive endpoints.

    Behavior:
    - Localhost requests (loopback IP) are allowed without auth when mode=nonlocal.
    - Non-local requests require:
      - HTTPS (by default)
      - A valid admin token in `Authorization: Bearer ...` or `X-RAG-Admin-Token`.
    """

    mode = _get_str_setting("RAG_ADMIN_AUTH_MODE", "adminAuthMode", "nonlocal").strip().lower()
    require_https_nonlocal = _get_bool_setting(
        "RAG_ADMIN_REQUIRE_HTTPS_NONLOCAL", "adminRequireHttpsNonLocal", True
    )

    if mode in {"off", "disabled", "0", "false"}:
        return

    is_loopback = _is_loopback_request(request)
    if mode == "nonlocal" and is_loopback:
        return

    if require_https_nonlocal and not is_loopback and not _is_https_request(request):
        raise HTTPException(
            status_code=403,
            detail="HTTPS is required for remote admin endpoints",
        )

    token = os.getenv("RAG_ADMIN_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=500,
            detail="RAG_ADMIN_TOKEN must be set to use admin endpoints remotely",
        )

    provided = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        provided = auth.split(" ", 1)[1].strip()
    if not provided:
        provided = request.headers.get("x-rag-admin-token", "").strip()

    if not provided or not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail="Unauthorized")

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
            # Use configured CA bundle if available
            verify_ssl = get_ca_bundle_path() or True
            resp = requests.request(method, url, json=json_payload, timeout=timeout, verify=verify_ssl)
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
            verify_ssl = get_ca_bundle_path() or True
            resp = requests.get(url, timeout=3, verify=verify_ssl)
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
    try:
        logger.info("Ensuring pgvector schema is up to date...")
        rag_core.ensure_vector_store_ready()
    except Exception as exc:
        logger.error("pgvector schema migration failed (server will likely fail): %s", exc)

    logger.info(
        "REST server startup complete (host=%s port=%s base=/api)",
        os.getenv("RAG_HOST", "127.0.0.1"),
        os.getenv("RAG_PORT", "8001")
    )

    yield
    logger.info("REST server shutting down")

app = FastAPI(title="retrieval-rest-server", lifespan=lifespan)

@app.exception_handler(ConfigurationError)
async def configuration_exception_handler(request: Request, exc: ConfigurationError):
    """Convert configuration errors into a stable JSON error response."""
    logger.error(f"Configuration error: {exc}")
    return JSONResponse(
        status_code=409,
        content={"error": "configuration_error", "message": str(exc)},
    )

@app.exception_handler(AuthenticationError)
async def authentication_exception_handler(request: Request, exc: AuthenticationError):
    """Convert authentication errors into a stable JSON error response."""
    logger.error(f"Authentication error: {exc}")
    return JSONResponse(
        status_code=401,
        content={"error": "authentication_error", "message": str(exc)},
    )

@app.exception_handler(ProviderError)
async def provider_exception_handler(request: Request, exc: ProviderError):
    """Convert upstream/provider errors into a stable JSON error response."""
    logger.error(f"Provider error: {exc}")
    return JSONResponse(
        status_code=502,
        content={"error": "provider_error", "message": str(exc)},
    )

# Allow cross-origin calls.
# Default is localhost-only; override via `RAG_CORS_ALLOW_ORIGINS` or settings.json.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_allow_origins(),
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
                if result is None:
                    raise RuntimeError("Backend returned None")
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
    status_code = 500
    error = None
    response = None
    token_count = None
    model = None

    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", 500)
    except Exception as e:
        status_code = 500
        error = str(e)
        raise
    finally:
        duration = time.perf_counter() - start
        duration_ms = int(duration * 1000)

        # Log to prometheus
        HTTP_INFLIGHT.dec()
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(method=method, path=path, status=status_code).observe(duration)
        _record_today_request(duration, path)

        # Log to performance_metrics table
        operation = None
        if path.endswith("/search"):
            operation = "search"
        elif path.endswith("/grounded_answer"):
            operation = "grounded_answer"
        elif path.endswith("/chat"):
            operation = "chat"
        elif path.endswith("/upsert_document"):
            operation = "upsert_document"

        if operation:
            try:
                pgvector_store.insert_performance_metric(
                    operation=operation,
                    duration_ms=duration_ms,
                    token_count=token_count,
                    model=model,
                    error=error,
                )
            except Exception as e:
                logger.error("Failed to insert performance metric: %s", e)

        # Log access in Apache combined log format style
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        query = str(request.url.query) if request.url.query else ''
        full_path = f"{request.url.path}?{query}" if query else request.url.path
        user_agent = request.headers.get('user-agent', '-')
        content_length = response.headers.get('content-length', '-') if response else '-'

        log_entry = (
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]} - '
            f'{client_ip} - - [{time.strftime("%d/%b/%Y:%H:%M:%S %z", time.localtime())}] '
            f'"{method} {full_path} HTTP/{request.scope.get("http_version", "1.1")}" '
            f'{status_code} {content_length} "-" "{user_agent}" {duration:.4f}s\n'
        )
        _append_access_log(log_entry)

    return response

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
        if result is None:
            logger.error("Backend.search returned None! Backend type: %s", type(backend))
            raise HTTPException(status_code=500, detail="Backend returned None")
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
    logger.info("Processing grounded answer request: %s", req.question)
    try:
        session_id = getattr(req, "session_id", None)
        if chat_store:
            if not session_id:
                try:
                    title = (req.question or "New Grounded Answer")[:50] + "..."
                    mode_str = None
                    try:
                        mode_str = str(backend.get_mode())
                    except Exception:  # pylint: disable=broad-exception-caught
                        mode_str = None
                    session_id = chat_store.create_session(
                        title=title,
                        metadata={"mode": mode_str, "kind": "grounded_answer"},
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to create grounded answer session: %s", exc)

            # Persist the user question as shown.
            if session_id:
                try:
                    chat_store.add_message(
                        session_id,
                        "user",
                        req.question,
                        display_content=req.question,
                        kind="user",
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to save grounded answer user message: %s", exc)

        # Build kwargs
        kwargs = {}
        if req.model:
            kwargs["model"] = req.model
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.config:
            kwargs.update(req.config)
             
        result = backend.grounded_answer(req.question, k=req.k or 5, **kwargs)
        if result is None:
            logger.error("Backend.grounded_answer returned None! Backend type: %s", type(backend))
            raise HTTPException(status_code=500, detail="Backend returned None")

        if chat_store and session_id:
            assistant_text = _extract_persistable_text(result)
            if assistant_text:
                try:
                    chat_store.add_message(
                        session_id,
                        "assistant",
                        assistant_text,
                        display_content=assistant_text,
                        sources=_extract_sources(result),
                        kind="assistant_grounded",
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to save grounded answer assistant message: %s", exc)

        # Include session_id in response for UI to attach or restore.
        result["session_id"] = session_id
        return result
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("grounded_answer failed: %s", exc)
        raise

@app.post(f"/{pth}/chat")
def api_chat(req: ChatReq):
    """Chat with the RAG backend, optionally maintaining history."""
    logger.info("Chat request received. Backend type: %s", type(backend))
    
    session_id = req.session_id
    user_message_id = None
    assistant_message_id = None
    if chat_store:
        # If client supplied a session_id (pre-generated), ensure the session exists.
        if session_id:
            try:
                existing = chat_store.get_session(session_id)
            except Exception:  # pylint: disable=broad-exception-caught
                existing = None
            if not existing:
                try:
                    last_msg = req.messages[-1] if req.messages else None
                    last_content = (
                        last_msg.display_content
                        if last_msg and getattr(last_msg, "display_content", None)
                        else (last_msg.content if last_msg else "New Chat")
                    )
                    title = last_content[:50] + "..."
                    mode_str = None
                    try:
                        mode_str = str(backend.get_mode())
                    except Exception:  # pylint: disable=broad-exception-caught
                        mode_str = None
                    chat_store.create_session(
                        title=title,
                        metadata={"mode": mode_str},
                        session_id=session_id,
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to ensure chat session exists: %s", e)

        # If new session needed
        if not session_id:
            try:
                # Use first few words of query as title
                # req.messages is List[ChatMessage]
                last_msg = req.messages[-1] if req.messages else None
                last_content = (
                    last_msg.display_content
                    if last_msg and getattr(last_msg, "display_content", None)
                    else (last_msg.content if last_msg else "New Chat")
                )
                title = last_content[:50] + "..."
                mode_str = None
                try:
                    mode_str = str(backend.get_mode())
                except Exception:  # pylint: disable=broad-exception-caught
                    mode_str = None
                session_id = chat_store.create_session(title=title, metadata={"mode": mode_str})
            except Exception as e:
                logger.error("Failed to create chat session: %s", e)
        
        # Save user message
        if req.messages:
            last_msg = req.messages[-1]
            try:
                user_message_id = chat_store.add_message(
                    session_id,
                    last_msg.role,
                    last_msg.content,
                    display_content=getattr(last_msg, "display_content", None),
                    kind="user" if last_msg.role == "user" else last_msg.role,
                )
            except Exception as e:
                logger.error("Failed to save user message: %s", e)

    try:
        # Build kwargs
        kwargs = {}
        if req.model:
            kwargs["model"] = req.model
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
            
        # Convert to backend-compatible dicts (avoid persistence-only fields like display_content).
        messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
        result = backend.chat(messages_dicts, **kwargs)
        
        if result is None:
             raise ProviderError("Backend returned None")
        
        # Save AI response (backend-agnostic, skip error payloads)
        if chat_store and session_id:
            assistant_text = _extract_persistable_text(result)
            if assistant_text:
                try:
                    assistant_message_id = chat_store.add_message(
                        session_id,
                        "assistant",
                        assistant_text,
                        display_content=assistant_text,
                        sources=_extract_sources(result),
                        kind="assistant",
                    )
                except Exception as e:
                    logger.error("Failed to save AI message: %s", e)
        
        # Include session_id in response
        result["session_id"] = session_id
        if user_message_id:
            result["user_message_id"] = user_message_id
        if assistant_message_id:
            result["assistant_message_id"] = assistant_message_id
        return result

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Chat failed: %s", exc)
        if "401" in str(exc) or "403" in str(exc) or "unauthorized" in str(exc).lower():
             raise AuthenticationError(f"Chat access denied: {exc}") from exc
        raise ProviderError(f"Chat failed: {exc}") from exc

@app.get(f"/{pth}/chat/history")
def api_chat_history_list(limit: int = 50, offset: int = 0):
    """List recent chat sessions."""
    if not chat_store:
        return []
    try:
        return chat_store.list_sessions(limit=limit, offset=offset)
    except Exception as e:
        logger.error("Failed to list chat history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{pth}/chat/history/{{session_id}}")
def api_chat_history_get(session_id: str):
    """Get messages for a specific session."""
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat storage usage not available")
    try:
        messages = chat_store.get_messages(session_id)
        if not messages:
             # Check if session exists at all
             session = chat_store.get_session(session_id)
             if not session:
                 raise HTTPException(status_code=404, detail="Session not found")
        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(f"/{pth}/chat/history/{{session_id}}/messages/{{message_id}}")
def api_chat_history_delete_message(session_id: str, message_id: str):
    """Delete a single message from a specific session."""
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat storage not available")
    try:
        deleted = chat_store.delete_message(session_id=session_id, message_id=message_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Message not found")
        return {"success": True, "session_id": session_id, "message_id": message_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete message: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"/{pth}/chat/history/{{session_id}}")
def api_chat_history_delete(session_id: str):
    """Delete a chat session and all its messages."""
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat storage not available")
    try:
        deleted = chat_store.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"success": True, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"/{pth}/chat/history/delete-all")
def api_chat_history_delete_all():
    """Delete all chat sessions. Use with caution!"""
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat storage not available")
    try:
        sessions = chat_store.list_sessions(limit=1000)
        deleted_count = 0
        for session in sessions:
            if chat_store.delete_session(session['id']):
                deleted_count += 1
        return {"success": True, "deleted_count": deleted_count}
    except Exception as e:
        logger.error("Failed to delete all sessions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

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
def api_documents_delete(req: DeleteDocsReq, _request: Request, _admin: None = Depends(require_admin_access)):
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
def api_set_mode(req: ConfigModeReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Set the backend mode (e.g., 'ollama', 'google')."""
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
def api_flush_cache(_request: Request, _admin: None = Depends(require_admin_access)):
    """Flush indexed content (pgvector + cache/indexed artifacts)."""
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


@app.get(f"/{pth}/metrics/performance")
def api_performance_metrics(hours: int = 24):
    """Return performance metrics from the last N hours."""
    try:
        return pgvector_store.get_performance_metrics(hours=hours)
    except Exception as exc:
        logger.error("Failed to get performance metrics: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(f"/{pth}/services")
def api_service_status(_request: Request, _admin: None = Depends(require_admin_access)):
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
            
            # Special handling for Ollama Cloud
            if service == "ollama":
                try:
                    mode = get_ollama_mode()
                    if mode == "cloud":
                        # In cloud mode, consider running if configured (API key present)
                        if get_ollama_api_key():
                            is_running = True
                    elif mode == "auto":
                        # In auto mode, running if local is running OR cloud is configured
                        if get_ollama_api_key():
                            is_running = True
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            
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
def api_start_service(service: str, _request: Request, _admin: None = Depends(require_admin_access)):
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
def api_stop_service(service: str, _request: Request, _admin: None = Depends(require_admin_access)):
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
def api_restart_service(service: str, _request: Request, _admin: None = Depends(require_admin_access)):
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
def auth_setup(_request: Request, _admin: None = Depends(require_admin_access)):
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
def auth_setup_post(
    request: Request,
    client_id: str = Form(...),
    client_secret: str = Form(...),
    _admin: None = Depends(require_admin_access),
):
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
                <head>
                    <title>Auth Success</title>
                    <style>
                        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; text-align: center; padding-top: 50px; background-color: #f5f5f7; }
                        h1 { color: #1d1d1f; }
                        p { color: #86868b; }
                    </style>
                </head>
                <body>
                    <h1 style="color: green;">Authentication Successful</h1>
                    <p>You have successfully connected your Google account.</p>
                    <p>This window will close automatically...</p>
                    <script>
                        // Notify opener if it exists
                        if (window.opener) {
                            window.opener.postMessage({ type: 'GOOGLE_AUTH_SUCCESS' }, '*');
                        }
                        // Close window after a short delay
                        setTimeout(() => {
                            window.close();
                        }, 2000);
                    </script>
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
def auth_logout(request: Request, provider: Optional[str] = None):
    """
    Log out by deleting the stored token.
    
    Args:
        provider: Optional provider to logout (google, openai_assistants). 
                  If omitted, logs out all.
    """
    try:
        logger.info("Logout requested (provider=%s)", provider)
        
        # Only logout auth_manager (Google) if provider is None or 'google'
        if not provider or provider == 'google':
            auth_manager.logout()
            
        logout_fn = getattr(backend, "logout", None)
        if callable(logout_fn):
            # Check if backend logout supports granular provider logout.
            sig = inspect.signature(logout_fn)
            if "provider" in sig.parameters:
                logout_kwargs = {}
                logout_kwargs["provider"] = provider
                logout_fn(**logout_kwargs)
            else:
                # Fallback for backends that don't support granular logout
                if not provider:
                    logout_fn()
                else:
                    logger.warning(
                        "Backend does not support granular logout for provider: %s",
                        provider,
                    )
                    
        logger.info("Logout successful")
        return JSONResponse({"status": "logged_out", "provider": provider})
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error logging out: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post(f"/{pth}/config/openai")
def save_openai_config(_request: Request, config: OpenAIConfigModel, _admin: None = Depends(require_admin_access)):
    """Save OpenAI API configuration to secrets/openai_config.json."""
    incoming_key = config.api_key
    existing_api_key = ""
    try:
        os.makedirs("secrets", exist_ok=True)

        existing_config: dict[str, Any] = {}
        if os.path.exists("secrets/openai_config.json"):
            try:
                with open("secrets/openai_config.json", "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                sanitized_err = _redact_error_message(str(exc), incoming_key)
                logger.warning("Failed to read existing OpenAI config: %s", sanitized_err)

        existing_api_key = existing_config.get("api_key", "")

        if incoming_key is None or incoming_key == MASKED_SECRET:
            if not existing_api_key:
                raise HTTPException(status_code=400, detail="API key is required for OpenAI configuration")
            api_key_to_save = existing_api_key
        else:
            api_key_to_save = incoming_key.strip() if incoming_key else ""
        
        config_data = {
            "api_key": api_key_to_save,
            "model": config.model,
            "assistant_id": config.assistant_id
        }
        with open("secrets/openai_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        
        try:
            os.chmod("secrets/openai_config.json", stat.S_IRUSR | stat.S_IWUSR)
        except OSError as perm_err:
            logger.warning("Failed to set permissions on OpenAI config: %s", perm_err)
        
        logger.info("OpenAI configuration saved to secrets/openai_config.json")
        return JSONResponse({"status": "success", "message": "Configuration saved"})
    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = _redact_error_message(str(e), incoming_key, existing_api_key)
        logger.error("Failed to save OpenAI config: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from e

@app.post(f"/{pth}/config/openai/reload")
def reload_openai_backend(_request: Request, _admin: None = Depends(require_admin_access)):
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
def get_openai_config(_request: Request, _admin: None = Depends(require_admin_access)):
    """Get OpenAI configuration (API key is masked; has_api_key indicates presence)."""
    existing_api_key: Optional[str] = None
    try:
        if not os.path.exists("secrets/openai_config.json"):
            masked_key, has_key = _mask_secret(None)
            return JSONResponse({
                "api_key": masked_key,
                "has_api_key": has_key,
                "model": "gpt-4-turbo-preview",
                "assistant_id": ""
            })
        
        with open("secrets/openai_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        existing_api_key = config.get("api_key", "")
        masked_key, has_key = _mask_secret(existing_api_key)
        return JSONResponse({
            "api_key": masked_key,
            "has_api_key": has_key,
            "model": config.get("model", "gpt-4-turbo-preview"),
            "assistant_id": config.get("assistant_id", "")
        })
    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = _redact_error_message(str(e), existing_api_key)
        logger.error("Failed to load OpenAI config: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from e


@app.post(f"/{pth}/config/pgvector")
def save_pgvector_config(_request: Request, config: PgvectorConfigModel, _admin: None = Depends(require_admin_access)):
    """Save pgvector configuration (password stored in secrets; other fields in settings.json)."""
    incoming_pw = config.password
    existing_pw: Optional[str] = None
    try:
        os.makedirs("secrets", exist_ok=True)

        # Load existing secret password if present
        secrets_path = Path("secrets/pgvector_config.json")
        existing_secret: Dict[str, Any] = {}
        if secrets_path.exists():
            try:
                existing_secret = json.loads(secrets_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to read existing pgvector secret config: %s", exc)
        existing_pw = existing_secret.get("password")

        if incoming_pw is None or incoming_pw == MASKED_SECRET:
            if not (existing_pw and str(existing_pw).strip()) and not (os.getenv("PGVECTOR_PASSWORD") or "").strip():
                raise HTTPException(status_code=400, detail="PGVECTOR password is required")
            pw_to_save = str(existing_pw) if (existing_pw and str(existing_pw).strip()) else None
        else:
            pw_to_save = incoming_pw.strip()

        if pw_to_save:
            secrets_path.write_text(json.dumps({"password": pw_to_save}, indent=2), encoding="utf-8")
            try:
                os.chmod(secrets_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError as perm_err:
                logger.warning("Failed to set permissions on pgvector secret config: %s", perm_err)

            os.environ["PGVECTOR_PASSWORD"] = pw_to_save

        # Persist non-secret settings in config/settings.json (merged)
        config_path = CONFIG_DIR / "settings.json"
        existing_settings: Dict[str, Any] = {}
        if config_path.exists():
            try:
                existing_settings = json.loads(config_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to read existing settings.json for merge: %s", exc)

        existing_settings.update(
            {
                "pgvectorHost": config.host,
                "pgvectorPort": config.port,
                "pgvectorDb": config.dbname,
                "pgvectorUser": config.user,
            }
        )
        config_path.write_text(json.dumps(existing_settings, indent=2), encoding="utf-8")

        # Update runtime env for immediate effect
        os.environ["PGVECTOR_HOST"] = config.host
        os.environ["PGVECTOR_PORT"] = str(config.port)
        os.environ["PGVECTOR_DB"] = config.dbname
        os.environ["PGVECTOR_USER"] = config.user

        logger.info("pgvector configuration saved")
        return JSONResponse({"status": "success", "message": "Configuration saved"})
    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = _redact_pgvector_error_message(str(e), incoming_pw, existing_pw)
        logger.error("Failed to save pgvector config: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from e


@app.get(f"/{pth}/config/pgvector")
def get_pgvector_config(_request: Request, _admin: None = Depends(require_admin_access)):
    """Get pgvector configuration (password masked; has_password indicates presence)."""
    existing_pw: Optional[str] = None
    try:
        config_path = CONFIG_DIR / "settings.json"
        settings: Dict[str, Any] = {}
        if config_path.exists():
            settings = json.loads(config_path.read_text(encoding="utf-8")) or {}

        host = os.getenv("PGVECTOR_HOST") or settings.get("pgvectorHost") or "127.0.0.1"
        port = int(os.getenv("PGVECTOR_PORT") or settings.get("pgvectorPort") or 5432)
        dbname = os.getenv("PGVECTOR_DB") or settings.get("pgvectorDb") or "agentic_rag"
        user = os.getenv("PGVECTOR_USER") or settings.get("pgvectorUser") or "agenticrag"

        env_pw = (os.getenv("PGVECTOR_PASSWORD") or "").strip()
        if env_pw:
            existing_pw = env_pw
        else:
            secrets_path = Path("secrets/pgvector_config.json")
            if secrets_path.exists():
                secret = json.loads(secrets_path.read_text(encoding="utf-8")) or {}
                existing_pw = secret.get("password")

        masked_pw, has_pw = _mask_secret(existing_pw)
        return JSONResponse(
            {
                "host": host,
                "port": port,
                "dbname": dbname,
                "user": user,
                "password": masked_pw,
                "has_password": has_pw,
            }
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = _redact_pgvector_error_message(str(e), existing_pw)
        logger.error("Failed to load pgvector config: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from e


@app.post(f"/{pth}/pgvector/test-connection")
def api_pgvector_test_connection(_request: Request, _admin: None = Depends(require_admin_access)):
    """Test pgvector/Postgres connectivity."""
    ok, msg = pgvector_store.test_connection()
    safe_msg = _redact_pgvector_error_message(msg, os.getenv("PGVECTOR_PASSWORD"))
    return JSONResponse({"success": bool(ok), "message": safe_msg})


@app.post(f"/{pth}/pgvector/migrate")
def api_pgvector_migrate(_request: Request, _admin: None = Depends(require_admin_access)):
    """Create pgvector schema + indexes (idempotent)."""
    try:
        result = rag_core.ensure_vector_store_ready()
        return JSONResponse(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        err = _redact_pgvector_error_message(str(exc), os.getenv("PGVECTOR_PASSWORD"))
        return JSONResponse({"status": "error", "error": err}, status_code=500)


@app.post(f"/{pth}/pgvector/backfill")
def api_pgvector_backfill(_request: Request, _admin: None = Depends(require_admin_access)):
    """Rebuild pgvector vectors from indexed artifacts."""
    try:
        result = rag_core.rebuild_index()
        return JSONResponse(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        err = _redact_pgvector_error_message(str(exc), os.getenv("PGVECTOR_PASSWORD"))
        return JSONResponse({"status": "error", "error": err}, status_code=500)


@app.get(f"/{pth}/pgvector/stats")
def api_pgvector_stats(_request: Request, _admin: None = Depends(require_admin_access)):
    """Return vector store stats for UI."""
    try:
        stats = pgvector_store.stats(embedding_model=rag_core.EMBED_MODEL_NAME)
        return JSONResponse({"status": "ok", **stats})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        err = _redact_pgvector_error_message(str(exc), os.getenv("PGVECTOR_PASSWORD"))
        return JSONResponse({"status": "error", "error": err}, status_code=500)

@app.get(f"/{pth}/config/openai/models")
def get_openai_models(_request: Request, _admin: None = Depends(require_admin_access)):
    """Get list of available OpenAI models using the configured API key."""
    api_key = ""
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
        safe_err = _redact_error_message(str(e), api_key)
        logger.error("Failed to fetch OpenAI models: %s", safe_err)
        raise HTTPException(status_code=500, detail=safe_err) from e

@app.post(f"/{pth}/config/openai/models")
def get_openai_models_post(req: OpenAIModelsReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Get list of available OpenAI models using the provided API key."""
    try:
        api_key = req.api_key
        # If masked, try to use stored key
        if api_key == MASKED_SECRET:
            if not os.path.exists("secrets/openai_config.json"):
                raise HTTPException(status_code=400, detail="OpenAI not configured and key masked")
            
            with open("secrets/openai_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            api_key = config.get("api_key", "")
            
        if not api_key or not api_key.strip():
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Fetch models from OpenAI API
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models_response = client.models.list()
        
        all_model_ids = [m.id for m in models_response.data]
        
        # Filter for models suitable for assistants (GPT, o1, and text models)
        suitable_models = []
        for model in models_response.data:
            model_id = model.id
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
                "message": f"Your OpenAI account only has access to: {', '.join(all_model_ids)}. The Assistants API requires GPT-4, GPT-3.5-turbo, or o1 models.",
                "available_models": all_model_ids
            })
        
        # Sort by creation date (newest first)
        suitable_models.sort(key=lambda x: x["created"], reverse=True)
        
        logger.info("Retrieved %d OpenAI models (from %d total) using provided key", len(suitable_models), len(models_response.data))
        return JSONResponse({"models": suitable_models})
        
    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        safe_err = _redact_error_message(str(e), api_key)
        logger.error("Failed to fetch OpenAI models (POST): %s", safe_err)
        raise HTTPException(status_code=500, detail=safe_err) from e

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

@app.get(f"/{pth}/config/autotune")
async def get_autotune_params(query: str, default_k: int = 12):
    """Peek at what parameters would be used for a given query."""
    top_k, max_context = rag_core._autotune_rag_params(query, default_k)
    return {
        "query": query,
        "predicted_k": top_k,
        "predicted_context": max_context,
        "mode": "Technical/Fact" if top_k <= 3 else ("Complex/Research" if top_k >= 8 else "Balanced")
    }

@app.post(f"/{pth}/config/app")
async def update_app_config(req: AppConfigReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Save application configuration."""
    config_dir = CONFIG_DIR
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "settings.json"

    try:
        payload = req.model_dump(by_alias=True)
        required_fields = ("apiEndpoint", "model")

        def _has_value(field_name: str) -> bool:
            value = payload.get(field_name)
            if isinstance(value, str):
                return bool(value.strip())
            return bool(value)

        # Treat Ollama as configured if we have a model and either:
        # - a local endpoint, or
        # - a stored Ollama Cloud API key (cloud-only users)
        has_model = _has_value("model")
        has_local_endpoint = _has_value("apiEndpoint")
        try:
            from src.core import ollama_config

            cloud_api_key = ollama_config.get_ollama_api_key()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            cloud_api_key = None
            logger.warning("Unable to read Ollama cloud config when saving app config: %s", exc)

        ollama_configured = has_model and (has_local_endpoint or bool(cloud_api_key))
        payload["ollamaConfigured"] = ollama_configured
        if ollama_configured:
            payload["ragMode"] = payload.get("ragMode", "ollama")
        else:
            payload["ragMode"] = "none"

        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to read existing app config for merge: %s", exc)

        existing.update(payload)
        with open(config_path, "w", encoding="utf-8") as f:
            # Preserve unknown keys (e.g., pgvector config) by merging.
            json.dump(existing, f, indent=2)
        
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

        try:
            backend.set_ollama_configured(ollama_configured)
        except Exception as backend_err:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to update backend Ollama configuration state: %s", backend_err)
        
        # Reload backends to pick up CA bundle changes
        if hasattr(backend, "reload_backend"):
            try:
                backend.reload_backend("google")
                backend.reload_backend("openai_assistants")
                logger.info("Reloaded Google and OpenAI backends to apply new configuration")
            except Exception as reload_backend_err:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to reload backends: %s", reload_backend_err)

        return {"status": "saved"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save app config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/config/app")
def api_get_app_config(_request: Request, _admin: None = Depends(require_admin_access)):
    """Get application configuration."""
    config_path = CONFIG_DIR / "settings.json"
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
def api_save_vertex_config(req: VertexConfigReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Save Vertex AI configuration."""
    config = {
        "VERTEX_PROJECT_ID": req.project_id,
        "VERTEX_LOCATION": req.location,
        "VERTEX_DATA_STORE_ID": req.data_store_id
    }
    # Save to file
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(VERTEX_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # Update current process env vars so it works immediately
        os.environ.update(config)
        
        return {"status": "saved"}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save vertex config: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/config/vertex")
def api_get_vertex_config(_request: Request, _admin: None = Depends(require_admin_access)):
    """Get Vertex AI configuration."""
    try:
        if VERTEX_CONFIG_PATH.exists():
            with open(VERTEX_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "VERTEX_PROJECT_ID": os.getenv("VERTEX_PROJECT_ID", ""),
            "VERTEX_LOCATION": os.getenv("VERTEX_LOCATION", "us-central1"),
            "VERTEX_DATA_STORE_ID": os.getenv("VERTEX_DATA_STORE_ID", "")
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to read vertex config: %s", e)
        return {}

# --- Ollama Cloud Configuration Endpoints ---

@app.get(f"/{pth}/ollama/mode")
def api_get_ollama_mode():
    """Get current Ollama mode (local/cloud/auto)."""
    try:
        from src.core.ollama_config import get_ollama_mode
        mode = get_ollama_mode()
        return {"mode": mode}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to get Ollama mode: %s", e)
        return {"mode": "local"}  # Default fallback

@app.post(f"/{pth}/ollama/mode")
def api_set_ollama_mode(req: OllamaModeReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Set Ollama mode (local/cloud/auto)."""
    mode = req.mode.strip().lower()
    if mode not in ["local", "cloud", "auto"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Must be 'local', 'cloud', or 'auto'"
        )
    
    try:
        # Read current settings
        config_path = CONFIG_DIR / "settings.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        # Update mode
        config["ollamaMode"] = mode
        
        # Save settings
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        # Reload configuration
        try:
            from src.core import rag_core
            rag_core.reload_settings()
            logger.info("Updated Ollama mode to %s and reloaded configuration", mode)
        except Exception as reload_err:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to reload rag_core: %s", reload_err)
        
        return {"status": "ok", "mode": mode}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to set Ollama mode: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/ollama/status")
def api_get_ollama_status():
    """Get Ollama connection status for both local and cloud."""
    try:
        from src.core.ollama_config import (
            get_ollama_mode,
            get_ollama_endpoint,
            get_ollama_local_endpoint,
            get_ollama_cloud_endpoint,
            test_cloud_connection,
        )
        import requests
        
        mode = get_ollama_mode()
        endpoint = get_ollama_endpoint()
        
        # Test local connection
        local_endpoint = get_ollama_local_endpoint()
        local_available = False
        local_status = None
        try:
            response = requests.get(
                f"{local_endpoint}/api/tags",
                timeout=2,
                proxies={"http": "", "https": ""},  # bypass system proxies for localhost
            )
            if response.ok:
                local_available = True
                local_status = "connected"
            else:
                local_status = "error"
        except Exception:  # pylint: disable=broad-exception-caught
            local_status = "disconnected"
        
        # Test cloud connection (only if mode is cloud or auto)
        cloud_available = False
        cloud_status = None
        if mode in ["cloud", "auto"]:
            try:
                success, _ = test_cloud_connection()
                cloud_available = success
                cloud_status = "connected" if success else "error"
            except Exception:  # pylint: disable=broad-exception-caught
                cloud_status = "disconnected"
        
        return OllamaStatusResp(
            mode=mode,
            endpoint=endpoint,
            cloud_available=cloud_available,
            local_available=local_available,
            cloud_status=cloud_status,
            local_status=local_status
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to get Ollama status: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/ollama/models")
def api_list_ollama_models_get():
    """List available models from current Ollama endpoint (configured)."""
    try:
        backend_instance = get_rag_backend()
        if hasattr(backend_instance, 'list_models'):
            models = backend_instance.list_models()
            return {"models": models}
        return {"models": []}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to list Ollama models: %s", e)
        return {"models": []}

@app.post(f"/{pth}/ollama/models")
def api_list_ollama_models_post(req: OllamaModelsReq):
    """List available models from Ollama endpoint with provided credentials."""
    try:
        from src.core.ollama_config import (
            get_ollama_endpoint,
            get_ollama_client_headers,
            get_requests_ca_bundle,
            get_ollama_cloud_proxy,
            get_ollama_api_key
        )
        import requests

        # Determine credentials to use (provided > configured)
        api_key = req.api_key if req.api_key else get_ollama_api_key()
        if req.api_key == MASKED_SECRET:
             api_key = get_ollama_api_key()

        endpoint = req.endpoint if req.endpoint else get_ollama_endpoint()
        proxy = req.proxy if req.proxy else get_ollama_cloud_proxy()
        ca_bundle = req.ca_bundle if req.ca_bundle else get_requests_ca_bundle()

        # Build headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Build proxies dict
        proxies = None
        if proxy:
            proxies = {"http": proxy, "https": proxy}

        # Build verify arg
        verify = True
        if ca_bundle and os.path.exists(ca_bundle):
            verify = ca_bundle

        # Make request
        # Convert http to https if cloud
        if endpoint.startswith("http://") and "ollama.com" in endpoint:
            endpoint = endpoint.replace("http://", "https://")

        resp = requests.get(
            f"{endpoint.rstrip('/')}/api/tags",
            headers=headers,
            timeout=10,
            verify=verify,
            proxies=proxies
        )
        
        if resp.ok:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return {"models": models}
        
        logger.warning(f"Failed to fetch models: {resp.status_code} {resp.text}")
        return {"models": []}

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to list Ollama models (POST): %s", e)
        return {"models": []}

@app.post(f"/{pth}/ollama/test-connection")
def api_test_ollama_connection(req: OllamaTestConnectionReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Test connection to Ollama Cloud with API key."""
    try:
        from src.core.ollama_config import (
            test_cloud_connection,
            save_ollama_cloud_config,
            save_ollama_cloud_proxy,
        )
        request_api_key = None if req.api_key == MASKED_SECRET else req.api_key
        
        # Test connection
        success, message = test_cloud_connection(
            api_key=request_api_key,
            endpoint=req.endpoint,
            ca_bundle=req.ca_bundle
        )
        
        # If test successful, persist provided values
        if success:
            try:
                if request_api_key:
                    save_ollama_cloud_config(
                        api_key=request_api_key,
                        endpoint=req.endpoint,
                        ca_bundle=req.ca_bundle
                    )
                    message += " (API key saved)"
                elif req.ca_bundle is not None:
                    # Save CA bundle even if API key not provided
                    save_ollama_cloud_config(
                        ca_bundle=req.ca_bundle,
                        endpoint=req.endpoint
                    )
                if req.proxy is not None:
                    save_ollama_cloud_proxy(req.proxy)
            except Exception as save_err:  # pylint: disable=broad-exception-caught
                logger.warning("Connection test succeeded but failed to save config: %s", save_err)
                # Don't fail the request if save fails
        
        return OllamaTestConnectionResp(success=success, message=message)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to test Ollama connection: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get(f"/{pth}/ollama/cloud-config")
def api_get_ollama_cloud_config(_request: Request, _admin: None = Depends(require_admin_access)):
    """Fetch stored Ollama Cloud configuration (API key is masked)."""
    api_key_raw: Optional[str] = None
    try:
        from src.core.ollama_config import (
            get_ollama_api_key,
            get_ollama_cloud_endpoint,
            get_configured_ca_bundle,
            get_ollama_cloud_proxy,
        )
        api_key_raw = get_ollama_api_key()
        masked_key, has_key = _mask_secret(api_key_raw)
        return {
            "api_key": masked_key,
            "has_api_key": has_key,
            "endpoint": get_ollama_cloud_endpoint(),
            "ca_bundle": get_configured_ca_bundle() or "",
            "proxy": get_ollama_cloud_proxy() or "",
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_msg = _redact_error_message(str(exc), api_key_raw)
        logger.error("Failed to read Ollama cloud config: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from exc

@app.post(f"/{pth}/ollama/cloud-config")
def api_save_ollama_cloud_config(req: OllamaCloudConfigReq, _request: Request, _admin: None = Depends(require_admin_access)):
    """Persist Ollama Cloud secrets/config without forcing a live test."""
    api_key: Optional[str] = None
    try:
        from src.core.ollama_config import (
            save_ollama_cloud_config,
            save_ollama_cloud_proxy,
        )

        api_key = None if req.api_key == MASKED_SECRET else req.api_key

        if any([api_key is not None, req.endpoint is not None, req.ca_bundle is not None]):
            save_ollama_cloud_config(
                api_key=api_key,
                endpoint=req.endpoint,
                ca_bundle=req.ca_bundle,
            )

        if req.proxy is not None:
            save_ollama_cloud_proxy(req.proxy)

        return {"status": "saved"}
    except ValueError as exc:
        error_msg = _redact_error_message(str(exc), api_key, req.api_key)
        raise HTTPException(status_code=400, detail=error_msg) from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_msg = _redact_error_message(str(exc), api_key, req.api_key)
        logger.error("Failed to save Ollama cloud config: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from exc

@app.get(f"/{pth}/logs/{{log_type}}")
def api_get_logs(log_type: str, _request: Request, lines: int = 500, _admin: None = Depends(require_admin_access)):
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
async def api_stream_logs(log_type: str, request: Request, _admin: None = Depends(require_admin_access)):
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
