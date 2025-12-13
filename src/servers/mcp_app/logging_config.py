import logging
import os
from datetime import datetime
import time
from pathlib import Path
import warnings
from starlette.middleware.base import BaseHTTPMiddleware


class ShutdownFilter(logging.Filter):
    """Filter out noisy shutdown errors."""
    def filter(self, record):
        # Filter out CancelledError which happens during graceful shutdown
        if record.exc_info:
            exc_type, _, _ = record.exc_info
            if exc_type and "CancelledError" in exc_type.__name__:
                return False
        # Also filter out the generic message if it mentions CancelledError
        if "CancelledError" in record.getMessage():
            return False
        return True


def update_logging_level(debug_mode: bool):
    """
    Dynamically update logging levels for all loggers based on debug mode.
    
    Args:
        debug_mode: If True, set log level to DEBUG; otherwise INFO.
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Update root logger and all relevant module loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Update all src.* loggers
    logging.getLogger("src").setLevel(log_level)
    logging.getLogger("src.core").setLevel(log_level)
    logging.getLogger("src.core.rag_core").setLevel(log_level)
    logging.getLogger("src.core.factory").setLevel(log_level)
    logging.getLogger("src.core.pgvector_store").setLevel(log_level)
    logging.getLogger("src.core.indexer").setLevel(log_level)
    logging.getLogger("src.core.extractors").setLevel(log_level)
    logging.getLogger("src.servers").setLevel(log_level)
    logging.getLogger("src.servers.mcp_server").setLevel(log_level)
    logging.getLogger("src.servers.mcp_app").setLevel(log_level)
    
    # Update environment variable
    os.environ["RAG_DEBUG_MODE"] = "true" if debug_mode else "false"
    
    logger = logging.getLogger(__name__)
    logger.info("MCP logging level updated to %s (debug_mode=%s)", 
               logging.getLevelName(log_level), debug_mode)


def configure_logging():
    """Configure process and access loggers."""
    os.makedirs("log", exist_ok=True)
    
    # Read debug mode from settings.json if available
    config_file = Path(__file__).resolve().parent.parent.parent / "config" / "settings.json"
    debug_mode = False
    if config_file.exists():
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                debug_mode = bool(config.get("debugMode", False))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    
    initial_level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=initial_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("log/mcp_server.log"),
            logging.StreamHandler(),
        ],
    )

    # Route warnings through logging so they also include timestamps.
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    _warnings_logger = logging.getLogger("py.warnings")
    _warnings_logger.handlers.clear()
    _warnings_logger.propagate = True
    logger = logging.getLogger(__name__)
    logger.setLevel(initial_level)
    
    # Apply debug mode to all loggers
    if debug_mode:
        update_logging_level(debug_mode)

    # Add shutdown filter to suppress CancelledError noise
    shutdown_filter = ShutdownFilter()
    logging.getLogger("uvicorn.error").addFilter(shutdown_filter)
    logging.getLogger("uvicorn").addFilter(shutdown_filter)
    # Also add to root logger just in case
    logging.getLogger().addFilter(shutdown_filter)

    access_logger = logging.getLogger("mcp_access")
    access_logger.setLevel(logging.INFO)
    access_logger.handlers.clear()
    access_logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler("log/mcp_access.log")
    file_handler.setFormatter(formatter)
    access_logger.addHandler(file_handler)

    return logger, access_logger


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP access for MCP server."""

    def __init__(self, app, access_logger):
        super().__init__(app)
        self.access_logger = access_logger

    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        self.access_logger.info(
            "%s %s %s %s %dms",
            request.method,
            request.url.path,
            request.client.host if request.client else "-",
            response.status_code,
            duration_ms,
        )
        return response


class AccessLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to emit Apache-style access logs to a dedicated file."""

    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        client_ip = getattr(request.client, "host", "unknown") if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""
        if query:
            path = f"{path}?{query}"
        status = response.status_code
        user_agent = request.headers.get("user-agent", "-")
        content_length = response.headers.get("content-length", "-")
        log_entry = (
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]} - '
            f"{client_ip} - - [{time.strftime('%d/%b/%Y:%H:%M:%S %z', time.localtime(start_time))}] "
            f'"{method} {path} HTTP/{request.scope.get("http_version", "1.1")}" '
            f"{status} {content_length} \"-\" \"{user_agent}\" {duration:.4f}s\n"
        )
        with open("log/mcp_server_access.log", "a") as handle:
            handle.write(log_entry)
        return response


def patch_uvicorn_access_logging():
    """Disable uvicorn access logging (handled by our middleware instead)."""
    try:
        import uvicorn
        from copy import deepcopy

        patched = deepcopy(getattr(uvicorn.config, "LOGGING_CONFIG", {}))
        if patched and "loggers" in patched and "uvicorn.access" in patched["loggers"]:
            patched["loggers"]["uvicorn.access"]["handlers"] = []
            patched["loggers"]["uvicorn.access"]["propagate"] = False
            patched["loggers"]["uvicorn.access"]["level"] = "WARNING"
            uvicorn.config.LOGGING_CONFIG = patched
        os.environ.setdefault("UVICORN_ACCESS_LOG", "false")
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).debug("Unable to patch uvicorn access logging: %s", exc)
