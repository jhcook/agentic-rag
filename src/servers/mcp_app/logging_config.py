import logging
import os
from datetime import datetime
import time
from starlette.middleware.base import BaseHTTPMiddleware


def configure_logging():
    """Configure process and access loggers."""
    os.makedirs("log", exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("log/mcp_server.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

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
