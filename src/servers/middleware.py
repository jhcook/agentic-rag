from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

class AccessLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP access for MCP server."""
    async def dispatch(self, request, call_next):
        import time
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ''
        if query:
            path = f"{path}?{query}"
        status = response.status_code
        user_agent = request.headers.get('user-agent', '-')
        content_length = response.headers.get('content-length', '-')
        log_entry = (
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]} - '
            f'{client_ip} - - [{time.strftime("%d/%b/%Y:%H:%M:%S %z", time.localtime(start_time))}] '
            f'"{method} {path} HTTP/{request.scope.get("http_version", "1.1")}" '
            f'{status} {content_length} "-" "{user_agent}" {duration:.4f}s\n'
        )
        with open('log/mcp_server_access.log', 'a') as f:
            f.write(log_entry)
        return response
