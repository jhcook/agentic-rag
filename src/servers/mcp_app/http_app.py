import os
from starlette.routing import Mount
from starlette.applications import Starlette
from mcp.server.fastmcp import FastMCP as _FastMCP

from src.servers.mcp_app.api import rest_api
from src.servers.mcp_app.logging_config import AccessLogMiddleware, patch_uvicorn_access_logging

def patch_streamable_http(access_logger=None):
    """Monkey-patch FastMCP to disable uvicorn access logging and mount REST routes."""
    patch_uvicorn_access_logging()

    # Disable uvicorn access log at serve time
    async def _run_streamable_http_no_access(self):
        import uvicorn

        starlette_app = self.streamable_http_app()
        config = uvicorn.Config(
            starlette_app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
            access_log=False,
            log_config=getattr(uvicorn.config, "LOGGING_CONFIG", None),
        )
        server = uvicorn.Server(config)
        await server.serve()

    _FastMCP.run_streamable_http_async = _run_streamable_http_no_access

    _orig_streamable_http_app = _FastMCP.streamable_http_app

    def _streamable_http_app_with_rest(self):
        base_app = _orig_streamable_http_app(self)
        
        # Mount REST API directly onto base_app
        base_app.mount("/rest", rest_api)
        
        prefix = getattr(self.settings, "streamable_http_path", "/")
        if prefix and prefix != "/":
            if not prefix.startswith("/"):
                prefix = f"/{prefix}"
            if prefix.endswith("/"):
                prefix = prefix[:-1]
            base_app.mount(f"{prefix}/rest", rest_api)
        
        return base_app

    _FastMCP.streamable_http_app = _streamable_http_app_with_rest
