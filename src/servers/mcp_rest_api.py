"""Deprecated legacy MCP REST shim.

This module previously exposed a permissive REST surface (including destructive
operations) intended as a compatibility layer. It is no longer used by the
runtime servers and is intentionally kept inert to avoid accidental exposure.

Use the primary REST server ([src/servers/rest_server.py](src/servers/rest_server.py))
or the MCP server endpoints instead.
"""

from fastapi import FastAPI
from starlette.responses import JSONResponse


rest_api = FastAPI(title="mcp-rest-shim (deprecated)")


@rest_api.get("/health")
async def rest_health():
    """Return a deprecation notice."""
    return JSONResponse(
        {
            "status": "deprecated",
            "detail": "This legacy REST shim is disabled. Use the main REST API on port 8001.",
        },
        status_code=410,
    )
