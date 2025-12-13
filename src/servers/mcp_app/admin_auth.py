"""Admin auth helpers for the MCP server REST surface.

The MCP server mounts a small REST API under `/rest` for convenience. Some
endpoints are destructive (e.g., deleting documents) and must be protected when
exposed beyond localhost.

Policy:
- Loopback requests are allowed without auth (developer-friendly default).
- Non-loopback requests require a bearer token in `Authorization: Bearer ...`
  or `X-RAG-Admin-Token`.
- If `RAG_ADMIN_AUTH_MODE` is set to "off", checks are disabled.

This mirrors the intent of the REST server's admin gate but is intentionally
self-contained to avoid cross-importing the large REST server module.
"""

from __future__ import annotations

import hmac
import json
import os
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException, Request


@lru_cache(maxsize=1)
def _load_settings_json() -> Dict[str, Any]:
    settings_path = Path(__file__).resolve().parents[3] / "config" / "settings.json"
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        # Fail closed: if settings are unreadable, rely on env-only behavior.
        return {}


def _get_str_setting(env_key: str, json_key: str, default: str) -> str:
    value = os.getenv(env_key)
    if value is not None and value.strip() != "":
        return value
    settings = _load_settings_json()
    json_value = settings.get(json_key)
    if isinstance(json_value, str) and json_value.strip() != "":
        return json_value
    return default


def _is_loopback_request(request: Request) -> bool:
    host = request.client.host if request.client else ""
    if host in {"testclient", "testserver"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def require_admin_access(request: Request) -> None:
    """Enforce admin access for sensitive MCP REST endpoints."""

    mode = _get_str_setting("RAG_ADMIN_AUTH_MODE", "adminAuthMode", "nonlocal").strip().lower()
    if mode in {"off", "disabled", "0", "false"}:
        return

    is_loopback = _is_loopback_request(request)
    if mode == "nonlocal" and is_loopback:
        return

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
