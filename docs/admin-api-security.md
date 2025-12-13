# Admin API security

This project exposes a mix of end-user endpoints (chat/search/index) and **administrative** endpoints (configuration, secrets, service control, logs).

The REST server is designed to be safe by default for the common "everything runs locally" case:
- Requests originating from **loopback** (e.g. `127.0.0.1`, `::1`) can access admin endpoints **without authentication** (default mode).
- Requests from **non-loopback** addresses require an admin token, and **HTTPS is required by default**.

## How admin access is enforced
Admin endpoints use the `require_admin_access` dependency in [src/servers/rest_server.py](../src/servers/rest_server.py).

### Defaults
- `RAG_ADMIN_AUTH_MODE=nonlocal`
  - Loopback requests: allowed without auth.
  - Non-loopback requests: require token (and HTTPS by default).
- `RAG_ADMIN_REQUIRE_HTTPS_NONLOCAL=true`
  - Enforces HTTPS for non-loopback requests.

### Token
Set an admin token for remote admin access:
- Env var: `RAG_ADMIN_TOKEN`

Send it with either:
- `Authorization: Bearer <token>`
- `X-RAG-Admin-Token: <token>`

### Proxy / load balancer deployments
If you terminate TLS at a reverse proxy (so the FastAPI app itself sees `http://`), enable proxy header trust:
- `RAG_ADMIN_TRUST_PROXY=true`

With this enabled, the server will honor:
- `X-Forwarded-For` / `X-Real-IP` (client IP for loopback detection)
- `X-Forwarded-Proto` (to detect `https`)

Only enable `RAG_ADMIN_TRUST_PROXY` when you control the proxy and it strips/overwrites these headers.

## CORS defaults
CORS is **localhost-only by default** to reduce CSRF-style risk against local admin endpoints.

Override via:
- Env var: `RAG_CORS_ALLOW_ORIGINS` (comma-separated)
- `config/settings.json`: `corsAllowOrigins` (list or comma-separated string)

## Admin endpoint surface
The following endpoint groups are intended to be treated as admin-only:
- Config + secrets management:
  - `/api/config/openai/*`
  - `/api/config/pgvector`
  - `/api/config/vertex`
  - `/api/config/app`
  - `/api/ollama/cloud-config`
- Service control + diagnostics:
  - `/api/services/*`
  - `/api/logs/*`
  - `/api/flush_cache`
  - `/api/config/mode` (POST)
- pgvector maintenance:
  - `/api/pgvector/*`
- Google OAuth setup UI:
  - `/api/auth/setup` (GET/POST)

Notes:
- The base prefix `/api` can be changed via `RAG_PATH`.
- If you run everything on localhost (UI + REST), the defaults allow admin actions without requiring any token.
