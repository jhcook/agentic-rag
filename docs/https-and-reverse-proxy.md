# HTTPS and reverse proxy deployment

For non-local deployments, you should place the REST server behind TLS.

By default, admin endpoints enforce:
- token authentication for non-loopback requests
- **HTTPS required for non-loopback requests**

These checks are implemented in [src/servers/rest_server.py](../src/servers/rest_server.py).

## Recommended deployment pattern
1. Run the FastAPI REST server on an internal interface (e.g. `127.0.0.1:8001` or a private network).
2. Terminate TLS at a reverse proxy (nginx/Caddy/Traefik/ALB/etc.).
3. Configure the proxy to:
   - set `X-Forwarded-Proto: https`
   - set `X-Forwarded-For` or `X-Real-IP` to the client IP
   - strip any incoming `X-Forwarded-*` headers from the public internet and replace them

Then set:
- `RAG_ADMIN_TRUST_PROXY=true`

This allows the server to correctly treat proxied requests as HTTPS and to correctly classify the client IP.

## Local-only development
If UI + REST are both running locally, admin endpoints are accessible without auth by default:
- `RAG_ADMIN_AUTH_MODE=nonlocal`

In this mode, loopback requests are treated as trusted for admin operations.

## Configuration summary
- `RAG_ADMIN_AUTH_MODE`
  - `nonlocal` (default): loopback has no auth; remote requires token.
  - `off`/`disabled`/`false`: disables admin checks (not recommended for non-local use).
- `RAG_ADMIN_REQUIRE_HTTPS_NONLOCAL`
  - `true` (default): require HTTPS for remote admin.
- `RAG_ADMIN_TRUST_PROXY`
  - `true`: trust `X-Forwarded-Proto` and `X-Forwarded-For`/`X-Real-IP`.

## Troubleshooting
- If you see `HTTPS is required for remote admin endpoints` behind a proxy:
  - ensure the proxy sets `X-Forwarded-Proto: https`
  - ensure `RAG_ADMIN_TRUST_PROXY=true`
- If loopback requests are unexpectedly treated as non-loopback behind a proxy:
  - ensure the proxy sets `X-Forwarded-For` (or `X-Real-IP`) correctly
  - ensure untrusted clients cannot spoof these headers
