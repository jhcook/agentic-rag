# Ollama Cloud Architecture Documentation

**Last Updated**: 2025-01-27  
**Status**: Active

---

## 1. Architecture Overview

The Ollama Cloud integration extends the existing agentic-rag system to support cloud-hosted LLM models alongside local Ollama instances. The architecture maintains a unified interface while supporting three operational modes: **local**, **cloud**, and **auto**.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (UI, REST API, MCP Server, CLI)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Layer                        │
│  (ollama_config.py)                                         │
│  - Mode selection (local/cloud/auto)                        │
│  - Endpoint resolution                                      │
│  - API key management                                       │
│  - Fallback logic                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                              ▼
┌───────────────┐            ┌──────────────────┐
│  Local Mode   │            │   Cloud Mode     │
│               │            │                  │
│  Ollama       │            │  Ollama Cloud    │
│  (localhost)  │            │  (https://...)   │
│               │            │                  │
│  No Auth      │            │  API Key Auth    │
│  No Network   │            │  HTTPS/TLS       │
└───────────────┘            └──────────────────┘
```

---

## 2. Trust Boundaries

### Local Mode (Trusted)
- **Boundary**: Application ↔ Local Ollama (localhost)
- **Trust Level**: **HIGH** - No network, local process
- **Security**: 
  - No authentication required
  - No encryption required (localhost)
  - No third-party involvement
- **Data Flow**: All data stays on local machine

### Cloud Mode (Untrusted Third-Party)
- **Boundary**: Application ↔ Ollama Cloud (Internet)
- **Trust Level**: **MEDIUM** - Third-party service, network transmission
- **Security**:
  - API key authentication (Bearer token)
  - HTTPS/TLS encryption (enforced)
  - SSRF protection (endpoint validation)
  - No local data storage by third party (assumed)
- **Data Flow**: User queries and document content sent to Ollama Cloud

### Auto Mode (Hybrid)
- **Boundary**: Application ↔ Ollama Cloud (primary) → Local Ollama (fallback)
- **Trust Level**: **MEDIUM** - Prefers cloud, falls back to local
- **Security**: Same as cloud mode, with automatic fallback
- **Data Flow**: Tries cloud first, falls back to local on failure

---

## 3. Data Flow Diagrams

### Local Mode Data Flow

```
User Query
    │
    ▼
Application (rag_core.py)
    │
    ▼
Local Ollama (http://127.0.0.1:11434)
    │
    ▼
Response
    │
    ▼
User
```

**Characteristics**:
- No network transmission
- No authentication
- All data stays local

### Cloud Mode Data Flow

```
User Query + Document Context
    │
    ▼
Application (rag_core.py)
    │
    ├─► Add API Key Header (Bearer token)
    │
    ▼
HTTPS Request (TLS encrypted)
    │
    ▼
Ollama Cloud (https://ollama.com)
    │
    ├─► Process with LLM
    │
    ▼
HTTPS Response (TLS encrypted)
    │
    ▼
Application
    │
    ▼
User
```

**Characteristics**:
- Network transmission over HTTPS
- API key authentication
- Data sent to third party
- Response returned to user

### Auto Mode Data Flow (with Fallback)

```
User Query
    │
    ▼
Application
    │
    ├─► Try Cloud Endpoint
    │   │
    │   ├─► Success → Return Response
    │   │
    │   └─► Failure (timeout/connection error)
    │       │
    │       ▼
    │   Fallback to Local
    │       │
    │       ▼
    │   Local Ollama
    │       │
    │       ▼
    └─► Return Response
```

**Characteristics**:
- Tries cloud first
- Automatic fallback on failure
- Logs fallback events
- User may not be aware of fallback (transparent)

---

## 4. Configuration Flow

### Configuration Sources (Priority Order)

1. **settings.json** (`config/settings.json`)
   - `ollamaMode`: "local" | "cloud" | "auto"
   - `ollamaCloudEndpoint`: Cloud endpoint URL
   - `apiEndpoint`: Local endpoint URL

2. **Secrets File** (`secrets/ollama_cloud_config.json`)
   - `api_key`: Ollama Cloud API key
   - `endpoint`: Cloud endpoint (optional)

3. **Environment Variables**
   - `OLLAMA_MODE`: Mode selection
   - `OLLAMA_CLOUD_API_KEY`: API key
   - `OLLAMA_CLOUD_ENDPOINT`: Cloud endpoint
   - `OLLAMA_API_BASE`: Local endpoint

### Configuration Resolution

```
get_ollama_mode()
    │
    ├─► settings.json["ollamaMode"]
    └─► OLLAMA_MODE env var
    └─► Default: "local"

get_ollama_api_key()
    │
    ├─► secrets/ollama_cloud_config.json["api_key"]
    └─► OLLAMA_CLOUD_API_KEY env var

get_ollama_cloud_endpoint()
    │
    ├─► settings.json["ollamaCloudEndpoint"]
    ├─► secrets/ollama_cloud_config.json["endpoint"]
    ├─► OLLAMA_CLOUD_ENDPOINT env var
    └─► Default: "https://ollama.com"
```

---

## 5. Authentication Flow

### Cloud Mode Authentication

```
1. User provides API key
    │
    ▼
2. Save to secrets/ollama_cloud_config.json (permissions: 600)
    │
    ▼
3. get_ollama_api_key() retrieves key
    │
    ▼
4. get_ollama_client_headers() creates:
   {"Authorization": "Bearer <api_key>"}
    │
    ▼
5. Headers added to all API requests
    │
    ▼
6. Ollama Cloud validates API key
    │
    ├─► Valid → Process request
    └─► Invalid → Return 401/403 error
```

### Security Measures

- **API Key Storage**: `secrets/` directory (not in settings.json)
- **File Permissions**: 600 (rw-------) enforced
- **Transmission**: Only in Authorization header (HTTPS)
- **Logging**: API keys redacted from all logs
- **Validation**: Endpoint must be HTTPS, validated for SSRF

---

## 6. Error Handling and Fallback

### Auto Mode Fallback Logic

```python
# Pseudo-code
try:
    # Try cloud endpoint
    response = completion(api_base=cloud_endpoint, headers=cloud_headers)
    return response
except (APIConnectionError, Timeout):
    if fallback_endpoint:  # Auto mode
        logger.warning("Cloud failed, falling back to local")
        response = completion(api_base=local_endpoint, headers={})
        return response
    else:
        raise  # Cloud mode - no fallback
```

### Error Categories

1. **Connection Errors** (APIConnectionError)
   - Network unreachable
   - DNS failure
   - Connection refused
   - **Action**: Fallback to local (auto mode)

2. **Timeout Errors** (Timeout)
   - Request timeout (>300s for completions)
   - **Action**: Fallback to local (auto mode)

3. **Authentication Errors** (401, 403)
   - Invalid API key
   - Insufficient permissions
   - **Action**: Return error (no fallback)

4. **Server Errors** (500+)
   - Ollama Cloud service issues
   - **Action**: Fallback to local (auto mode)

5. **Client Errors** (400, 404)
   - Invalid request
   - Model not found
   - **Action**: Return error (no fallback)

---

## 7. Security Architecture

### Security Layers

1. **Configuration Security**
   - API keys in `secrets/` (not committed to git)
   - File permissions: 600
   - Environment variable support

2. **Network Security**
   - HTTPS enforcement (cloud endpoints)
   - TLS certificate verification
   - SSRF protection (endpoint validation)

3. **Authentication Security**
   - Bearer token authentication
   - API key redaction in logs
   - Secure header transmission

4. **Error Security**
   - API keys redacted from error messages
   - Generic error messages for security-sensitive failures
   - No sensitive data in logs

### Threat Model

| Threat | Mitigation |
|--------|------------|
| API key exposure | Stored in secrets/, redacted from logs |
| Man-in-the-middle | HTTPS/TLS enforced |
| SSRF attacks | Endpoint validation, private IP blocking |
| Credential theft | File permissions (600), not in settings.json |
| Data leakage | Local mode option, user control |

---

## 8. Component Interactions

### Core Components

1. **ollama_config.py**
   - Configuration management
   - Endpoint resolution
   - API key management
   - URL validation
   - Fallback logic

2. **rag_core.py**
   - LLM completion calls
   - Fallback implementation
   - Error handling

3. **llm_client.py**
   - Async client wrapper
   - Header injection
   - Concurrency control

4. **factory.py**
   - Backend factory
   - Model listing
   - Endpoint selection

### Integration Points

```
UI/Settings
    │
    ▼
REST API (rest_server.py)
    │
    ├─► Save config → ollama_config.save_ollama_cloud_config()
    │
    ├─► Test connection → ollama_config.test_cloud_connection()
    │
    └─► Get mode → ollama_config.get_ollama_mode()
         │
         ▼
    Backend (factory.py)
         │
         ▼
    RAG Core (rag_core.py)
         │
         ├─► Get endpoint → ollama_config.get_ollama_endpoint_with_fallback()
         │
         └─► Completion → litellm.completion() with headers
```

---

## 9. Performance Considerations

### Latency

- **Local Mode**: ~10-100ms (no network)
- **Cloud Mode**: ~100-2000ms (network + processing)
- **Auto Mode**: Cloud latency + fallback delay if needed

### Timeouts

- **Health Checks**: 5 seconds
- **Completions**: 300 seconds (5 minutes)
- **Model Listing**: 5 seconds

### Retry Strategy

- **No Automatic Retries**: Failures trigger fallback (auto mode) or error
- **Fallback**: Immediate fallback to local on connection/timeout errors
- **User Retry**: Users can retry failed requests manually

---

## 10. Monitoring and Observability

### Logging

- **Mode Changes**: Logged when mode is set
- **Fallback Events**: Warning logged when fallback occurs
- **Connection Failures**: Error logged (API keys redacted)
- **Configuration Saves**: Info logged (paths only, no secrets)

### Metrics (Future)

- Cloud vs local usage
- Fallback frequency
- Error rates by endpoint
- Latency comparisons

---

## 11. Extension Points

### Adding New Modes

1. Extend `OllamaMode` type in `ollama_config.py`
2. Add endpoint resolution logic
3. Update `get_ollama_endpoint_with_fallback()`
4. Update UI to support new mode

### Custom Endpoints

- Supported via `ollamaCloudEndpoint` setting
- Must pass HTTPS and SSRF validation
- Must support Ollama API format

### Authentication Methods

- Currently: Bearer token (API key)
- Future: OAuth2, API key rotation, etc.

---

## 12. Migration and Backward Compatibility

### Existing Deployments

- **Default Mode**: "local" (maintains current behavior)
- **Settings Migration**: Auto-detects existing `apiEndpoint`
- **No Breaking Changes**: All existing code continues to work

### Upgrade Path

1. Install update
2. System detects existing `apiEndpoint`
3. Sets `ollamaMode` to "local" automatically
4. User can optionally enable cloud mode via UI

---

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `src/core/ollama_config.py` | Configuration management, endpoint resolution, API key handling |
| `src/core/rag_core.py` | LLM completion calls with fallback |
| `src/core/llm_client.py` | Async client wrapper |
| `src/core/factory.py` | Backend factory, model listing |
| `config/settings.json` | User configuration (mode, endpoints) |
| `secrets/ollama_cloud_config.json` | API key storage (secure) |

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-27


