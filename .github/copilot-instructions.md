# AI Agent Instructions for Agentic RAG

## Architecture Overview

**Agentic RAG** is a retrieval-augmented generation system supporting multiple LLM backends (Ollama local/cloud, OpenAI Assistants, Google Vertex AI). It combines pgvector-based vector search, document indexing, and LangChain extractors with FastAPI REST + MCP servers and a React/Vite UI.

### Core Components

1. **Backend Abstraction** ([src/core/factory.py](../src/core/factory.py), [src/core/interfaces.py](../src/core/interfaces.py))
   - `RAGBackend` protocol: unified interface for all backends
   - `OllamaBackend`: local/cloud Ollama via [src/core/rag_core.py](../src/core/rag_core.py)
   - `RemoteBackend`: HTTP client to REST API
   - `HybridBackend`: combines local and remote backends with fallback
   - `GoogleGeminiBackend`, `OpenAIAssistantsBackend`: cloud providers
   - Runtime switching via REST API `/api/config/mode` (no restart required)

2. **Canonical Content & Index** ([src/core/rag_core.py](../src/core/rag_core.py), [src/core/document_repo.py](../src/core/document_repo.py))
   - Canonical indexed text artifacts: stored under `cache/indexed/` (env `RAG_INDEXED_DIR`)
   - Vector index: PostgreSQL + pgvector built from sentence-transformers embeddings ([src/core/embeddings.py](../src/core/embeddings.py))
   - Index rebuild: `rebuild_index()` rebuilds pgvector from `cache/indexed/` artifacts

3. **Servers**
   - **REST API** ([src/servers/rest_server.py](../src/servers/rest_server.py)): FastAPI on port 8001, serves UI + backend operations
   - **MCP Server** ([src/servers/mcp_server.py](../src/servers/mcp_server.py)): FastMCP on port 8000, model context protocol for AI agents
   - Both servers log to [log/](../log/) directory

4. **UI** ([ui/src/App.tsx](../ui/src/App.tsx))
   - React + Vite + shadcn/ui on port 5173
   - Features: chat, search, file manager, settings, backend switching, conversation history
   - State managed in App.tsx, communicates with REST API

### Data Flow

```
UI (React) → REST API (FastAPI) → Factory → Backend (Ollama/OpenAI/Google)
                                          ↓
               Indexed Artifacts (cache/indexed) ← pgvector Index (Postgres)
```

Chat persistence: SQLite via [src/core/chat_store.py](../src/core/chat_store.py) with sessions and messages tables.

## Configuration

- **Primary**: [config/settings.json](../config/settings.json) - runtime config (API endpoints, model names, temperature, debug mode)
- **Environment**: `.env` file (fallback, loaded by start.py)
- **Settings priority**: settings.json > environment variables
- **Ollama Config**: [src/core/ollama_config.py](../src/core/ollama_config.py) handles local/cloud/auto modes
  - Cloud mode: API key in `ollamaCloudApiKey`, endpoint in `ollamaCloudEndpoint`
  - Auto mode: tries cloud first, falls back to local on failure
- **CA Bundle Support**: [src/core/config_paths.py](../src/core/config_paths.py) - `caBundlePath` in settings.json for corporate proxies

## Critical Workflows

### Starting Services

**Command**: `python start.py [--role monolith|server|client]`
- **monolith** (default): starts all services (Ollama, MCP, REST, UI)
- **server**: backend only (skips UI)
- **client**: UI only (skips Ollama, assumes remote backend)

**Process**:
1. Creates/activates virtualenv at `.venv/`
2. Installs requirements from [requirements.txt](../requirements.txt)
3. Starts services in order: Ollama → MCP → REST → UI
4. Logs to [log/](../log/) directory (ollama_server.log, mcp_server.log, rest_server.log, ui_server.log)
5. Graceful shutdown on SIGINT/SIGTERM

**Flags**: `--skip-ollama`, `--skip-mcp`, `--skip-rest`, `--skip-ui`, `--recreate-venv`, `--no-browser`

### Testing

**Command**: `.venv/bin/pytest tests/`
- Config: [pytest.ini](../pytest.ini) (log_cli_level = INFO)
- Fixtures: [tests/conftest.py](../tests/conftest.py)
- Key tests: [tests/test_rag_core.py](../tests/test_rag_core.py), [tests/test_ollama_cloud_integration.py](../tests/test_ollama_cloud_integration.py)

### Adding/Switching Backends

1. Create class implementing `RAGBackend` protocol in [src/core/](../src/core/)
2. Register in [src/core/factory.py](../src/core/factory.py) `get_rag_backend()`
3. Add config model to [src/core/models.py](../src/core/models.py)
4. Update REST API endpoints in [src/servers/rest_server.py](../src/servers/rest_server.py)
5. Add UI settings panel in [ui/src/features/settings/SettingsView.tsx](../ui/src/features/settings/SettingsView.tsx)

## Tech Stack & Conventions

### Stack Mapping (Node.js → Python)
When converting or implementing features:
- **Framework**: Node.js → FastAPI
- **Validation**: Zod/Joi → Pydantic
- **ORM**: Prisma/TypeORM → SQLAlchemy (Async) or Tortoise-ORM
- **Testing**: Jest → Pytest
- **Package Manager**: NPM/Yarn → Poetry/Pip

### Python Code Style
- **Formatting**: Black + isort (see [.cursor/rules/lean-code.mdc](../.cursor/rules/lean-code.mdc))
- **Functions**: ≤50 lines, early returns over nested conditionals
- **Type hints**: Required for clarity (enforced by @QA role)
- **Docstrings**: One-liners unless more detail essential (enforced by @Scribe role)
- **Imports**: Standard lib > Third-party > Local, no unused imports

### Security & Compliance (SOC 2 + GDPR)
**CRITICAL**: All code must follow [.cursor/rules/global-compliance-requirements.mdc](../.cursor/rules/global-compliance-requirements.mdc)
- **Secrets**: Never commit API keys, tokens, passwords. Use environment variables or config files (gitignored)
- **Logging**: Redact secrets in logs (see `_redact_api_key()` in [src/core/ollama_config.py](../src/core/ollama_config.py))
- **Personal Data**: No PII in logs, define retention/deletion, document third-party transfers
- **TLS/HTTPS**: All external API calls must use HTTPS
- **Timeouts**: All HTTP requests must have timeouts (see [src/core/rag_core.py](../src/core/rag_core.py) httpx usage)

### Backend Integration Patterns
- **LLM Calls**: Use [src/core/llm_client.py](../src/core/llm_client.py) wrappers (`sync_completion`, `safe_completion`) for unified error handling
- **Config Reload**: Call `reload_llm_config()` after settings changes
- **Ollama Endpoint**: Use `ollama_config.get_ollama_endpoint()` for mode-aware endpoint resolution
- **API Keys**: Mask in responses using `_redact_api_key()`

### React/UI Patterns
- **Functional components** with hooks (no class components)
- **shadcn/ui**: Use existing components from [ui/src/components/ui/](../ui/src/components/ui/)
- **API calls**: Via REST endpoints at `http://localhost:8001/api/`
- **Toast notifications**: Use `toast()` from sonner
- **State**: Lifted to App.tsx for cross-feature sharing

## Common Tasks

### Debugging Indexing Issues
- Check [log/rest_server.log](../log/rest_server.log) for errors
- Run [tests/debug_indexing.py](../tests/debug_indexing.py) to inspect store
- Verify pgvector index rebuild: `curl -X POST http://localhost:8001/api/rebuild-index`

### Adding New Models
- **Ollama**: Update `model` in settings.json, pull via `ollama pull <model>`
- **OpenAI**: Update `openaiConfig.model` via REST API `/api/config/openai`
- **Google**: Update vertex config via `/api/config/vertex`

### Document Extraction
- **Supported formats**: PDF, DOCX, TXT, JSON, MD via [src/core/extractors.py](../src/core/extractors.py)
- **LangChain loaders**: PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
- **Custom extractors**: Add to `extract_text_from_file()` switch statement

## Key Files Reference

| File | Purpose |
|------|---------|
| [start.py](../start.py) | Multi-platform service launcher with role-based deployment |
| [src/core/rag_core.py](../src/core/rag_core.py) | Core RAG logic: indexing, search, rerank, grounding, verification |
| [src/core/factory.py](../src/core/factory.py) | Backend factory + HybridBackend orchestration |
| [src/core/ollama_config.py](../src/core/ollama_config.py) | Ollama local/cloud/auto mode management |
| [src/servers/rest_server.py](../src/servers/rest_server.py) | FastAPI REST endpoints + health checks + Prometheus metrics |
| [src/servers/mcp_server.py](../src/servers/mcp_server.py) | FastMCP server for AI agent integration |
| [config/settings.json](../config/settings.json) | Runtime configuration (API endpoints, models, debug mode) |
| [docs/models-and-configuration.md](../docs/models-and-configuration.md) | Backend switching guide |
| [docs/ollama-cloud-architecture.md](../docs/ollama-cloud-architecture.md) | Cloud integration architecture |

## AI Agent Team Roles & Workflows

This project uses persona-based code review with automated governance workflows ([.cursor/rules/](../.cursor/rules/)):

### Agent Personas
- **@Architect**: System design, trust boundaries, data flows, SOC 2/GDPR by design, availability
- **@Sentinel**: Security, secrets, privacy enforcement, technical SOC 2/GDPR controls, TLS/HTTPS validation
- **@QA**: Tests, lint, processing integrity, behavioral GDPR (deletion/export correctness)
- **@Scribe**: Docs, naming, auditability, lawful basis documentation, commit message quality
- **@BackendEngineer**: Python logic conversion, functional equivalence, idiomatic code
- **@FrontendDev**: React/React Native UX-first implementation, platform conventions

### Compliance Workflows
- **`/preflight`** ([.cursor/commands/preflight.md](../.cursor/commands/preflight.md)): Pre-commit review by all roles against staged changes
  - Any role returning `VERDICT: BLOCK` prevents commit
  - Enforces SOC 2 + GDPR at code level before changes are committed
- **`/commit`** ([.cursor/commands/commit.md](../.cursor/commands/commit.md)): Automated commit with council review + message generation
  - Runs @CommitAuditCouncil workflow
  - Generates 80-char imperative commit message (no emojis)
  - Only commits if all roles return `APPROVE`

### Non-Negotiables
1. **Type Hints**: Every Python function argument and return must be typed
2. **Async/Await**: Properly handle Python's `asyncio` event loop, no blocking main thread
3. **Security Gate**: No code finalized until @Sentinel verifies input sanitization and secret handling
4. **Docstrings**: All public modules, classes, and methods must have docstrings
5. **Git Hygiene**: Clean `.gitignore`, no IDE config dirs (.vscode/, .idea/) or build artifacts committed
6. **GDPR Self-Check** (before finalizing code):
   - No personal data in logs
   - No client-side personal data without justification
   - New data collection has clear purpose + secure transmission
   - No raw personal data in analytics/telemetry
